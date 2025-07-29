# app.py
import os
import json
import uuid
import traceback
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from waitress import serve
import threading

# 导入核心处理逻辑
import utils.sentence_matching as sm

# --- Flask 应用初始化与目录配置 ---
app = Flask(__name__, static_folder='static', static_url_path='')

# --- 关键修正: 使用绝对路径确保路径的稳定性 ---
# 获取 app.py 文件所在的目录的绝对路径
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# 基于应用根目录构建文件夹的绝对路径
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
RESULTS_FOLDER = os.path.join(APP_ROOT, 'results')
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'output')

# 确保这些目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 将绝对路径存入 Flask 的配置中
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# --- 修正结束 ---

# 使用字典来跟踪后台任务状态，以任务ID为键
tasks = {}

# ==============================================================================
# 路由 - 页面服务
# ==============================================================================

@app.route('/')
def index():
    """服务处理页面 (index.html)"""
    return send_from_directory('static', 'index.html')

@app.route('/proofread')
def proofread_page():
    """服务校对页面 (proofread.html)"""
    return send_from_directory('static', 'proofread.html')

# ==============================================================================
# 路由 - API 端点
# ==============================================================================

@app.route('/process', methods=['POST'])
def process_video():
    """
    处理端点：接收视频和句子，启动后台分析任务。
    """
    if 'videoFile' not in request.files or not request.form.get('sentence'):
        return jsonify({"error": "缺少视频文件或目标句子。"}), 400

    video_file = request.files['videoFile']
    sentence = request.form.get('sentence')
    
    if video_file.filename == '':
        return jsonify({"error": "未选择文件。"}), 400

    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "message": "任务已开始，正在初始化..."}

    # 在后台线程中运行耗时的处理任务
    thread = threading.Thread(target=run_processing_task, args=(task_id, video_path, sentence, filename))
    thread.start()

    return jsonify({"status": "processing", "task_id": task_id})

@app.route('/task_status/<task_id>')
def get_task_status(task_id):
    """获取指定后台任务的状态和结果。"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found", "error": "任务ID不存在。"}), 404
    return jsonify(task)


@app.route('/clip', methods=['POST'])
def clip_audio():
    """
    裁剪端点：接收校对后的时间戳和文件信息，执行最终裁剪。
    """
    data = request.json
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    clip_source_path = data.get('clip_source_path')
    original_video_filename = data.get('original_video_filename')

    if None in [start_time, end_time, clip_source_path, original_video_filename]:
        return jsonify({"error": "请求中缺少必要的参数（时间戳或文件路径）。"}), 400

    if not os.path.exists(clip_source_path):
         return jsonify({"error": f"用于裁剪的源文件未找到: {clip_source_path}"}), 404

    try:
        base_name = os.path.splitext(original_video_filename)[0]
        # 使用更具描述性的文件名，并添加唯一标识符
        output_filename = f"{base_name}_{uuid.uuid4().hex[:6]}_clipped.wav"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        success = sm.finalize_clip(
            source_path=clip_source_path,
            start_time=float(start_time),
            end_time=float(end_time),
            output_path=output_path
        )
        
        if success:
            # 返回可供下载的音频文件URL
            return jsonify({
                "message": "音频裁剪成功！", 
                "download_url": f"/output/{output_filename}"
            })
        else:
            return jsonify({"error": "后端裁剪音频失败。"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"裁剪音频时发生服务器错误: {e}"}), 500

# ==============================================================================
# 路由 - 文件服务
# ==============================================================================

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """服务上传的视频文件。"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<path:filename>')
def serve_result(filename):
    """服务生成的JSON结果文件。"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/output/<path:filename>')
def serve_output(filename):
    """服务最终裁剪的音频文件。"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

# ==============================================================================
# 后台任务函数
# ==============================================================================

def run_processing_task(task_id, video_path, sentence, original_filename):
    """在后台线程中运行的完整处理流程。"""
    try:
        tasks[task_id]["message"] = "正在分析音频，此过程可能需要几分钟..."
        
        # 注意：这里的参数需要根据你的实际需求和环境进行配置
        result_data = sm.find_sentence_timestamps(
            audio_path=video_path,
            target_sentence=sentence,
            model_path_or_size="medium", # 推荐使用模型大小，而非固定路径
            device="cuda",
            compute_type="float16",
            confidence_threshold=50,
            search_mode='exhaustive',
            enable_source_separation=True,
            demucs_models_path=None,
            export_transcription_path=None, # 在Web服务中通常不直接导出文件
            clip_vocals_only=False
        )
        
        if not result_data:
            tasks[task_id] = {"status": "error", "error": "处理失败，未能在视频中找到匹配的句子。"}
            return

        # 准备要保存到JSON的完整数据
        json_filename = f"{os.path.splitext(original_filename)[0]}_{uuid.uuid4().hex[:6]}.json"
        json_path = os.path.join(app.config['RESULTS_FOLDER'], json_filename)
        
        final_json_data = {
            "status": "success",
            "sentence": sentence,
            "original_video_filename": original_filename,
            "video_url": f"/uploads/{original_filename}",
            "predicted_start": result_data.get('start_time'),
            "predicted_end": result_data.get('end_time'),
            "clip_source_path": result_data.get('clip_source_path'),
            "score": result_data.get('score', 'N/A') # 假设函数会返回分数
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_data, f, ensure_ascii=False, indent=4)

        # 任务完成，更新状态和结果
        tasks[task_id] = {
            "status": "completed",
            "message": "处理完成！",
            "result_json_url": f"/results/{json_filename}",
            "video_url": f"/uploads/{original_filename}"
        }

    except Exception as e:
        traceback.print_exc()
        tasks[task_id] = {"status": "error", "error": f"后台处理时发生严重错误: {e}"}


if __name__ == '__main__':
    host = "127.0.0.1"
    port = 5000
    print(f"--- 服务器正在启动 ---")
    print(f"请在浏览器中打开处理页面: http://{host}:{port}")
    serve(app, host=host, port=port)
