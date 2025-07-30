# app.py
import os
import json
import uuid
import traceback
from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from waitress import serve
import threading

# 确保 utils 目录在 Python 路径中
import utils.sentence_matching as sm

# --- Flask 应用初始化与目录配置 ---
app = Flask(__name__, static_folder='static', static_url_path='')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
RESULTS_FOLDER = os.path.join(APP_ROOT, 'results')
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 使用字典来跟踪后台任务状态
tasks = {}

# ==============================================================================
# 页面服务路由
# ==============================================================================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/proofread')
def proofread_page():
    return send_from_directory('static', 'proofread.html')

# ==============================================================================
# API 端点
# ==============================================================================
@app.route('/process', methods=['POST'])
def process_video():
    """接收视频和多个句子，启动后台批量分析任务。"""
    if 'videoFile' not in request.files:
        return jsonify({"error": "缺少视频文件。"}), 400
    
    sentences_text = request.form.get('sentence', '')
    sentences = [s.strip() for s in sentences_text.splitlines() if s.strip()]
    if not sentences:
        return jsonify({"error": "目标句子不能为空。"}), 400

    # 【修改】获取前端音源分离复选框的状态
    # 如果复选框被勾选，request.form.get('separateVocals') 的值会是 'on'。
    enable_separation = request.form.get('separateVocals') == 'on'

    video_file = request.files['videoFile']
    if video_file.filename == '':
        return jsonify({"error": "未选择文件。"}), 400

    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "message": "任务已开始，正在初始化..."}

    # --- 关键修改: 将 enable_separation 传递给后台线程 ---
    thread = threading.Thread(target=run_processing_task, args=(task_id, video_path, sentences, filename, enable_separation))
    thread.start()

    return jsonify({"status": "processing", "task_id": task_id})

@app.route('/task_status/<task_id>')
def get_task_status(task_id):
    """获取指定后台任务的状态。"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found", "error": "任务ID不存在。"}), 404
    return jsonify(task)

@app.route('/clip', methods=['POST'])
def clip_audio():
    """接收校对后的时间戳，执行最终裁剪。"""
    data = request.json
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    clip_source_path = data.get('clip_source_path')
    original_video_filename = data.get('original_video_filename')
    sentence_id = data.get('sentence_id', 'clip')

    if None in [start_time, end_time, clip_source_path, original_video_filename]:
        return jsonify({"error": "请求中缺少必要的参数。"}), 400

    if not os.path.exists(clip_source_path):
         return jsonify({"error": f"裁剪源文件未找到: {clip_source_path}"}), 404

    try:
        base_name = os.path.splitext(original_video_filename)[0]
        output_filename = f"{base_name}_{sentence_id}.wav"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        success = sm.finalize_clip(
            source_path=clip_source_path,
            start_time=float(start_time),
            end_time=float(end_time),
            output_path=output_path
        )
        
        if success:
            # 在请求上下文中安全地使用 url_for
            return jsonify({
                "message": "音频裁剪成功！", 
                "download_url": url_for('serve_output', filename=output_filename)
            })
        else:
            return jsonify({"error": "后端裁剪音频失败。"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"裁剪时发生服务器错误: {e}"}), 500

# ==============================================================================
# 文件服务路由
# ==============================================================================
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# ==============================================================================
# 后台任务函数
# ==============================================================================
def run_processing_task(task_id, video_path, sentences, original_filename, enable_separation):
    """在后台线程中运行的完整处理流程。"""
    try:
        def progress_callback(message):
            if task_id in tasks:
                tasks[task_id]["message"] = message

        # 【修改】
        # 1. enable_source_separation: 使用前端传入的布尔值。
        # 2. clip_vocals_only: 硬编码为 False，确保无论是否分离，
        #    后续的 clip_source_path 始终指向原始文件。
        results_data = sm.find_multiple_sentences_timestamps(
            audio_path=video_path,
            target_sentences=sentences,
            model_path_or_size="medium",
            device="cuda",
            compute_type="float16",
            confidence_threshold=70,
            search_mode='exhaustive',
            enable_source_separation=enable_separation,
            clip_vocals_only=False,
            progress_callback=progress_callback
        )
        
        if results_data is None:
            tasks[task_id] = {"status": "error", "error": "处理失败，后台任务发生严重错误。"}
            return

        # --- 关键修改: 彻底移除 url_for，手动构造相对路径 ---
        json_filename = f"{os.path.splitext(original_filename)[0]}_{task_id[:8]}.json"
        json_path = os.path.join(app.config['RESULTS_FOLDER'], json_filename)
        
        final_json_data = {
            "original_video_filename": original_filename,
            "video_url": f"/uploads/{original_filename}",
            "clip_source_path": results_data.get('clip_source_path'),
            "clips": results_data.get('clips', [])
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_data, f, ensure_ascii=False, indent=4)

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
    import multiprocessing
    multiprocessing.freeze_support()
    
    host = "127.0.0.1"
    port = 5000
    print(f"--- 服务器正在启动 ---")
    print(f"请在浏览器中打开: http://{host}:{port}")
    serve(app, host=host, port=port)