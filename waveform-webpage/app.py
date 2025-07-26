# app.py

import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import webbrowser
import traceback
# --- 关键修正 1: 从 werkzeug.utils 导入 secure_filename ---
from werkzeug.utils import secure_filename

# 根据您上传的文件名，确保 import 正确
import sentence_matching as sm

# --- Flask 应用初始化与配置 ---
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- 全局任务状态 ---
current_task = {
    "status": "idle",
    "video_path": None,
    "sentence": None,
    "initial_start": 0.0,
    "initial_end": 0.0,
    "clip_source_path": None,
    "error": None
}

# --- API 端点定义 ---

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload_and_start_task', methods=['POST'])
def upload_and_start_task():
    global current_task
    if current_task["status"] == "processing":
        return jsonify({"error": "一个任务正在处理中，请稍候。"}), 409

    if 'videoFile' not in request.files:
        return jsonify({"error": "请求中未包含视频文件部分。"}), 400
    
    file = request.files['videoFile']
    sentence = request.form.get('sentence')

    if file.filename == '' or not sentence:
        return jsonify({"error": "未选择文件或缺少目标句子。"}), 400

    # --- 关键修正 2: 直接调用导入的 secure_filename 函数 ---
    filename = secure_filename(file.filename)
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(saved_path)
    print(f"文件已上传并保存到: {saved_path}")

    current_task = { "status": "processing", "video_path": saved_path, "sentence": sentence }

    def task_runner():
        global current_task
        try:
            result = sm.find_sentence_timestamps(
                audio_path=saved_path,
                target_sentence=sentence,
                model_path_or_size="D:/ACGN/gal/whisper/models/faster-whisper-medium",
                device="cuda",
                compute_type="float16",
                confidence_threshold=80,
                search_mode='exhaustive',
                enable_source_separation=True,
                demucs_models_path=None,
                export_transcription_path="full_transcription.txt",
                clip_vocals_only=False
            )
            if result:
                current_task.update({
                    "status": "pending_correction",
                    "initial_start": result['start_time'],
                    "initial_end": result['end_time'],
                    "clip_source_path": result['clip_source_path']
                })
            else:
                current_task.update({"status": "error", "error": "未能在视频中找到匹配的句子或处理失败。"})
        except Exception:
            current_task.update({"status": "error", "error": f"后台发生未知错误: {traceback.format_exc()}"})

    threading.Thread(target=task_runner).start()
    return jsonify({"message": "文件上传成功，任务已开始处理。"})

@app.route('/get_task_result', methods=['GET'])
def get_task_result():
    global current_task
    if current_task["status"] == "pending_correction":
        video_filename = os.path.basename(current_task["video_path"])
        return jsonify({
            "status": current_task["status"],
            "sentence": current_task["sentence"],
            "video_url": f"/videos/{video_filename}",
            "initial_start": current_task["initial_start"],
            "initial_end": current_task["initial_end"],
        })
    else:
        return jsonify({"status": current_task["status"], "error": current_task.get("error")})

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/save_correction', methods=['POST'])
def save_correction():
    global current_task
    data = request.json
    final_start, final_end = data.get('start_time'), data.get('end_time')

    if final_start is None or final_end is None:
        return jsonify({"error": "请求中缺少时间戳。"}), 400

    try:
        base_name = os.path.splitext(os.path.basename(current_task["video_path"]))[0]
        output_filename = f"{base_name}_clipped.wav"
        output_path = os.path.join("output", output_filename)
        os.makedirs("output", exist_ok=True)

        sm.finalize_clip(
            source_path=current_task["clip_source_path"],
            start_time=float(final_start),
            end_time=float(final_end),
            output_path=output_path
        )
        
        print(f"成功裁剪并保存文件到: {output_path}")
        current_task = {"status": "done"}
        return jsonify({"message": "校对成功，音频已裁剪。", "output_path": output_path})
    except Exception as e:
        return jsonify({"error": f"裁剪音频时出错: {e}"}), 500

if __name__ == '__main__':
    # 引入 waitress
    from waitress import serve
    
    # 检查是否是主进程，避免在重载时重复打开浏览器
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:5000")
    
    # 使用 waitress.serve 启动应用，它比 'flask run' 稳定得多
    print("--- 启动 Waitress 生产服务器 ---")
    serve(app, host="127.0.0.1", port=5000)
