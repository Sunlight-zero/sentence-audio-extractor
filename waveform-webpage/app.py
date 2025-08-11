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
# --- 新增 ---
# 导入新的 Anki 处理模块
import utils.anki_handler as anki

# --- Flask 应用初始化与目录配置 ---
app = Flask(__name__, static_folder='static', static_url_path='')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
RESULTS_FOLDER = os.path.join(APP_ROOT, 'results')
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'output')
# --- 新增 ---
# SENTENCES_FOLDER 用于存放从 Anki 拉取的句子和 ID 映射
SENTENCES_FOLDER = os.path.join(APP_ROOT, 'sentences')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SENTENCES_FOLDER, exist_ok=True) # --- 新增 ---
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# --- 新增 ---
app.config['SENTENCES_FOLDER'] = SENTENCES_FOLDER

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

# --- 新增 API: 从 Anki 获取句子 ---
@app.route('/api/anki/sentences', methods=['GET'])
def get_anki_sentences():
    """从 AnkiConnect 获取需要处理的句子。"""
    try:
        deck_name = request.args.get('deck', 'luna temporary') # 允许前端指定牌组
        sentences_path = os.path.join(app.config['SENTENCES_FOLDER'], 'anki_sentences')
        result = anki.extract_sentences_from_anki(path=sentences_path, deck_name=deck_name)
        return jsonify({
            "success": True,
            "sentences": result["sentences_text"],
            "message": f"成功从牌组 '{deck_name}' 提取 {len(result['id_to_sentence_map'])} 个句子。"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# --- 修改 API: 处理音视频文件 ---
@app.route('/process', methods=['POST'])
def process_videos():
    """接收一个或多个视频和多个句子，启动后台批量分析任务。"""
    # --- 修改: 使用 getlist 处理多个文件 ---
    video_files = request.files.getlist('videoFiles')
    if not video_files or all(f.filename == '' for f in video_files):
        return jsonify({"error": "缺少视频文件或未选择文件。"}), 400
    
    sentences_text = request.form.get('sentence', '')
    sentences = [s.strip() for s in sentences_text.splitlines() if s.strip()]
    if not sentences:
        return jsonify({"error": "目标句子不能为空。"}), 400

    enable_separation = request.form.get('separateVocals') == 'on'

    video_paths = []
    for video_file in video_files:
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        video_paths.append(video_path)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "message": "任务已开始，正在初始化..."}

    # --- 修改: 将 video_paths (列表) 传递给后台线程 ---
    thread = threading.Thread(target=run_processing_task, args=(task_id, video_paths, sentences, enable_separation))
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
            return jsonify({
                "message": "音频裁剪成功！", 
                "download_url": url_for('serve_output', filename=output_filename)
            })
        else:
            return jsonify({"error": "后端裁剪音频失败。"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"裁剪时发生服务器错误: {e}"}), 500

# --- 新增 API: 上传校对好的音频到 Anki ---
@app.route('/api/anki/upload', methods=['POST'])
def upload_to_anki():
    """接收前端校对好的片段数据，并调用 anki_handler 上传。"""
    try:
        clips_to_upload = request.json.get('clips')
        if not clips_to_upload:
            return jsonify({"success": False, "error": "没有提供可上传的片段。"}), 400
        
        # 这里可以启动一个新线程来执行上传，避免长时间阻塞请求
        # 但为简单起见，我们先同步执行
        anki.upload_clips_to_anki(clips_to_upload)
        
        return jsonify({"success": True, "message": "上传任务已成功提交到 Anki。"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

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
# 后台任务函数 (核心修改处)
# ==============================================================================
def run_processing_task(task_id, video_paths, sentences, enable_separation):
    """在后台线程中运行的完整处理流程，实现“找到即删除”逻辑。"""
    try:
        def progress_callback(message):
            if task_id in tasks:
                tasks[task_id]["message"] = message
        
        sentence_map_path = os.path.join(app.config['SENTENCES_FOLDER'], 'anki_sentences.json')
        sentence_to_note_id = {}
        if os.path.exists(sentence_map_path):
            with open(sentence_map_path, 'r', encoding='utf-8') as f:
                id_to_sentence = json.load(f)
                sentence_to_note_id = {v: k for k, v in id_to_sentence.items()}
        
        # --- 核心修改 1: 使用 set 来存储待处理的句子，方便高效移除 ---
        remaining_sentences = set(sentences)
        all_clips_results = []
        total_videos = len(video_paths)

        for i, video_path in enumerate(video_paths, 1):
            # --- 核心修改 2: 如果所有句子都找到了，就提前结束循环 ---
            if not remaining_sentences:
                progress_callback("所有句子都已找到匹配项，提前结束处理。")
                break

            original_filename = os.path.basename(video_path)
            progress_callback(f"处理视频 {i}/{total_videos}: {original_filename} (待匹配: {len(remaining_sentences)}句)")
            
            # --- 核心修改 3: 只把“待处理”的句子传递给匹配函数 ---
            results_data = sm.find_multiple_sentences_timestamps(
                audio_path=video_path,
                target_sentences=list(remaining_sentences), # 将 set 转换为 list
                model_path_or_size="medium",
                device="cuda",
                compute_type="float16",
                confidence_threshold=70,
                search_mode='exhaustive',
                enable_source_separation=enable_separation,
                clip_vocals_only=False,
                progress_callback=lambda msg: progress_callback(f"[视频 {i}/{total_videos}] {msg}")
            )
            
            if results_data and results_data.get('clips'):
                clip_source_path = results_data.get('clip_source_path')
                found_in_this_video = set()

                for clip in results_data.get('clips', []):
                    clip['clip_source_path'] = clip_source_path
                    clip['original_video_filename'] = original_filename
                    clip['video_url'] = f"/uploads/{original_filename}"
                    clip['note_id'] = sentence_to_note_id.get(clip['sentence'])
                    all_clips_results.append(clip)
                    
                    # --- 核心修改 4: 记录下本次找到的句子 ---
                    found_in_this_video.add(clip['sentence'])
                
                # --- 核心修改 5: 从待处理集合中，移除本次已经找到的句子 ---
                if found_in_this_video:
                    remaining_sentences -= found_in_this_video
                    progress_callback(f"视频 {original_filename} 中新找到 {len(found_in_this_video)} 句。")


        if not all_clips_results:
            tasks[task_id] = {"status": "completed", "message": "处理完成，但未能从任何视频中匹配到任何句子。", "result_json_url": None}
            return

        json_filename = f"results_{task_id[:8]}.json"
        json_path = os.path.join(app.config['RESULTS_FOLDER'], json_filename)
        
        final_json_data = {
            "clips": all_clips_results
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_data, f, ensure_ascii=False, indent=4)

        tasks[task_id] = {
            "status": "completed",
            "message": f"处理完成！从 {total_videos} 个视频中总共匹配到 {len(all_clips_results)} 个句子。",
            "result_json_url": f"/results/{json_filename}",
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
