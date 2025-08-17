# utils/anki_handler.py
"""
该模块封装了所有与 AnkiConnect API 的交互逻辑，包括：
- 从 Anki 获取需要处理的笔记和句子。
- 将处理好的音频文件上传回 Anki 并更新对应的笔记字段。
"""

import requests
import json
import base64
import os
from typing import Dict, List, Any
import re
# --- 【核心修正】导入 hashlib 模块用于计算哈希值 ---
import hashlib

# AnkiConnect API 的 URL
ANKICONNECT_URL = "http://localhost:8765"

def invoke(action, **params):
    """
    一个辅助函数，用于向 AnkiConnect 发送请求。
    """
    payload = {"action": action, "version": 6, "params": params}
    try:
        response = requests.post(ANKICONNECT_URL, data=json.dumps(payload))
        response.raise_for_status()
        response_json = response.json()
        if response_json.get('error'):
            if "collection not available" in response_json.get('error', ''):
                 raise Exception("AnkiConnect 错误: Anki 配置文件当前正被使用或未完全加载。请稍等片刻或重启 Anki 后再试。")
            raise Exception(f"AnkiConnect 错误: {response_json['error']}")
        return response_json['result']
    except requests.exceptions.RequestException as e:
        raise Exception(f"无法连接到 AnkiConnect。请确保 Anki 正在运行并且 AnkiConnect 插件已安装。\n错误详情: {e}")

def extract_sentences_from_anki(path: str = "sentences", deck_name: str = "luna temporary") -> Dict[str, Any]:
    """
    从指定的 Anki 牌组中提取没有音频的例句。
    """
    print(f"正在查找 '{deck_name}' 牌组中所有笔记...")
    query = f'"deck:{deck_name}"'
    note_ids = invoke("findNotes", query=query)
    if not note_ids:
        print(f"在 '{deck_name}' 牌组中没有找到需要处理的笔记。")
        return {"sentences_text": "", "id_to_sentence_map": {}}

    print(f"找到了 {len(note_ids)} 个笔记。正在获取笔记详情...")
    notes_info = invoke('notesInfo', notes=note_ids)
    
    id_to_sentence_map = {}
    sentences_list = []
    
    for note in notes_info:
        note_id = note['noteId']
        fields = note['fields']
        
        if 'example_sentence' not in fields or not fields['example_sentence']['value']:
            continue
        
        sentence = fields['example_sentence']['value']
        sentence = sentence.replace("<b>", "").replace("</b>", "").replace("<br>", " ").strip()
        
        if sentence:
            sentences_list.append(sentence)
            id_to_sentence_map[str(note_id)] = sentence
    
    with open(path + ".json", mode='w', encoding="UTF-8") as json_file:
        json.dump(id_to_sentence_map, json_file, ensure_ascii=False, indent=4)
    
    sentences_text = "\n".join(sentences_list)
    with open(path + ".txt", mode='w', encoding="UTF-8") as file_to_save:
        file_to_save.write(sentences_text)
    
    print(f"已提取并保存所有句子到 {path}.txt，对应关系保存至 {path}.json")
    return {"sentences_text": sentences_text, "id_to_sentence_map": id_to_sentence_map}


def upload_clips_to_anki(clips_data: List[Dict[str, Any]], final_deck: str = "Japanese::temporary"):
    """
    将确认后的音频片段上传到 Anki 并移动卡片。
    """
    if not clips_data:
        raise ValueError("没有提供可上传的音频片段数据。")

    print("\n--- 开始上传音频至 Anki ---")
    processed_count = 0
    skipped_count = 0

    for clip in clips_data:
        note_id_str = clip.get('note_id')
        sentence = clip.get('sentence')
        audio_base64 = clip.get('audio_base64')
        original_video_filename = clip.get('original_video_filename', 'audio')

        if not all([note_id_str, sentence, audio_base64]):
            print(f"  - [警告] 缺少必要数据 (note_id, sentence, or audio_base64)，跳过一个片段。")
            skipped_count += 1
            continue
        
        try:
            note_id = int(note_id_str)
        except (ValueError, TypeError):
            print(f"  - [错误] 笔记 ID '{note_id_str}' 不是一个有效的整数，已跳过。")
            skipped_count += 1
            continue

        print(f"\n正在处理笔记 ID: {note_id} (句子: {sentence[:20]}...)")
        
        try:
            if ',' in audio_base64:
                pure_base64_data = audio_base64.split(',', 1)[1]
            else:
                pure_base64_data = audio_base64

            # --- 【核心修正】使用音频内容的 SHA256 哈希值作为文件名 ---
            # 1. 从 Base64 字符串解码为二进制数据
            audio_bytes = base64.b64decode(pure_base64_data)
            # 2. 计算 SHA256 哈希值
            sha256_hash = hashlib.sha256(audio_bytes).hexdigest()
            # 3. 创建新的文件名
            audio_filename = f"{sha256_hash}.wav"
            # --- 修正结束 ---

            invoke('storeMediaFile', filename=audio_filename, data=pure_base64_data)
            print(f"  - [成功] 音频文件 '{audio_filename}' 已上传。")
            
            update_payload = {
                "note": {
                    "id": note_id,
                    "fields": {
                        "audio_for_example_sentence": f"[sound:{audio_filename}]"
                    }
                }
            }
            invoke('updateNoteFields', **update_payload)
            print(f"  - [成功] 笔记 {note_id} 的音频字段已更新。")

            card_ids = invoke('findCards', query=f'nid:{note_id}')
            if card_ids:
                invoke('changeDeck', cards=card_ids, deck=final_deck)
                print(f"  - [成功] {len(card_ids)} 张关联卡片已移动到 '{final_deck}'。")
                processed_count += 1
            else:
                print(f"  - [警告] 未找到笔记 {note_id} 关联的卡片，无法移动。")
                skipped_count += 1

        except Exception as e:
            print(f"  - [错误] 处理笔记 {note_id} 时发生错误: {e}")
            skipped_count += 1

    print("\n--------------------")
    print("Anki 上传处理完成！")
    print(f"总计成功处理: {processed_count} 个笔记")
    print(f"总计跳过/失败: {skipped_count} 个笔记")
