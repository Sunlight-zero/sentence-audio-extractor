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
from typing import Dict, List, Any, Optional
import re
# --- 【核心修正】导入 hashlib 模块用于计算哈希值 ---
import hashlib
import shutil

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

def extract_sentences_from_anki(
        path: str = "sentences",
        deck_name: Optional[str] = None,
        field_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
    """
    从指定的 Anki 牌组中提取没有音频的例句。
    参数为空时强制读取 anki_settings.json 中的配置。
    """
    settings = load_anki_settings()
    search_config = settings["search_config"]
    
    deck_name = deck_name or search_config["deck_name"]
    field_names = field_names or search_config["field_names"]

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

        for field_name in field_names:
            if field_name in fields and fields[field_name]['value']:
                break
        else:
            continue # 继续 notes_info 的循环
        
        sentence = fields[field_name]['value']
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


def load_anki_settings() -> Dict[str, Any]:
    """
    加载全局 Anki 设置。如果文件不存在或格式错误，将抛出异常。
    """
    settings_path = os.path.join(os.path.dirname(__file__), "anki_settings.json")
    
    if not os.path.exists(settings_path):
        example_path = settings_path.replace(".json", ".example.json")
        if os.path.exists(example_path):
             print(f"[提示] 未找到配置文件，已自动根据模板创建: {os.path.basename(settings_path)}")
             shutil.copy(example_path, settings_path)
        else:
             raise FileNotFoundError(f"找不到 Anki 配置文件且未发现模板: {settings_path}\n请确保 utils 目录下存在 anki_settings.json。")
    
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)

def change_note_type(note_id: int, target_type: str) -> None:
    """
    根据 JSON 配置修改笔记的模板类型。
    规则存储在 utils/anki_note_mappings/{target_type}.json
    """
    # 1. 寻找转换规则文件
    mapping_dir = os.path.join(os.path.dirname(__file__), "anki_note_mappings")
    mapping_file = os.path.join(mapping_dir, f"{target_type}.json")
    
    if not os.path.exists(mapping_file):
        print(f"  - [警告] 未找到模板 '{target_type}' 的配置文件: {mapping_file}，已跳过")
        return

    with open(mapping_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    config_note_type = config.get("note_type")
    if config_note_type != target_type:
        print(f"  - [警告] 配置文件 '{mapping_file}' 中的 note_type 与目标类型不匹配，已跳过")
        return
    
    target_fields_list = config.get("target_fields", [])
    field_mapping = config.get("field_mapping", {})

    # 2. 获取旧笔记信息
    note_info_list = invoke("notesInfo", notes=[note_id])
    if not note_info_list:
        print(f"  - [错误] 找不到笔记 ID: {note_id}")
        return
    old_note = note_info_list[0]

    # 3. 构建新字段字典
    # 初始化所有目标字段为空字符串
    new_fields = {field: "" for field in target_fields_list}
    
    # 根据映射从旧笔记中提取值
    old_fields_data = old_note.get("fields", {})
    for old_name, target_name in field_mapping.items():
        if old_name in old_fields_data:
            new_fields[target_name] = old_fields_data[old_name]["value"]

    # 4. 执行 API 调用
    try:
        invoke(
            "updateNoteModel",
            note={
                "id": note_id,
                "modelName": target_type,
                "fields": new_fields,
            }
        )
        print(f"  - [成功] 笔记 {note_id} 类型已转换为 {target_type}")
    except Exception as e:
        print(f"  - [失败] 转换笔记 {note_id} 类型时出错: {e}")

def upload_clips_to_anki(
        clips_data: List[Dict[str, Any]], 
        final_deck: Optional[str] = None,
        target_note_type: Optional[str] = None
    ):
    """
    将确认后的音频片段上传到 Anki 并移动卡片。
    参数为空时强制读取 anki_settings.json 中的配置。
    """
    settings = load_anki_settings()
    upload_config = settings["upload_config"]
    conversion_config = settings.get("note_type_conversion", {})

    final_deck = final_deck or upload_config["final_deck"]
    source_note_type = upload_config["source_note_type"]
    audio_field_name = upload_config["audio_field_name"]
    
    # 如果函数参数没传 target_note_type，则检查配置中是否启用了自动转换
    if target_note_type is None and conversion_config.get("enabled"):
        target_note_type = conversion_config.get("target_note_type")

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
        
        note_info = invoke("notesInfo", notes=[note_id])[0]
        # 使用配置文件中的源笔记类型进行校验
        if note_info.get("modelName") != source_note_type:
            print(f"  - [跳过] 笔记类型 '{note_info.get('modelName')}' 不是 '{source_note_type}'。")
            skipped_count += 1
            continue

        try:
            if ',' in audio_base64:
                pure_base64_data = audio_base64.split(',', 1)[1]
            else:
                pure_base64_data = audio_base64

            # --- 使用音频内容的 SHA256 哈希值作为文件名 ---
            # sentence_prefix = sentence[:20]
            # 1. 从 Base64 字符串解码为二进制数据
            audio_bytes = base64.b64decode(pure_base64_data)
            # 2. 计算 SHA256 哈希值
            sha256_hash = hashlib.sha256(audio_bytes).hexdigest()
            # 3. 创建新的文件名
            expected_audio_filename = f"{sha256_hash}.mp3"

            store_result = invoke(
                'storeMediaFile', 
                filename=expected_audio_filename, 
                data=pure_base64_data
            )
            audio_filename: str = store_result
            print(f"  - [成功] 音频文件 '{audio_filename}' 已上传。")
            
            update_payload = {
                "note": {
                    "id": note_id,
                    "fields": {
                        audio_field_name: f"[sound:{audio_filename}]"
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
            
            if target_note_type is not None:
                change_note_type(note_id, target_note_type)

        except Exception as e:
            print(f"  - [错误] 处理笔记 {note_id} 时发生错误: {e}")
            skipped_count += 1

    print("\n--------------------")
    print("Anki 上传处理完成！")
    print(f"总计成功处理: {processed_count} 个笔记")
    print(f"总计跳过/失败: {skipped_count} 个笔记")
