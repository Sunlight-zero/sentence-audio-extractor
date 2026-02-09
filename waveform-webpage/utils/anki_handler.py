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
        deck_name: str = "luna temporary",
        field_names: List[str] = ["example_sentence", "sentence1"]
    ) -> Dict[str, Any]:
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


def change_note_type(note_id: int, target_type: str) -> None:
    """修改笔记的模板类型（原地修改）"""

    if target_type != "word-multi-stcs":
        raise NotImplementedError(f"未知的目标笔记类型：{target_type}")

    def get_note_info(note_ids: int) -> Dict[str, Any]:
        """获取单个笔记信息"""
        return_notes_list = invoke("notesInfo", notes=[note_ids])
        assert len(return_notes_list) == 1, "获取的笔记数量不符！"
        return invoke("notesInfo", notes=[note_ids])[0]

    def extract_field(note: Dict[str, Any], field_name: str) -> str:
        """提取笔记中的字段值"""
        field = note["fields"].get(field_name)
        return field["value"] if field else ""

    def build_field_map() -> Dict[str, str]:
        """构建从源模板到目标模板的字段映射"""
        return {
            "word": "word",
            "rubytextHtml": "ruby",
            "audio_for_word": "audio",
            "etymology": "etymology",
            "illustration": "illustration",
            "example_sentence": "sentence1",
            "audio_for_example_sentence": "audio_sentence1",
            "screenshot": "screenshot1",
            "remarks": "remark1",
            "humanVoice": "allow_listen1",
            "source": "source1",
            "dictionaryInfo": "dictionaryInfo",
            "dictionaryContent": "dictionaryContent",
        }


    def build_target_fields(note: Dict[str, Any]) -> Dict[str, str]:
        """构建目标模板的字段内容"""
        # 初始化所有目标字段为空
        fields = {name: "" for name in [
            "word", "ruby", "audio", "etymology", "illustration",
            "sentence1", "audio_sentence1", "screenshot1", "remark1", "allow_listen1", "source1",
            "sentence2", "audio_sentence2", "screenshot2", "remark2", "allow_listen2", "source2",
            "sentence3", "audio_sentence3", "screenshot3", "remark3", "allow_listen3", "source3",
            "sentence4", "audio_sentence4", "screenshot4", "remark4", "allow_listen4", "source4",
            "sentence5", "audio_sentence5", "screenshot5", "remark5", "allow_listen5", "source5",
            "sentence6", "audio_sentence6", "screenshot6", "remark6", "allow_listen6", "source6",
            "dictionaryInfo", "dictionaryContent",
        ]}

        # 根据字段映射填充内容
        field_map = build_field_map()
        for source_field, target_field in field_map.items():
            fields[target_field] = extract_field(note, source_field)

        return fields

    note = get_note_info(note_id)
    fields = build_target_fields(note)
    
    # 准备字段映射，AnkiConnect 的 updateNoteModel 需要这个格式
    field_map = {}
    for source_field, target_field in build_field_map().items():
        if source_field in note["fields"]:
            field_map[source_field] = target_field

    # 使用 updateNoteModel 原地修改笔记类型
    invoke(
        "updateNoteModel",
        note={
            "id": note_id,
            "modelName": target_type,
            "fields": fields,
        }
    )

def upload_clips_to_anki(
        clips_data: List[Dict[str, Any]], 
        final_deck: str = "Japanese::temporary",
        target_note_type: Optional[str] = None
    ):
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
            
            if target_note_type is not None:
                change_note_type(note_id, target_note_type)
                print(f"  - [成功] 笔记 {note_id} 的类型已更新为 {target_note_type}。")

        except Exception as e:
            print(f"  - [错误] 处理笔记 {note_id} 时发生错误: {e}")
            skipped_count += 1

    print("\n--------------------")
    print("Anki 上传处理完成！")
    print(f"总计成功处理: {processed_count} 个笔记")
    print(f"总计跳过/失败: {skipped_count} 个笔记")
