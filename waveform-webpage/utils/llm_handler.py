from openai import OpenAI
import json
import re
from typing import Optional, Dict, List
import pathlib


def load_llm_client():
    current_file_path = pathlib.Path(__file__)
    current_dir = current_file_path.parent
    llm_api_json_path = current_dir / "llm_api_key.json"

    with open(llm_api_json_path, "r") as file:
        api_info = json.load(file)
        api_url: str = api_info["api_url"]
        api_key: str = api_info["api_key"]
        model_name: str = api_info["model_name"]
    
    client = OpenAI(api_key=api_key, base_url=api_url)
    return client, model_name

def llm_query(prompt: str | list[Dict[str, str]], 
              system_prompt: Optional[str]=None,
              **kwargs
             ) -> Optional[str]:
    client, model_name = load_llm_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # 修正：正确处理不同类型的prompt并构建messages列表
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        messages.extend(prompt)
    else:
        raise NotImplementedError(f"不支持的 prompt 类型：{type(prompt)}")

    try:
        # 核心修改：将 response_format 和 temperature 等参数通过 kwargs 传入
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            **kwargs
        )
        content: str = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"尝试获取回复过程中出错: {e}")
        return None

def _extract_hiragana(text: str) -> set:
    """从字符串中提取所有平假名字符。"""
    return set(re.findall(r'[ぁ-ん]', text))

def llm_normalize(
    texts: List[str], 
    max_retries: int = 3, 
    validation_threshold: float = 0.8
) -> List[str]:
    """
    使用 LLM 将日文文本列表（被视为一个段落的多个词）标准化为平假名，并进行验证和重试。
    """
    system_prompt = (
        "You are a highly precise Japanese linguistic processor. Your task is to convert a list of Japanese words, "
        "which together form a coherent paragraph, into their correct hiragana readings. You must analyze the full "
        "paragraph context to determine the correct pronunciation for each word. Your output must be a single, "
        "valid JSON object."
    )
    
    # 将词语列表连接成一个完整的段落以提供上下文
    full_context_paragraph = "".join(texts)

    # --- Few-shot 示例 ---
    example_input_words = ["今日は", "、", "放", "課" ,"後", "みんな", "で", "練習", "する", "日", "だから", "、", "スコア", "忘れ", "ない", "よう", "に", "し", "ない", "と"]
    example_context = "".join(example_input_words)
    example_output_json = json.dumps(
        {
            "results": [
                {"input": "今日は", "output": "きょうは"}, {"input": "、", "output": "、"},
                {"input": "放", "output": "ほう"}, {"input": "課", "output": "か"},
                {"input": "後", "output": "ご"}, {"input": "みんな", "output": "みんな"},
                {"input": "で", "output": "で"}, {"input": "練習", "output": "れんしゅう"},
                {"input": "する", "output": "する"}, {"input": "日", "output": "ひ"},
                {"input": "だから", "output": "だから"}, {"input": "、", "output": "、"},
                {"input": "スコア", "output": "すこあ"}, {"input": "忘れ", "output": "わすれ"},
                {"input": "ない", "output": "ない"}, {"input": "よう", "output": "よう"},
                {"input": "に", "output": "に"}, {"input": "し", "output": "し"},
                {"input": "ない", "output": "ない"}, {"input": "と", "output": "と"}
            ]
        }, ensure_ascii=False, indent=2
    )

    for attempt in range(max_retries):
        print(f"LLM 标准化尝试 ({attempt + 1}/{max_retries})...")
        try:
            prompt_content = json.dumps(texts, ensure_ascii=False)
            full_prompt = (
                "Please convert each Japanese word in the `words_to_convert` array into its hiragana reading. "
                "Use the `full_context_paragraph` to determine the correct pronunciation for each word.\n\n"
                "RULES:\n"
                "1. Your response MUST be a single valid JSON object with a key `\"results\"` whose value is an array of objects.\n"
                "2. Each object in the array must have two keys: `\"input\"` (the original or corrected word) and `\"output\"` (the hiragana string).\n"
                "3. The number of objects in the `\"results\"` array must exactly match the `words_to_convert` array.\n"
                "4. The input words are from a speech-to-text system and may contain minor errors. Use the context to correct them. The `\"input\"` field in your output should contain the *corrected* word.\n"
                "5. Convert all Kanji and Katakana to hiragana. Preserve all original hiragana and punctuation in the `\"output\"` field.\n\n"
                "--- FEW-SHOT EXAMPLE ---\n"
                f"CONTEXT: \"{example_context}\"\n"
                f"INPUT WORDS: {json.dumps(example_input_words, ensure_ascii=False)}\n"
                f"EXPECTED JSON OUTPUT:\n{example_output_json}\n"
                "--- END EXAMPLE ---\n\n"
                "--- ACTUAL TASK ---\n"
                f"CONTEXT: \"{full_context_paragraph}\"\n"
                f"INPUT WORDS: {prompt_content}\n"
                "YOUR JSON OUTPUT:"
            )

            raw_response = llm_query(
                full_prompt, 
                system_prompt,
                response_format={"type": "json_object"},
                temperature=0.1
            )

            if not raw_response:
                print("LLM 返回空回复，正在重试...")
                continue

            response_data = json.loads(raw_response)
            
            if not isinstance(response_data, dict) or "results" not in response_data:
                print(f"LLM 返回的 JSON 结构不正确（缺少 'results' 键），正在重试...")
                continue
                
            normalized_pairs = response_data["results"]

            if not isinstance(normalized_pairs, list) or len(normalized_pairs) != len(texts):
                print(f"LLM 返回了格式不匹配的 JSON（'results' 不是列表或长度不符），正在重试...")
                continue
            
            correctly_processed_count = 0
            final_hiragana_list = []
            
            for i, pair in enumerate(normalized_pairs):
                # 检查基本结构
                if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
                    # 结构错误的直接跳过，不计入正确数
                    continue

                original_text = texts[i]
                llm_input = pair["input"]
                llm_output = pair["output"]
                
                # 条件1: LLM的输入字段必须与原文完全一致
                is_input_match = (original_text == llm_input)
                # 条件2: 原文中的假名必须全部出现在输出中
                original_hiragana = _extract_hiragana(original_text)
                is_hiragana_preserved = original_hiragana.issubset(_extract_hiragana(llm_output))

                if is_input_match and is_hiragana_preserved:
                    correctly_processed_count += 1
                
                final_hiragana_list.append(llm_output)

            # 在循环结束后，根据比例进行最终验证
            proportion = correctly_processed_count / len(texts)
            if proportion >= validation_threshold:
                print(f"LLM 标准化成功并通过验证 (一致率: {proportion:.2%})。")
                cleaned_results = [re.sub(r'[^ぁ-ん]', '', text) for text in final_hiragana_list]
                return cleaned_results
            else:
                print(f"验证失败：一致率 {proportion:.2%} 低于阈值 {validation_threshold:.2%}。正在重试...")
                # continue 会自动进入下一次循环

        except json.JSONDecodeError:
            print(f"LLM 返回的不是有效的 JSON，正在重试... 回复: {raw_response}")
        except Exception as e:
            print(f"LLM 标准化过程中出现未知错误: {e}，正在重试...")

    raise Exception(f"LLM 在 {max_retries} 次尝试后仍无法完成标准化。")
