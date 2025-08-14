from openai import OpenAI
import json
import re
from typing import Optional, Dict, List
import pathlib
import time
import threading
from collections import deque


# --- 新增：速率限制器 ---
class RateLimiter:
    """
    一个线程安全的速率限制器，用于控制API调用频率。
    """
    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period = period_seconds
        self.lock = threading.Lock()
        self.timestamps = deque()

    def acquire(self):
        """
        获取一个调用许可。如果达到速率限制，则阻塞直到可以继续。
        """
        with self.lock:
            while True:
                current_time = time.monotonic()
                # 移除超出时间窗口的旧时间戳
                while self.timestamps and self.timestamps[0] <= current_time - self.period:
                    self.timestamps.popleft()

                if len(self.timestamps) < self.max_calls:
                    self.timestamps.append(current_time)
                    return
                
                # 计算需要等待的时间
                oldest_call_time = self.timestamps[0]
                wait_time = oldest_call_time + self.period - current_time
                if wait_time > 0:
                    time.sleep(wait_time)

# --- 【核心修改】将速率限制调整为每 120 秒 10 次 ---
GEMINI_RATE_LIMITER = RateLimiter(10, 120)


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
    # 新增：应用速率限制
    GEMINI_RATE_LIMITER.acquire()
    
    client, model_name = load_llm_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    # 【核心修改】允许直接传递一个完整的消息列表
    elif isinstance(prompt, list):
        messages += prompt
    else:
        raise NotImplementedError(f"不支持的 prompt 类型：{type(prompt)}")

    try:
        # 【核心修改】直接使用构建好的 messages 列表
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

# --- 新增：标准化相关函数 ---
def _extract_hiragana(text: str) -> set:
    """从字符串中提取所有平假名字符。"""
    return set(re.findall(r'[ぁ-ん]', text))

def llm_normalize(
    texts: List[str], 
    max_retries: int = 3
) -> List[str]:
    """
    【多轮对话优化版】使用 LLM 将日文文本列表高效标准化为平假名。
    通过模拟对话来强化规则和输出格式，以减少幻觉。
    """
    # 1. 智能筛选：找出需要转换的词及其索引
    words_to_convert = []
    has_kanji_or_katakana = re.compile(r'[一-龯ァ-ン]')
    for i, text in enumerate(texts):
        if has_kanji_or_katakana.search(text):
            words_to_convert.append({"index": i, "word": text})

    if not words_to_convert:
        return [re.sub(r'[^ぁ-ん]', '', text) for text in texts]

    # 2. 【核心修改】将所有规则和角色定义放入 System Prompt
    system_prompt = (
        "You are a highly precise Japanese linguistic processor. Your task is to provide the correct hiragana readings for a given list of words, based on the full sentence context provided.\n"
        "RULES:\n"
        "1. Your response MUST be a single valid JSON object with a key `\"results\"`.\n"
        "2. The `\"results\"` value must be an array of objects.\n"
        "3. Each object must have two keys: `\"index\"` (the original index) and `\"output\"` (the resulting hiragana string).\n"
        "4. The number of objects in the `\"results\"` array must exactly match the number of objects in the input `words_to_convert` array."
    )
    
    # 3. 【核心修改】构建多轮对话历史作为 Few-shot 示例
    # 示例的用户输入
    example_context = "今日は放課後みんなで練習する日だから、スコア忘れないようにしないと"
    example_words_to_convert = [
        {"index": 0, "word": "今日"}, {"index": 2, "word": "放課後"},
        {"index": 6, "word": "練習"}, {"index": 8, "word": "日"},
        {"index": 11, "word": "スコア"}, {"index": 12, "word": "忘れ"}
    ]
    example_user_prompt = (
        f"CONTEXT: \"{example_context}\"\n"
        f"WORDS TO CONVERT: {json.dumps(example_words_to_convert, ensure_ascii=False)}"
    )
    # 示例的助手（模型）应答
    example_assistant_response = json.dumps(
        {
            "results": [
                {"index": 0, "output": "きょう"}, {"index": 2, "output": "ほうかご"},
                {"index": 6, "output": "れんしゅう"}, {"index": 8, "output": "ひ"},
                {"index": 11, "output": "すこあ"}, {"index": 12, "output": "わすれ"}
            ]
        }, ensure_ascii=False
    )

    # 实际任务的用户输入
    full_context_paragraph = "".join(texts)
    actual_user_prompt = (
        f"CONTEXT: \"{full_context_paragraph}\"\n"
        f"WORDS TO CONVERT: {json.dumps(words_to_convert, ensure_ascii=False)}"
    )

    # 将上述内容组合成一个对话历史
    conversation_history = [
        {"role": "user", "content": example_user_prompt},
        {"role": "assistant", "content": example_assistant_response},
        {"role": "user", "content": actual_user_prompt}
    ]

    for attempt in range(max_retries):
        print(f"LLM 多轮对话标准化尝试 ({attempt + 1}/{max_retries})...")
        try:
            # 4. 【核心修改】调用 llm_query，传入完整的对话历史
            raw_response = llm_query(
                conversation_history, # 传入整个对话列表
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
                
            normalized_parts = response_data["results"]

            if not isinstance(normalized_parts, list) or len(normalized_parts) != len(words_to_convert):
                print(f"LLM 返回了格式不匹配的 JSON（'results' 长度不符），回复：{normalized_parts}，正在重试...")
                continue
            
            # 5. 本地重组 (逻辑不变)
            reconstructed_list = list(texts) 
            for part in normalized_parts:
                if isinstance(part, dict) and "index" in part and "output" in part:
                    original_index = part["index"]
                    hiragana_output = part["output"]
                    if 0 <= original_index < len(reconstructed_list):
                        reconstructed_list[original_index] = hiragana_output
                else:
                    raise ValueError("LLM 响应中的条目格式不正确")

            print(f"LLM 多轮对话标准化成功。")
            cleaned_results = [re.sub(r'[^ぁ-ん]', '', text) for text in reconstructed_list]
            return cleaned_results

        except json.JSONDecodeError:
            print(f"LLM 返回的不是有效的 JSON，正在重试... 回复: {raw_response}")
        except Exception as e:
            print(f"LLM 标准化过程中出现未知错误: {e}，正在重试...")

    raise Exception(f"LLM 在 {max_retries} 次尝试后仍无法完成标准化。")
