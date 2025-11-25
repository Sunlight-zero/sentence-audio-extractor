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
    # 1. 构造目标 HTML 句子
    sentence_in_hypertext = ""
    cnt = 0
    has_kanji_or_katakana = re.compile(r'[一-龯ァ-ン]')
    words_to_convert = []
    for idx, word in enumerate(texts):
        if has_kanji_or_katakana.search(word):
            words_to_convert.append(word)
            sentence_in_hypertext += f'<span id="{idx}">{word}</span>'
        else:
            sentence_in_hypertext += word

    if len(words_to_convert) == 0:
        return [re.sub(r'[^ぁ-ん]', '', text) for text in texts] # 清除所有非平假名的对象

    system_prompt = (
        "You are a highly precise Japanese linguistic processor. Your mission is to identify words encapsulated in `<span id='...'>` tags within a given Japanese sentence and provide their correct hiragana readings based on the context.\n"
        "RULES:\n"
        "1. The user will provide a sentence where target words are marked like this: `<span id=\"N\">word</span>`, where 'N' is a unique, ascending integer ID.\n"
        "2. Your response MUST be a single, valid JSON object and nothing else. Do not add any explanatory text before or after the JSON.\n"
        "3. The JSON object must have a single root key: `\"results\"`.\n"
        "4. The value of `\"results\"` must be an array of objects.\n"
        "5. Each object in the array corresponds to exactly one `<span>` tag from the input sentence.\n"
        "6. Each object MUST contain exactly three keys:\n"
        "   - `\"id\"`: The integer ID extracted directly from the `id=\"N\"` attribute of the span tag.\n"
        "   - `\"word\"`: The original text content copied exactly from within the corresponding `<span>` tag.\n"
        "   - `\"hiragana\"`: The contextually appropriate hiragana reading for the `word`.\n"
        "7. The number of objects in the `\"results\"` array must EXACTLY match the total number of `<span>` tags in the user's input."
    )
    
    # 3. 【核心修改】构建多轮对话历史作为 Few-shot 示例
    # 示例的用户输入
    # example_context = '<span id="1">今日</span>は<span id="2">放課後</span>みんなで</span>は<span id="3">練習</span>する<span id="4">日</span>だから、スコア<span id="5">忘</span>れないようにしないと'
    # example_user_prompt = example_context
    # # 示例的助手（模型）应答
    # example_assistant_response = json.dumps(
    #     {
    #         "results": [
    #             {"id": 1, "word": "今日", "hiragana": "きょう"},
    #             {"id": 2, "word": "放課後", "hiragana": "ほうかご"},
    #             {"id": 3, "word": "練習", "hiragana": "れんしゅう"},
    #             {"id": 4, "word": "日", "hiragana": "ひ"},
    #             {"id": 5, "word": "忘", "hiragana": "わすれ"}
    #         ]
    #     }, ensure_ascii=False
    # )

    example_user_prompt = '京都で<span id="3">夏目</span><span id="4">漱</span><span id="5">石</span>と<span id="7">申す</span><span id="8">方</span>は、<span id="11">紅葉</span>の<span id="13">様子</span>を<span id="15">纏</span>めた<span id="17">報告</span>を、<span id="20">僅</span>か<span id="22">十分</span>で<span id="24">仕上げ</span>ると<span id="27">高言</span>した'
    example_assistant_response = json.dumps(
    {
        "results": [
                {"id": 3, "word": "夏目", "hiragana": "なつめ"},
                {"id": 4, "word": "漱", "hiragana": "そう"},
                {"id": 5, "word": "石", "hiragana": "せき"},
                {"id": 7, "word": "申す", "hiragana": "もうす"},
                {"id": 8, "word": "方", "hiragana": "かた"},
                {"id": 11, "word": "紅葉", "hiragana": "もみじ"},
                {"id": 13, "word": "様子", "hiragana": "ようす"},
                {"id": 15, "word": "纏", "hiragana": "まと"},
                {"id": 17, "word": "報告", "hiragana": "ほうこく"},
                {"id": 20, "word": "僅", "hiragana": "わず"},
                {"id": 22, "word": "十分", "hiragana": "じっぷん"},
                {"id": 24, "word": "仕上げ", "hiragana": "しあげ"},
                {"id": 27, "word": "高言", "hiragana": "こうげん"}
            ]
        }, ensure_ascii=False
    )

    # 实际任务的用户输入
    actual_user_prompt = (
        sentence_in_hypertext
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
                if isinstance(part, dict) and "id" in part and "word" in part and "hiragana" in part:
                    idx = part["id"]
                    original_word = part["word"]
                    if idx >= len(texts):
                        raise IndexError("LLM 返回 idx 的值出错，列表越界")
                    if texts[idx] != original_word:
                        raise ValueError("LLM 返回的 word 值与原词不一致")
                    hiragana = part["hiragana"]
                    if 0 <= idx < len(reconstructed_list):
                        reconstructed_list[idx] = hiragana
                else:
                    raise ValueError("LLM 响应中的条目格式不正确：列表元素不为 dict 或缺少相应的键")

            print(f"LLM 多轮对话标准化成功。")
            cleaned_results = [re.sub(r'[^ぁ-ん]', '', text) for text in reconstructed_list]
            return cleaned_results

        except json.JSONDecodeError:
            print(f"LLM 返回的不是有效的 JSON，正在重试... 回复: {raw_response}")
        except Exception as e:
            print(f"LLM 标准化过程中出现未知错误: {e}，正在重试...")

    raise Exception(f"LLM 在 {max_retries} 次尝试后仍无法完成标准化。")
