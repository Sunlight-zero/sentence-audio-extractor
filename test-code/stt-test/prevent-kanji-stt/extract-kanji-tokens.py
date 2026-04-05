import base64
import os
import re

# multilingual.tiktoken ファイルのパス
token_file_path = os.path.join(os.path.dirname(__file__), "./multilingual.tiktoken")

# 漢字が含まれているか判定する関数
def contains_kanji(text):
    for char in text:
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF or  # CJK Unified Ideographs
            0x3400 <= code <= 0x4DBF or  # CJK Unified Ideographs Extension A
            0xF900 <= code <= 0xFAFF or  # CJK Compatibility Ideographs
            0x20000 <= code <= 0x2EBEF   # CJK Unified Ideographs Extension B-F
        ):
            return True
    return False

# multilingual.tiktokenファイルを読み込み、漢字と「起始字节」Token 一并提取
def extract_suppress_tokens(file_path):
    suppress_ids = []
    
    # 导致汉字生成的 UTF-8 起始字节范围：
    # 3字节汉字起始: 0xE4 - 0xEF
    # 4字节汉字起始: 0xF0 - 0xF4
    kanji_start_bytes = set(range(0xE4, 0xF5))
    
    print(f"解析 {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2: continue
            base64_token, token_id = parts[0], int(parts[1])
            token_bytes = base64.b64decode(base64_token)
            
            try:
                # 尝试解码为 utf-8 字符串
                decoded_text = token_bytes.decode('utf-8')
                if contains_kanji(decoded_text):
                    suppress_ids.append(token_id)
            except UnicodeDecodeError:
                # 如果解码失败，说明这是一个字节片段（micro-token）
                # 检查它是否是汉字的起始字节
                if len(token_bytes) == 1:
                    byte_val = token_bytes[0]
                    if byte_val in kanji_start_bytes:
                        suppress_ids.append(token_id)
                        
    return sorted(list(set(suppress_ids)))

# トークンIDをファイルに書き出す
def save_tokens_to_file(tokens, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        # 使用空格分隔，更符合 Whisper 的 suppress_tokens 参数格式，且文件体积更小
        file.write(" ".join(map(str, tokens)))

# 提取所有需要拦截的 Token
all_suppress_ids = extract_suppress_tokens(token_file_path)

# 保存到 kanji_tokens.txt
output_path = os.path.join(os.path.dirname(__file__), "suppress_tokens.txt")
save_tokens_to_file(all_suppress_ids, output_path)

print(f"成功提取 {len(all_suppress_ids)} 个 Tokens（包含完整汉字及起始字节）。")
print(f"已保存到 '{output_name}'。")
