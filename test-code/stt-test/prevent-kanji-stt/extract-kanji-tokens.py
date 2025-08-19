import base64

# multilingual.tiktoken ファイルのパス
token_file_path = "./multilingual.tiktoken"

# 漢字が含まれているか判定する関数
def contains_kanji(text):
    for char in text:
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF or  # CJK統合漢字（基本漢字） U+4E00～U+9FFF
            0x3400 <= code <= 0x4DBF or  # CJK統合漢字拡張A U+3400～U+4DBF
            0xF900 <= code <= 0xFAFF     # CJK互換漢字 U+F900～U+FAFF
        ):
            return True
    return False

# multilingual.tiktokenファイルを読み込み、デコードしてトークンIDを取得する
def load_tokens(file_path):
    tokens = []
    found_kanji = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            base64_token, token_id = line.strip().split()
            decoded_token = base64.b64decode(base64_token).decode('utf-8', errors='ignore')
            tokens.append((decoded_token, int(token_id)))
    return tokens


# 漢字を含むトークンIDだけを抽出
def get_kanji_only_token_ids(tokens):
    kanji_token_ids = []
    for token, token_id in tokens:
        if contains_kanji(token):
            kanji_token_ids.append(token_id)
    return kanji_token_ids

# トークンIDをファイルに書き出す
def save_tokens_to_file(tokens, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for token_id in tokens:
            file.write(f"{token_id}\n")

# トークンを読み込む
tokens = load_tokens(token_file_path)

# 漢字トークンだけ取得
kanji_token_ids = get_kanji_only_token_ids(tokens)

# ファイルに保存
save_tokens_to_file(kanji_token_ids, "kanji_tokens.txt")

print("漢字を含むトークンIDが 'kanji_tokens.txt' に保存されました。")
