from typing import Tuple

def fuzzy_match(text: str, pattern: str) -> Tuple[int, int, float]:
    """
    寻找与字符串 text 最匹配的文本串 pattern，并返回相应的字符串位置和分数
    返回值为 (l, r, score)，其中 $l, r$ 表示 text[l:r] 为最匹配的模板串
    """
    pass