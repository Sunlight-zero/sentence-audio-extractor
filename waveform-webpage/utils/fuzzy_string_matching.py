from typing import Tuple, List
from fast_match import fast_fuzzy_match


def fuzzy_match(words: List[str], pattern: str) -> Tuple[float, int, int]:
    """
    寻找与字符串 text 最匹配的文本串 pattern，并返回相应的字符串位置和分数
    返回值为 (l, r, score)，其中 $l, r$ 是左闭右闭的，即 words[l:(r+1)] 是句子全部单词
    """
    return fast_fuzzy_match(words, pattern)
