#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "utfcpp/utf8.h"

constexpr int INF = 1e9;

template <typename T>
T** malloc_2d(size_t m, size_t n)
{
    T* data_arr = new T[n * m];
    T** ptrs = new T*[m];
    for (size_t i = 0; i < m; i++)
    {
        ptrs[i] = data_arr + (i * n);
    }
    return ptrs;
}

template <typename T>
void delete_2d(T **array)
{
    delete[] array[0];
    delete[] array;
}

std::string join(const std::vector<std::string>& words) {
    size_t total_length = 0;
    for (const auto& s : words) {
        total_length += s.size();
    }

    std::string result;
    result.reserve(total_length);

    for (size_t i = 0; i < words.size(); ++i) {
        result.append(words[i]);
    }

    return result;
}

// 模糊字符串匹配函数
std::tuple<float, int, int> fuzzy_match(
    const std::vector<std::string> &raw_words,
    const std::string &raw_pattern
) {
    std::vector<char32_t> pattern, text;
    std::vector<std::vector<char32_t>> words(raw_words.size());
    pattern.reserve(raw_pattern.length() / 2);
    std::string raw_text = join(raw_words);
    text.reserve(raw_text.length() / 2);
    utf8::utf8to32(raw_pattern.begin(), raw_pattern.end(), std::back_inserter(pattern));
    utf8::utf8to32(raw_text.begin(), raw_text.end(), std::back_inserter(text));

    for (size_t i = 0; i < raw_words.size(); i++)
    {
        words[i].reserve(raw_words[i].length() / 2);
        utf8::utf8to32(raw_words[i].begin(), raw_words[i].end(), std::back_inserter(words[i]));
    }

    size_t m = pattern.size(), n = text.size();
    int **dp = malloc_2d<int>(m + 1, n + 1);

    for (size_t i = 0; i <= m; i++)
    {
        dp[i][0] = i;
    }
    for (size_t j = 0; j <= n; j++)
    {
        dp[0][j] = 0;
    }

    for (size_t i = 1; i <= m; i++)
    {
        for (size_t j = 1; j <= n; j++)
        {
            dp[i][j] = std::min(
                dp[i - 1][j - 1] + (pattern[i - 1] != text[j - 1]),
                std::min(dp[i - 1][j], dp[i][j - 1]) + 1
            );
        }
    }

    int best_match = INF;
    size_t l = 0, end_idx = 0;
    for (size_t k = 0, idx = 0; k <= n && idx < words.size(); k += words[idx].size(), idx++)
    {
        if (dp[m][k] < best_match)
        {
            best_match = dp[m][k];
            l = k;
            end_idx = idx;
        }
    }

    if (best_match > (int) m)
    {
        return std::make_tuple(0.f, 0, 0);
    }

    float best_score = 100. * (float) (m - best_match) / m;
    size_t start_idx = 0, i, j;
    for (i = m, j = l; i > 0 && j > 0;)
    {
        if (dp[i - 1][j - 1] <= dp[i][j - 1] && dp[i - 1][j - 1] <= dp[i - 1][j])
        {
            i--;
            j--;
        }
        else if (dp[i - 1][j] <= dp[i][j - 1])
        {
            i--;
        }
        else
        {
            j--;
        }
    }

    for (size_t idx = 0, i = 0; idx < words.size(); i += words[idx].size(), idx++)
    {
        if (i <= j && j < i + words[idx].size())
        {
            start_idx = idx;
            break;
        }
    }

    delete_2d(dp);

    return std::make_tuple(best_score, start_idx, end_idx - 1);
}

// pybind11 的绑定代码
// PYBIND11_MODULE 宏创建了一个名为 example 的 Python 模块
// "m" 是一个 py::module_ 类型的变量，代表这个模块
PYBIND11_MODULE(fast_match, m) {
    m.doc() = "Faster fuzzy matching for Python"; // 可选的模块文档字符串

    // m.def() 将 C++ 函数 "add" 暴露给 Python
    // 第一个参数 "add" 是 Python 中调用的函数名
    // 第二个参数是 C++ 函数的指针 &add
    // 第三个参数是函数的文档字符串
    m.def("fast_fuzzy_match", &fuzzy_match, "Fast matching main function");
}