#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

constexpr size_t MAXN = 2005;
int dp[MAXN][MAXN];

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string a, b;
    cin >> a >> b;

    size_t m = a.length(), n = b.length();

    for (size_t i = 0; i <= m; i++)
    {
        dp[i][0] = i;
    }
    for (size_t j = 0; j <= n; j++)
    {
        dp[0][j] = j;
    }

    for (size_t i = 1; i <= m; i++)
    {
        for (size_t j = 1; j <= n; j++)
        {
            if (a[i - 1] == b[j - 1])
            {
                dp[i][j] = dp[i - 1][j - 1];
            }
            else
            {
                dp[i][j] = min(dp[i - 1][j - 1], min(dp[i][j - 1], dp[i - 1][j])) + 1;
            }
        }
    }

    cout << dp[m][n] << "\n";
    return 0;
}
