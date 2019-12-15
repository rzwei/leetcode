#include <numeric>
#include <future>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <set>
#include <queue>
#include <assert.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <thread>
#include <functional>
#include <mutex>
#include <string>
using namespace std;

class NumMatrix {
public:
    vector<vector<int>> sums;
    int n, m;
    NumMatrix(vector<vector<int>> a) {
        n = a.size();
        if (n == 0)
        {
            a.push_back({ 0 });
            return;
        }
        m = a[0].size();
        sums.assign(n + 1, vector<int>(m + 1));
        for (int i = 1; i <= n; ++i)
        {
            int cur = 0;
            for (int j = 1; j <= m; ++j)
            {
                cur += a[i - 1][j - 1];
                sums[i][j] = sums[i - 1][j] + cur;
            }
        }
    }

    int sumRegion(int row1, int col1, int row2, int col2) {
        return sums[row2 + 1][col2 + 1] - sums[row2 + 1][col1] - sums[row1][col2 + 1] + sums[row1][col1];
    }
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

//1286. Iterator for Combination
class CombinationIterator {
    string s;
    int n, len;
    vector<int> pos;
    bool valid = true;
public:
    CombinationIterator(string s, int n) : pos(len, 0), s(s), n(n), len(s.size()) {
        sort(s.begin(), s.end());
        for (int i = 0; i < n; ++i)
        {
            pos[i] = 1;
        }
    }

    string next() {
        string ret;
        ret.clear();
        for (int i = 0; i < len; ++i)
        {
            if (pos[i])
            {
                ret.push_back(s[i]);
            }
        }

        int i = len - 1;
        int cnt = 0;
        valid = true;
        if (pos[i] == 1)
        {
            for (; i >= 0 && pos[i] == 1; --i)
            {
                cnt++;
            }
            if (cnt == n)
            {
                valid = false;
            }
            for (int j = len - 1; j > i; --j)
            {
                pos[j] = 0;
            }
            for (; i >= 0; --i)
            {
                if (pos[i] == 1)
                {
                    pos[i] = 0;
                    pos[i + 1] = 1;
                    for (int j = 0; j < cnt; ++j)
                    {
                        pos[i + 2 + j] = 1;
                    }
                    break;
                }
            }
        }
        else
        {
            for (; i >= 0 && pos[i] == 0; --i);
            pos[i] = 0;
            pos[i + 1] = 1;
        }
        return ret;
    }

    bool hasNext() {
        return valid;
    }
};

class Solution
{
    //1288. Remove Covered Intervals
    int removeCoveredIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());
        int last = 0, ans = 1;
        for (int i = 0; i < intervals.size(); i++) {
            if (intervals[last][0] <= intervals[i][0] &&
                intervals[i][1] <= intervals[last][1])
                continue;
            ans++;
            last = i;
        }
        return ans;
    }

    //1289. Minimum Falling Path Sum II
    int minFallingPathSum(vector<vector<int>>& a) {
        int n = a.size(), m = a[0].size();
        vector<vector<int>> dp(n, vector<int>(m, INT_MAX));
        for (int j = 0; j < m; ++j) dp[0][j] = a[0][j];
        for (int i = 1; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                for (int k = 0; k < m; ++k)
                {
                    if (j == k) continue;
                    dp[i][j] = min(dp[i][j], dp[i - 1][k] + a[i][j]);
                }
            }
        }
        int ans = INT_MAX;
        for (int j = 0; j < m; ++j)
        {
            ans = min(ans, dp[n - 1][j]);
        }
        return ans;
    }
    
    //1287. Element Appearing More Than 25% In Sorted Array
    int findSpecialInteger(vector<int>& a) {
        int n = a.size();
        for (int i = 0; i < n; )
        {
            int v = a[i];
            int cnt = 0;
            while (i < n && a[i] == v)
            {
                cnt++;
                i++;
            }
            if (cnt > n / 4.0)
                return v;
        }
        return -1;
    }


    //1290. Convert Binary Number in a Linked List to Integer
	int getDecimalValue(ListNode* head) {
		int ans = 0;
		while (head)
		{
			ans = ans * 2 + head->val;
			head = head->next;
		}
		return ans;
	}

    vector<int> solve_len_5124(int n, int low, int high)
    {
        vector<int> ans;
        for (int i = 1; i + n - 1 < 10; ++i)
        {
            int t = 0;
            for (int j = 0; j < n; ++j)
            {
                t = t * 10 + i + j;
            }
            if (low <= t && t <= high)
                ans.push_back(t);
        }
        return ans;
    }
    //1291. Sequential Digits
    vector<int> sequentialDigits(int low, int high) {
        vector<int> ans;
        for (int l = 1; l <= 9; ++l)
        {
            for (auto& e : solve_len_5124(l, low, high))
            {
                ans.push_back(e);
            }
        }
        return ans;
    }

    //1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
	int maxSideLength(vector<vector<int>>& mat, int t) {
		int n = mat.size(), m = mat[0].size();
		NumMatrix nm(mat);
		int l = 0, r = max(n, m) + 1;
		while (l < r)
		{
			int mid = (l + r) / 2;
			bool valid = false;
			for (int i = 0; i + mid - 1 < n; ++i)
			{
				for (int j = 0; j + mid - 1 < m; ++j)
				{
					int val = nm.sumRegion(i, j, i + mid - 1, j + mid - 1);
					if (val <= t)
					{
						valid = true;
					}
				}
			}
			if (!valid)
			{
				r = mid;
			}
			else
			{
				l = mid + 1;
			}
		}
		return l - 1;
	}

    /*
    //1293. Shortest Path in a Grid with Obstacles Elimination
    int dr[] = { 0, 1, 0, -1 };
    int dc[] = { 1, 0, -1, 0 };

    int const maxn = 40 + 1;
    int dp[maxn][maxn][maxn * maxn];
    class Solution {
    public:
        int n, m;

        int dfs(int i, int j, int left, vector<vector<int>>& g)
        {
            if (i == n - 1 && j == m - 1) return 0;
            int& ans = dp[i][j][left];
            if (ans != -1) return ans;
            ans = INT_MAX / 4;
            for (int d = 0; d < 4; ++d)
            {
                int ni = i + dr[d], nj = j + dc[d];
                if (0 <= ni && ni < n && 0 <= nj && nj < m)
                {
                    if (g[ni][nj] == 1 && left > 0)
                    {
                        ans = min(ans, dfs(ni, nj, left - 1, g) + 1);
                    }
                    else if (g[ni][nj] == 0)
                    {
                        ans = min(ans, dfs(ni, nj, left, g) + 1);
                    }
                }
            }
            return ans;
        }
        int shortestPath(vector<vector<int>>& grid, int k) {
            if (grid[0][0] == 1)
            {
                if (k == 0) return -1;
                k--;
                grid[0][0] = 0;
            }
            memset(dp, -1, sizeof(dp));
            n = grid.size(), m = grid[0].size();
            int ret = dfs(0, 0, k, grid);
            if (ret == INT_MAX / 4) return -1;
            return ret;
        }
    };
    */

    int minFallingPathSum(vector<vector<int>>& arr) {

    }
};
int main()
{

}
