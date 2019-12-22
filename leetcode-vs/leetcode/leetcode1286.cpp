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
	ListNode* next;
	ListNode(int x) : val(x), next(NULL) {}
};

class Solution
{
	//5283. Convert Binary Number in a Linked List to Integer
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
	//5124. Sequential Digits
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

	//5285. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
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
	//5286. Shortest Path in a Grid with Obstacles Elimination
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

	//5291. Find Numbers with Even Number of Digits
	int findNumbers(vector<int>& a) {
		int ans = 0;
		for (auto e : a)
		{
			int len = 0;
			while (e)
			{
				len++;
				e /= 10;
			}
			ans += (len % 2 == 0);
		}
		return ans;
	}

	//5292. Divide Array in Sets of K Consecutive Numbers
	bool isPossibleDivide(vector<int>& nums, int k) {
		map<int, int> cnt;
		for (auto& e : nums)
			cnt[e] ++;
		while (!cnt.empty())
		{
			int u = cnt.begin()->first;
			for (int i = 0; i < k; ++i)
			{
				int v = i + u;
				if (cnt.count(v))
				{
					if (--cnt[v] == 0)
					{
						cnt.erase(v);
					}
				}
				else
				{
					return false;
				}
			}
		}
		return true;
	}

	//5293. Maximum Number of Occurrences of a Substring
	int maxFreq(string s, int maxLetters, int minSize, int maxSize) {
		map<string, int> cnt;
		vector<int> win(26);
		int j = 0;
		int len = s.size();
		for (int i = 0; i < len; ++i)
		{
			vector<bool> vis(26);
			int cur = 0;
			for (int j = 0; j < 26 && i - j >= 0; ++j)
			{
				int k = i - j;
				if (vis[s[k] - 'a'] == 0) cur++;
				vis[s[k] - 'a'] = 1;

				if (cur > maxLetters) break;

				if (minSize <= i - k + 1 && i - k + 1 <= maxSize)
				{
					cnt[s.substr(k, i - k + 1)] ++;
				}
				else if (i - k + 1 > maxSize) break;
			}
		}
		string ans;
		int ans_cnt = 0;
		for (auto& e : cnt)
		{
			if (e.second > ans_cnt)
			{
				ans_cnt = e.second;
				ans = e.first;
			}
		}
		return ans_cnt;
	}


	//5294. Maximum Candies You Can Get from Boxes
	int maxCandies(vector<int>& status, vector<int>& candies, vector<vector<int>>& keys, vector<vector<int>>& containedBoxes, vector<int>& initialBoxes) {
		queue<int> q;
		for (auto& e : initialBoxes) q.push(e);
		int ans = 0;
		int n = status.size();

		vector<int> vis(n);     // hasCandies
		vector<int> hasKey(n);  // left keys
		vector<int> hasBox(n);  // need key

		while (!q.empty())
		{
			auto u = q.front(); q.pop();
			hasBox[u] = 1;

			if (!vis[u])
			{
				if (status[u] == 1)
				{
					ans += candies[u];
					vis[u] = 1;
				}
				else
				{
					hasBox[u] = 1;
				}
			}

			for (auto& box : containedBoxes[u])
			{
				hasBox[box] = 1;
			}

			for (auto& key : keys[u])
			{
				hasKey[key] = 1;
			}

			for (int box = 0; box < n; ++box)
			{
				if (hasBox[box] && status[box] && vis[box] == 0)
				{
					ans += candies[box];
					q.push(box);
					vis[box] = 1;
				}
			}

			for (int key = 0; key < n; ++key)
			{
				if (hasBox[key] && hasKey[key] && vis[key] == 0)
				{
					vis[key] = 1;
					q.push(key);

					ans += candies[key];
					hasBox[key] = 0;
				}
			}
		}
		return ans;
	}
};
int main()
{

}
