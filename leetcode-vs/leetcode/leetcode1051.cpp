#include <bitset>
#include <set>
#include <queue>
#include <assert.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

template<typename T, typename V>
T first_less_than(T f, T l, V v)
{
    auto it = lower_bound(f, l, v);
    return it == f ? l : --it;
}


/*int memo[11][1 << 11];

class Solution {
public:
	int dfs(int u, vector<vector<int>> &wk, bitset<12> &vis, vector<vector<int>> &bk)
	{
		if (u == wk.size()) return 0;
		if (memo[u][vis.to_ulong()] != -1) return memo[u][vis.to_ulong()];
		int ans = INT_MAX / 2;
		for (int i = 0; i < bk.size(); ++i)
		{
			if (vis[i] == 0)
			{
				vis[i] = 1;
				int dist = abs(wk[u][0] - bk[i][0]) + abs(wk[u][1] - bk[i][1]);
				ans = min(ans, dfs(u + 1, wk, vis, bk) + dist);
				vis[i] = 0;
			}
		}
		return memo[u][vis.to_ulong()] = ans;
	}
	//1057. Campus Bikes II
	int assignBikes(vector<vector<int>>& workers, vector<vector<int>>& bikes) {
		bitset<12>vis;
		memset(memo, -1, sizeof(memo));
		return dfs(0, workers, vis, bikes);
	}
};
*/


/*
int memo[11][2][11][2];

class Solution {
public:
	int dfs(int i, int lim, int pre, int num, int d, vector<int> &a)
	{
		if (i == a.size())
		{
			return pre;
		}
		if (memo[i][lim][pre][num] != -1) return memo[i][lim][pre][num];
		int ans = 0;
		if (lim)
		{
			for (int v = 0; v <= a[i]; ++v)
			{
				int nx = (num | (v != 0));
				ans += dfs(i + 1, v == a[i], nx ? pre + (v == d) : 0, nx, d, a);
			}
		}
		else
		{
			for (int v = 0; v < 10; ++v)
			{
				int nx = (num | (v != 0));
				ans += dfs(i + 1, 0, nx ? pre + (v == d) : 0, nx, d, a);
			}
		}
		return memo[i][lim][pre][num] = ans;
	}

	//1 - v , count d
	int count(int v, int d)
	{
		vector<int> digits;
		while (v)
		{
			digits.push_back(v % 10);
			v /= 10;
		}
		if (digits.size() == 0) return 0;
		reverse(digits.begin(), digits.end());
		int ans = 0;
		memset(memo, -1, sizeof(memo));
		for (int i = 0; i <= digits[0]; ++i)
		{
			int nx = (i != 0);
			ans += dfs(1, digits[0] == i, nx ? i == d : 0, nx, d, digits);
		}
		return ans;
	}
	//1058. Digit Count in Range
	int digitsCount(int d, int low, int high) {
		return count(high, d) - count(low - 1, d);
	}
};
*/
class Solution
{
public:
	//1051. Height Checker
	int heightChecker(vector<int>& a) {
		int n = a.size();
		auto b = a;
		sort(a.begin(), a.end());
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			if (a[i] != b[i]) ans++;
		}
		return ans;
	}

	//1052. Grumpy Bookstore Owner
	int maxSatisfied(vector<int>& a, vector<int>& b, int x) {
		int n = a.size();
		vector<int> left(n), right(n);
		int u = 0;
		for (int i = 0; i < n; ++i)
		{
			left[i] = u;
			if (b[i] == 1) u += 0;
			else u += a[i];
		}
		u = 0;
		for (int i = n - 1; i >= 0; --i)
		{
			right[i] = u;
			if (b[i] == 1) u += 0;
			else u += a[i];
		}
		u = 0;
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			u += a[i];
			if (i >= x) u -= a[i - x];
			int v = u + right[i];
			if (i - x + 1 >= 0)
				v += left[i - x + 1];
			ans = max(ans, v);
		}
		return ans;
	}

	//1053. Previous Permutation With One Swap
	vector<int> prevPermOpt1(vector<int>& a) {
		int n = a.size();
		map<int, int> pre;

		auto first_less_than = [&](int val) {
			auto it = pre.lower_bound(val);
			if (it == pre.begin()) return pre.end();
			return --it;
		};

		for (int i = n - 1; i >= 0; --i)
		{
			auto it = first_less_than(a[i]);
			if (it != pre.end())
			{
				swap(a[i], a[it->second]);
				return a;
			}
			pre[a[i]] = i;
		}
		return a;
	}

	//1054. Distant Barcodes
    vector<int> rearrangeBarcodes(vector<int>& a) {
        int n = a.size();
        int const maxn = 10000 + 1;
        vector<int> cnt(maxn);
        int max_n = 0, max_cnt = 0;
        for (int &e : a)
        {
            if (++ cnt[e] > max_cnt)
            {
                max_cnt = cnt[e];
                max_n = e;
            }
        }
        int pos = 0;
        for (int i = 0; i < maxn; ++i)
        {
            int u = (i == 0 ? max_n : i);
            while (cnt[u] > 0)
            {
                a[pos] = u;
                pos += 2;
                if (pos >= n) pos = 1;
                cnt[u] --;
            }
        }
        return a;
    }
	//1055. Fixed Point
	int fixedPoint(vector<int>& A) {
		int n = A.size();
		for (int i = 0; i < n; ++i)
		{
			if (i == A[i]) return i;
		}
		return -1;
	}

	//1056. Index Pairs of a String
	vector<vector<int>> indexPairs(string s, vector<string>& words) {
		set<string> ss(words.begin(), words.end());
		int n = s.size();
		vector<vector<int>> ans;
		for (int i = 0; i < n; ++i)
		{
			for (int j = i; j < n; ++j)
			{
				if (ss.count(s.substr(i, j - i + 1)))
				{
					ans.push_back({ i, j });
				}
			}
		}
		return ans;
	}

	//1066. Campus Bikes II
	int assignBikes(vector<vector<int>>& workers, vector<vector<int>>& bikes) {
		int n = workers.size(), m = bikes.size();
		int B = (1 << m);
		int const maxn = INT_MAX / 2;
		vector<int> dp(B, maxn), nx(B, maxn);
		dp[0] = 0;
		for (int i = 0; i < n; ++i)
		{
			fill(nx.begin(), nx.end(), maxn);
			for (int s = 0; s < B; ++s)
			{
				for (int j = 0; j < m; ++j)
				{
					// nx[s] = INT_MAX;
					if ((s >> j) & 1)
					{
						nx[s] = min(nx[s], dp[s ^ (1 << j)] + abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1]));
					}
				}
			}
			dp = nx;
		}
		return *min_element(dp.begin(), dp.end());
	}

	//1074. Number of Submatrices That Sum to Target
	int numSubmatrixSumTarget(vector<vector<int>>& a, int t) {
		int n = a.size(), m = a[0].size();

		vector<vector<int>> sums(n, vector<int>(m + 1));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				sums[i][j + 1] = sums[i][j] + a[i][j];
			}
		}

		int ans = 0;
		map<int, int> pre;
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				pre.clear();
				pre[0] = 1;
				int s = 0;
				for (int k = 0; k < n; ++k)
				{
					int v = sums[k][i + 1] - sums[k][j];
					s += v;
					if (pre.count(s - t)) ans += pre[s - t];
					pre[s] ++;
				}
			}
		}
		return ans;
	}

	bool check_1071(string& p, string t)
	{
		if (p.size() % t.size()) return false;
		int n = p.size(), m = t.size();
		for (int i = 0; i < n; ++i)
		{
			if (p[i] != t[i % m]) return false;
		}
		return true;
	}

	//1071. Greatest Common Divisor of Strings
	string gcdOfStrings(string str1, string str2) {
		if (str1.size() < str2.size()) swap(str1, str2);
		int n = str2.size();
		for (int i = n; i >= 1; --i)
		{
			if (check_1071(str1, str2.substr(0, i)) && check_1071(str2, str2.substr(0, i)))
			{
				return str2.substr(0, i);
			}
		}
		return "";
	}

};
int main()
{
    Solution sol;
    vector<int> a;
    a = { 3, 1, 1, 3 };
    auto r = sol.prevPermOpt1(a);
    for (auto &e : r) cout << e << " ";
    cout << endl;
	return 0;
}
