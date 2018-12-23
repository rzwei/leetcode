#include <bitset>
#include "headers.h"
#include <sstream>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <stack>
#include <climits>
#include <string>
#include <utility>
#include <cmath>
#include <numeric>
using namespace std;
typedef long long ll;

class Solution
{
public:
	bool less(string &a, string &b, vector<int> &order)
	{
		int i = 0, j = 0;
		int la = a.size(), lb = b.size();
		int l = max(la, lb);
		while (i < l)
		{
			int ai = 0, bj = 0;
			if (i < la) ai = order[a[i] - 'a'];
			if (i < lb) bj = order[b[i] - 'a'];
			if (ai == bj) i++;
			else return ai < bj;
		}
		return true;
	}

	//953. Verifying an Alien Dictionary
	bool isAlienSorted(vector<string>& words, string order) {
		vector<int> old(26);
		for (int i = 0; i < order.size(); ++i)
		{
			old[order[i] - 'a'] = i;
		}
		int n = words.size();
		for (int i = 1; i < n; ++i)
		{
			if (!less(words[i - 1], words[i], old)) return false;
		}
		return true;
	}

	//954. Array of Doubled Pairs
	bool canReorderDoubled(vector<int>& a) {
		if (accumulate(a.begin(), a.end(), 0) % 3) return false;
		auto cmp = [](int a, int b) {
			return abs(a) < abs(b);
		};
		sort(a.begin(), a.end(), cmp);
		int n = a.size();
		int cur = 0;
		map<int, int> cnt;
		for (int i = 0; i < n; ++i)
		{
			if (a[i] % 2 == 0 && cnt[a[i] / 2] > 0)
			{
				cur++;
				cnt[a[i] / 2]--;
			}
			else
			{
				cnt[a[i]]++;
			}
		}
		return cur >= n / 2;
	}

	//955. Delete Columns to Make Sorted II
	int minDeletionSize(vector<string>& A) {
		int n = A.size(), m = A[0].size(), i, j, ans = 0;
		vector<bool> vis(n);
		for (j = 0; j < m; ++j)
		{
			for (i = 0; i < n - 1; ++i)
			{
				if (A[i][j] > A[i + 1][j] && vis[i] == 0)
				{
					ans++;
					break;
				}
			}
			if (i < n - 1) continue;
			for (i = 0; i < n - 1; ++i)
			{
				if (A[i][j] < A[i + 1][j]) vis[i] = 1;
			}
		}
		return ans;
	}

	//956. Tallest Billboard
	int tallestBillboard(vector<int>& a) {
		int const offset = accumulate(a.begin(), a.end(), 0), maxn = 2 * offset + 1;
		int n = a.size();
		vector<int> dp(maxn, INT_MIN), pre(maxn, INT_MIN);
		pre[offset] = 0;
		for (int e : a)
		{
			for (int v = 0; v < maxn; ++v)
			{
				dp[v] = pre[v];
				if (v - e >= 0) dp[v] = max(dp[v], pre[v - e] + e);
				if (v + e < maxn) dp[v] = max(dp[v], pre[v + e] + e);
			}
			pre = dp;
		}
		return pre[offset] / 2;
	}

	void next_957(vector<int> &a, vector<int> &b)
	{
		b[0] = 0;
		b[7] = 0;
		for (int i = 1; i < 7; ++i)
		{
			b[i] = (a[i - 1] == 0 && a[i + 1] == 0 || a[i - 1] == 1 && a[i + 1] == 1);
		}
	}

	//957. Prison Cells After N Days
	vector<int> prisonAfterNDays(vector<int>& cells, int N) {
		map<vector<int>, int> vis;
		int u = 0;
		vector<int> a = cells, b = cells;
		for (int i = 0; i < N; ++i)
		{
			next_957(a, b);
			if (vis.count(b))
			{
				int s = vis[b], t = i;
				int v = (N - s - 1) % (t - s);
				for (auto &e : vis)
				{
					if (e.second == s + v) return e.first;
				}
			}
			else
			{
				vis[b] = u++;
			}
			swap(a, b);
		}
		return a;
	}

	//958. Check Completeness of a Binary Tree
	bool isCompleteTree(TreeNode* root) {
		queue<TreeNode *> q;
		q.push(root);
		while (!q.empty() && q.front())
		{
			q.push(q.front()->left);
			q.push(q.front()->right);
			q.pop();
		}
		while (!q.empty() && !q.front()) q.pop();
		return q.size() == 0;
	}

	//960. Delete Columns to Make Sorted III
	int minDeletionSize(vector<string>& a) {
		int n = a.size(), m = a[0].size();

		vector<int> dp(m, 1);
		for (int i = 1; i < m; ++i)
		{
			for (int j = 0; j < i; ++j)
			{

				bool f = true;
				for (int k = 0; f && k < n; ++k)
					if (a[k][j] > a[k][i]) f = 0;
				if (f)
				{
					dp[i] = max(dp[i], dp[j] + 1);
				}
			}
		}
		return m - *max_element(dp.begin(), dp.end());
	}

	void dfs_959(int i, int j, int d, int n, vector<int> &invalid, vector<int> &vis)
	{
		//cout << i << " " << j << " " << d << endl;
		static int dr[] = { 0, -1, 0, 1 };
		static int dc[] = { -1, 0, 1, 0 };
		int u = 4 * (i * n + j);
		vis[u + d] = 1;
		int dt;
		if (d == 0 || d == 2) dt = 2 - d;
		else dt = 4 - d;
		int nx = i + dr[d], ny = j + dc[d];
		int v = 4 * (nx * n + ny);
		if (0 <= nx && nx < n && 0 <= ny && ny < n && !vis[v + dt])
			dfs_959(nx, ny, dt, n, invalid, vis);
		dt = d - 1;
		if (dt < 0) dt += 4;
		if (!vis[u + dt] && !invalid[u + dt]) dfs_959(i, j, dt, n, invalid, vis);
		dt = d + 1;
		if (dt >= 4) dt -= 4;
		if (!vis[u + dt] && !invalid[u + d]) dfs_959(i, j, dt, n, invalid, vis);
	}

	//959. Regions Cut By Slashes
	int regionsBySlashes(vector<string>& g) {
		int n = g.size();
		vector<int> invalid(4 * n * n);
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (g[i][j] == ' ') continue;
				int u = 4 * (i * n + j);
				if (g[i][j] == '/')
				{
					invalid[u + 1] = 1;
					invalid[u + 3] = 1;
				}
				else
				{
					invalid[u + 0] = 1;
					invalid[u + 2] = 1;
				}
			}
		}
		vector<int> vis(4 * n * n);
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				for (int d = 0; d < 4; ++d)
				{
					int u = 4 * (i * n + j);
					if (!vis[u + d])
					{
						ans++;
						dfs_959(i, j, d, n, invalid, vis);
					}
				}
			}
		}
		return ans;
	}

	ll sqr(ll x) {
		return x * x;
	}
	ll dis(vector<int>& a, vector<int>& b) {
		return sqr(a[0] - b[0]) + sqr(a[1] - b[1]);
	}

	//963. Minimum Area Rectangle II
	double minAreaFreeRect(vector<vector<int>>& p) {
		int n = p.size();
		double ret = 1e300;
		for (int i = 0; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				for (int u = 0; u < n; ++u) {
					if (u == i || u == j) continue;
					for (int v = u + 1; v < n; ++v) {
						if (v == i || v == j) continue;
						int X1 = p[i][0] + p[j][0];
						int Y1 = p[i][1] + p[j][1];
						int X2 = p[u][0] + p[v][0];
						int Y2 = p[u][1] + p[v][1];
						if (X1 != X2 || Y1 != Y2) continue;
						ll D1 = dis(p[i], p[j]);
						ll D2 = dis(p[u], p[v]);
						if (D1 == D2) {
							ret = min(ret, sqrt(dis(p[i], p[u])) * sqrt(dis(p[i], p[v])));
						}
					}
				}
			}
		}
		if (ret > 1e200) ret = 0;
		return ret;
	}

	//961. N-Repeated Element in Size 2N Array
	int repeatedNTimes(vector<int>& A) {
		for (auto i = 2; i < A.size(); ++i) {
			if (A[i - 2] == A[i - 1] || A[i - 2] == A[i]) return A[i - 2];
			if (A[i - 1] == A[i]) return A[i];
		}
		return A[A.size() - 1];
	}
};

int main()
{
	return 0;
}