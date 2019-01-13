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
	int minDeletionSizeIII(vector<string>& a) {
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

	//966. Vowel Spellchecker
	vector<string> spellchecker(vector<string> &wordlist, vector<string> &queries)
	{
		vector<bool> words(128);
		words['a'] = 1;
		words['e'] = 1;
		words['o'] = 1;
		words['i'] = 1;
		words['u'] = 1;
		map<string, int> set_case, set_vowel, set_same;

		int n = wordlist.size();

		for (int i = 0; i < n; ++i)
			if (!set_same.count(wordlist[i]))
				set_same[wordlist[i]] = i;

		for (int i = 0; i < n; ++i)
		{
			auto e = wordlist[i];
			for (char &c : e)
				c = tolower(c);

			if (!set_case.count(e))
				set_case[e] = i;

			for (char &c : e)
				if (words[c])
					c = 'a';

			if (!set_vowel.count(e))
				set_vowel[e] = i;
		}

		n = queries.size();

		vector<string> ans(n);
		for (int i = 0; i < n; ++i)
		{
			if (set_same.count(queries[i]))
			{
				ans[i] = wordlist[set_same[queries[i]]];
				continue;
			}

			for (char &c : queries[i]) c = tolower(c);

			if (set_case.count(queries[i]))
			{
				ans[i] = wordlist[set_case[queries[i]]];
				continue;
			}

			for (char &c : queries[i])
			{
				if (words[c])
					c = 'a';
			}

			if (set_vowel.count(queries[i]))
			{
				ans[i] = wordlist[set_vowel[queries[i]]];
			}
		}
		return ans;
	}

	//965. Univalued Binary Tree
	bool isUnivalTree(TreeNode* root) {
		if (!root) return true;
		if (root->left && root->val != root->left->val) return false;
		if (root->right && root->val != root->right->val) return false;
		return isUnivalTree(root->left) && isUnivalTree(root->right);
	}

	//967. Numbers With Same Consecutive Differences
	void dfs_967(int i, int val, int d, int n, vector<int> &ans)
	{
		// cout << i << " " << val << endl;
		if (i == n)
		{
			ans.push_back(val);
			return;
		}

		int p = val % 10;
		if (p + d < 10)
			dfs_967(i + 1, val * 10 + p + d, d, n, ans);
		if (d && p - d >= 0)
			dfs_967(i + 1, val * 10 + p - d, d, n, ans);
	}

	//967. Numbers With Same Consecutive Differences
	vector<int> numsSameConsecDiff(int N, int K) {
		if (N == 1)
		{
			return { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		}
		vector<int> ans;
		for (int i = 1; i < 10; ++i)
			dfs_967(1, i, K, N, ans);
		return ans;
	}


	
	int dfs_968(TreeNode *u, int isCam, int isCov, map<pair<TreeNode *, int>, int> &memo)
	{
		if (!u) return 0;
		auto key = make_pair(u, (isCam << 1) | isCov);
		if (memo.count(key)) return memo[key];
		auto l = u->left, r = u->right;
		int ans = INT_MAX;

		if (isCam)
		{
			ans = min(ans, dfs_968(u->left, 0, 1, memo) + dfs_968(u->right, 0, 1, memo));
			ans = min(ans, dfs_968(u->left, 1, 1, memo) + dfs_968(u->right, 1, 1, memo) + 1);
		}
		else
		{
			if (isCov)
			{
				if (u->left || u->right) ans = min(ans, dfs_968(u->left, 0, 0, memo) + dfs_968(u->right, 0, 0, memo));
				ans = min(ans, dfs_968(u->left, 1, 1, memo) + dfs_968(u->right, 1, 1, memo) + 1);
				if (u->left) ans = min(ans, dfs_968(u->left->left, 1, 1, memo) + 1 + dfs_968(u->left->right, 1, 1, memo) + dfs_968(u->right, 0, 1, memo));
				if (u->right) ans = min(ans, dfs_968(u->left, 0, 1, memo) + dfs_968(u->right->left, 1, 1, memo) + 1 + dfs_968(u->right->right, 1, 1, memo));
			}
			else
			{
				ans = min(ans, dfs_968(u->left, 1, 1, memo) + dfs_968(u->right, 1, 1, memo) + 1);
			}
		}
		memo[key] = ans;
		return ans;
	}

	//968. Binary Tree Cameras
	int minCameraCover(TreeNode* root) {
		map<pair<TreeNode *, int>, int> memo;
		return dfs_968(root, 0, 1, memo);
	}


	//969. Pancake Sorting
	vector<int> pancakeSort(vector<int>& a) {
		int n = a.size();
		vector<int> ans;
		for (int i = n - 1; i >= 0; --i)
		{
			int mx = i;
			for (int j = 0; j <= i; ++j)
			{
				if (a[mx] < a[j]) mx = j;
			}
			reverse(a.begin(), a.begin() + mx + 1);
			ans.push_back(mx + 1);
			reverse(a.begin(), a.begin() + i + 1);
			ans.push_back(i + 1);
		}
		return ans;
	}

	//970. Powerful Integers
	vector<int> powerfulIntegers(int x, int y, int bound) {
		vector<int> ans;
		for (ll i = 1; i <= bound;)
		{
			for (ll j = 1; j <= bound;)
			{
				if (i + j <= bound)
				{
					ans.push_back(i + j);
				}
				else break;
				if (j * y == j) break;
				j *= y;
			}
			if (i * x == i) break;
			i *= x;
		}
		sort(ans.begin(), ans.end());
		ans.erase(unique(ans.begin(), ans.end()), ans.end());
		return ans;
	}

/*
	map<vector<int>, pair<int, vector<int>>> memo;

	pair<int, vector<int>> dfs(TreeNode *u, int &i, vector<int> &v)
	{
		if (!u) return { 0,{} };


		if (u->val != v[i + 1]) return { -1,{} };

		i = i + 1;
		int ni = -1;
		int old = i;
		auto l = dfs(u->left, i, v);
		auto r = dfs(u->right, i, v);
		int li = i;

		int ans = INT_MAX;
		vector<int> vals;
		if (l.first != -1 && r.first != -1 && ans > l.first + r.first)
		{
			ans = l.first + r.first;
			vals.clear();

			for (int e : l.second) vals.push_back(e);
			for (int e : r.second) vals.push_back(e);

			ni = i;
		}

		i = old;
		l = dfs(u->right, i, v);
		r = dfs(u->left, i, v);
		int ri = i;
		if (l.first != -1 && r.first != -1 && ans > l.first + r.first + 1)
		{
			ans = l.first + r.first + 1;
			vals.clear();
			for (int e : l.second) vals.push_back(e);
			for (int e : r.second) vals.push_back(e);
			vals.push_back(u->val);

			ni = i;
		}

		if (ans == INT_MAX) return { -1, {} };
		auto ret = make_pair(ans, vals);
		i = ni;
		return ret;
	}

	//971. Flip Binary Tree To Match Preorder Traversal
	vector<int> flipMatchVoyage(TreeNode* root, vector<int>& voyage) {
		int i = -1;
		memo.clear();
		vector<int> path;
		auto ret = dfs(root, i, voyage);
		if (ret.first == -1) return { -1 };
		return ret.second;
	}
*/
	//973. K Closest Points to Origin
	vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
		auto cmp = [](vector<int> &a, vector<int> &b) {
			return a[0] * a[0] + a[1] * a[1] < b[0] * b[0] + b[1] * b[1];
		};
		sort(points.begin(), points.end(), cmp);
		vector<vector<int>> ans;
		for (int i = 0; i < min(K, (int)points.size()); ++i)
		{
			ans.push_back(points[i]);
		}
		return ans;
	}

	//974. Subarray Sums Divisible by K
	int subarraysDivByK(vector<int>& a, int k) {
		int n = a.size();
		vector<int> sums(n + 1);
		for (int i = 1; i <= n; ++i)
		{
			sums[i] = (sums[i - 1] + a[i - 1] % k + k) % k;
		}

		vector<int> cnt(k + 1);
		for (int i = 0; i <= n; ++i) cnt[sums[i]]++;

		int ans = 0;

		for (int i = 0; i <= n; ++i)
		{
			cnt[sums[i]]--;
			ans += cnt[sums[i]];
		}
		return ans;
	}

	//975. Odd Even Jump
	int oddEvenJumps(vector<int>& a) {
		int n = a.size();
		vector<vector<int>> next(2, vector<int>(n, -1));
		vector<int> stk;
		map<int, int> last;
		for (int i = n - 1; i >= 0; --i)
		{
			auto it = last.lower_bound(a[i]);
			if (it != last.end())
				next[1][i] = it->second;
			last[a[i]] = i;
		}

		last.clear();

		for (int i = n - 1; i >= 0; --i)
		{
			auto it = last.upper_bound(a[i]);
			if (it != last.begin())
			{
				--it;
				next[0][i] = it->second;
			}
			last[a[i]] = i;
		}

		vector<vector<bool>> vis(2, vector<bool>(n));
		vis[1][n - 1] = 1;
		for (int i = 0; i < n; ++i)
		{
			if (!vis[1][i])
			{
				int u = i;
				int f = 1;
				while (u != -1 && !vis[f][u] && u != n - 1)
				{
					u = next[f][u];
					f = 1 - f;
				}

				if (u == n - 1)
				{
					u = i;
					f = 1;

					vis[f][u] = 1;
					while (u != n - 1 && !vis[f][u])
					{
						u = next[f][u];
						f = 1 - f;
						vis[f][u] = 1;
					}
				}
			}
		}

		int ans = 0;
		for (int i = 0; i < n; ++i) ans += vis[1][i];
		return ans;
	}

	//976. Largest Perimeter Triangle
	int largestPerimeter(vector<int>& A) {
		sort(A.begin(), A.end());
		for (int i = A.size() - 1; i > 1; --i)
			if (A[i] < A[i - 1] + A[i - 2])
				return A[i] + A[i - 1] + A[i - 2];
		return 0;
	}
};

int main()
{
	return 0;
}