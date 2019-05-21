#include <bitset>
#include "headers.h"
#include <random>
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
int const INF = INT_MAX / 8;

/*
//1032. Stream of Characters
class Node {
public:
	Node* next[26];
	bool isend;
	char v;
	Node() :next{ nullptr }
	{
		isend = false;
	}
};

class StreamChecker {
public:
	int const max_len = 2000;
	Node* root;
	list<char> q;
	StreamChecker(vector<string>& words) {
		root = new Node();
		for (auto& s : words)
		{
			reverse(s.begin(), s.end());
			add(s);
		}
	}

	void add(string& s)
	{
		auto u = root;
		for (auto& c : s)
		{
			if (u->next[c - 'a'] == nullptr)
			{
				u->next[c - 'a'] = new Node();
				u->next[c - 'a']->v = c;
			}
			u = u->next[c - 'a'];
		}
		u->isend = true;
	}

	bool query(char c) {
		q.push_back(c);
		if (q.size() > max_len) q.pop_back();
		auto it = --q.end();
		auto u = root;
		while (u)
		{
			if (u->next[*it - 'a'])
			{
				u = u->next[*it - 'a'];
				if (u->isend) return true;
				if (it == q.begin()) return false;
				else --it;
			}
			else
			{
				return false;
			}
		}
		return false;
	}
};
*/

class Solution {
public:
	int dfs_1000(int l, int r, int m, vector<int>& sums, int k, vector<vector<vector<int>>>& memo)
	{
		if ((r - l + 1 - m) % (k - 1)) return INF;
		if (memo[l][r][m] != -1) return memo[l][r][m];
		if (l == r) return m == 1 ? 0 : INF;
		if (m == 1) return dfs_1000(l, r, k, sums, k, memo) + sums[r + 1] - sums[l];
		int ans = INF;
		for (int i = l; i < r; i += k - 1)
		{
			ans = min(ans, dfs_1000(l, i, 1, sums, k, memo) + dfs_1000(i + 1, r, m - 1, sums, k, memo));
		}
		memo[l][r][m] = ans;
		return ans;
	}

	//1000. Minimum Cost to Merge Stones
	int mergeStones(vector<int> & a, int k) {
		int n = a.size();
		vector<int> sums(n + 1);
		for (int i = 0; i < n; ++i)
		{
			sums[i + 1] = sums[i] + a[i];
		}
		vector<vector<vector<int>>> memo(n + 1, vector<vector<int>>(n + 1, vector<int>(n + 1, -1)));
		return dfs_1000(0, n - 1, 1, sums, k, memo);
	}

	int dfs_964(int cur, int target, int x, map<pair<int, int>, int> & memo)
	{
		if (target == 0) return 0;
		if (target == 1) return cost(cur);
		if (memo.count({ cur, target }))
			return memo[{cur, target}];

		if (cur > 30) return INF;

		int r = target % x;
		int t = target / x;
		int ans = INF;
		ans = min(ans, dfs_964(cur + 1, t, x, memo) + r * cost(cur));
		ans = min(ans, dfs_964(cur + 1, t + 1, x, memo) + (x - r) * cost(cur));
		memo[{cur, target}] = ans;
		return ans;
	}
	int cost(int x)
	{
		return x > 0 ? x : 2;
	}
	//964. Least Operators to Express Number
	int leastOpsExpressTarget(int x, int target) {
		map<pair<int, int>, int> memo;
		return dfs_964(0, target, x, memo) - 1;
	}

	//1005. Maximize Sum Of Array After K Negations
	int largestSumAfterKNegations(vector<int>& a, int k) {
		int n = a.size();
		vector<int> pos, neg;
		for (auto& e : a)
		{
			if (e >= 0) pos.push_back(e);
			else neg.push_back(e);
		}
		sort(pos.begin(), pos.end());
		sort(neg.begin(), neg.end());
		int ans = 0;
		for (int i = 0; i < neg.size(); ++i)
		{
			if (k > 0)
			{
				ans += -neg[i];
				k--;
			}
			else ans += neg[i];
		}

		if (k == 0)
		{
			for (int i = 0; i < pos.size(); ++i) ans += pos[i];
		}
		else
		{
			for (int i = 0; i < pos.size(); ++i) ans += pos[i];
			if (k & 1)
			{
				int n_min = INT_MAX, p_min = INT_MAX;
				if (!neg.empty()) n_min = -neg.back();
				if (!pos.empty()) p_min = pos[0];
				ans -= 2 * min(n_min, p_min);
			}
		}
		return ans;
	}

	int calc(int i)
	{
		if (i == 0) return 0;
		if (i == 1) return 1;
		if (i == 2) return i * (i - 1);
		if (i >= 3) return i * (i - 1) / (i - 2);
		return -1;
	}

	//1006. Clumsy Factorial
	int clumsy(int N) {
		int ans = 0;
		int f = 1;
		for (int i = N; i >= 0; )
		{
			ans += f * calc(i);
			if (i - 3 >= 1) ans += (i - 3);
			if (f == 1) f = -1;
			i -= 4;
		}
		return ans;
	}

	int solve(vector<int> & a, vector<int> & b)
	{
		int n = a.size();
		int ans = INT_MAX;
		for (int v = 1; v <= 6; ++v)
		{
			int cur = 0;
			for (int i = 0; i < n; ++i)
			{
				if (a[i] == v) continue;
				else
				{
					if (b[i] == v)
					{
						cur++;
					}
					else
					{
						cur = INT_MAX;
						break;
					}
				}
			}
			ans = min(ans, cur);
		}
		return ans;
	}

	//1007. Minimum Domino Rotations For Equal Row
	int minDominoRotations(vector<int> & a, vector<int> & b) {
		int ret = min(solve(a, b), solve(b, a));
		if (ret != INT_MAX) return ret;
		return -1;
	}

	TreeNode* build_1008(int l, int r, vector<int> & a)
	{
		if (r - l + 1 == 1) return new TreeNode(a[l]);
		if (l > r) return nullptr;
		auto ret = new TreeNode(a[l]);
		int f = r + 1;
		for (int i = l; i <= r; ++i)
		{
			if (a[i] > a[l])
			{
				f = i;
				break;
			}
		}
		ret->left = build_1008(l + 1, f - 1, a);
		ret->right = build_1008(f, r, a);
		return ret;
	}

	//1008. Construct Binary Search Tree from Preorder Traversal
	TreeNode* bstFromPreorder(vector<int> & preorder) {
		return build_1008(0, preorder.size() - 1, preorder);
	}

	//1012. Complement of Base 10 Integer
	int bitwiseComplement(int N) {
		if (N == 0) return 1;
		int ans = 0;
		int t = 1;
		while (N)
		{
			if (N & 1)
			{

			}
			else
			{
				ans |= t;
			}
			N >>= 1;
			t <<= 1;
		}
		return ans;
	}

	//1013. Pairs of Songs With Total Durations Divisible by 60
	int numPairsDivisibleBy60(vector<int>& time) {
		vector<int> cnt(60);
		int n = time.size();
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			int v = time[i] % 60;
			ans += cnt[(60 - v) % 60];
			cnt[v] ++;
		}
		return ans;
	}

	int check(vector<int>& a, int v)
	{
		int ans = 1, cur = 0;
		for (auto& e : a)
		{
			cur += e;
			if (cur > v)
			{
				ans++;
				cur = e;
			}
		}
		return ans;
	}
	//1014. Capacity To Ship Packages Within D Days
	int shipWithinDays(vector<int>& weights, int D) {
		int l = 0, r = 0;
		for (auto& e : weights) r += e, l = max(l, e);
		while (l < r)
		{
			int m = (l + r) / 2;
			if (check(weights, m) <= D)
			{
				r = m;
			}
			else
			{
				l = m + 1;
			}
		}
		return l;
	}


	int dfs_1015(bitset<10>& vis, int i, int f, vector<int>& num, map<vector<long long>, int> &memo)
	{
		if (i == num.size())
		{
			return 1;
		}
		int ans = 0;
		vector<long long> key = { (long long)vis.to_ullong(), i, f };
		if (memo.count(key)) return memo[key];
		if (f)
		{
			for (int j = 0; j < num[i]; ++j)
			{
				if (!vis[j])
				{
					vis[j] = 1;
					ans += dfs_1015(vis, i + 1, 0, num, memo);
					vis[j] = 0;
				}
			}
			if (!vis[num[i]])
			{
				vis[num[i]] = 1;
				ans += dfs_1015(vis, i + 1, f, num, memo);
				vis[num[i]] = 0;
			}
		}
		else
		{
			for (int j = 0; j <= 9; ++j)
			{
				if (!vis[j])
				{
					vis[j] = 1;
					ans += dfs_1015(vis, i + 1, 0, num, memo);
					vis[j] = 0;
				}
			}
		}
		memo[key] = ans;
		return ans;
	}

	//1015. Numbers With 1 Repeated Digit
	int numDupDigitsAtMostN(int N) {
		map<vector<long long>, int> memo;
		vector<int> num;
		int tot = N;
		while (N)
		{
			num.push_back(N % 10);
			N /= 10;
		}
		reverse(num.begin(), num.end());
		int ans = 0;
		bitset<10> vis;
		for (int i = 0; i < num.size(); ++i)
		{
			if (i == 0)
			{
				vis[num[i]] = 1;
				ans += dfs_1015(vis, i + 1, 1, num, memo);
				vis[num[i]] = 0;
				for (int j = 1; j < num[i]; ++j)
				{
					vis[j] = 1;
					ans += dfs_1015(vis, i + 1, 0, num, memo);
					vis[j] = 0;
				}
			}
			else
			{
				for (int j = 1; j <= 9; ++j)
				{
					vis[j] = 1;
					ans += dfs_1015(vis, i + 1, 0, num, memo);
					vis[j] = 0;
				}
			}
		}
		return tot - ans;
	}

	//int dfs(int s, int n, int N)
	//{
	//	if (n > N) return 0;
	//	int ans = 1;
	//	if (n * 10 > N) return ans;
	//	for (int i = 0; i < 10; ++i)
	//	{
	//		if (s & (1 << i)) continue;
	//		if (!s && i == 0) continue;
	//		ans += dfs(s | (1 << i), n * 10 + i, N);
	//	}
	//	return ans;
	//}

	//int numDupDigitsAtMostN(int N) {
	//	return N + 1 - dfs(0, 0, N);
	//}

	// 1020. Partition Array Into Three Parts With Equal Sum
	bool canThreePartsEqualSum(vector<int>& a) {
		int n = a.size();
		int sum = accumulate(a.begin(), a.end(), 0);
		if (sum % 3) return false;
		int tar = sum / 3, cur = 0;
		bool find_tar = false;
		for (int i = 0; i < n; ++i)
		{
			cur += a[i];
			if (!find_tar && cur == tar) find_tar = true;
			else if (cur == tar * 2 && find_tar) return true;
		}
		return false;
	}

	//1022. Smallest Integer Divisible by K
	int smallestRepunitDivByK(int K) {
		set<int> vis;
		int cur = 0, len = 0;
		while (true)
		{
			cur = (cur * 10 + 1) % K;
			len++;
			if (cur == 0) return len;
			if (vis.count(cur)) return -1;
			vis.insert(cur);
		}
		return -1;
	}

	//1021. Best Sightseeing Pair
	int maxScoreSightseeingPair(vector<int>& a) {
		int n = a.size();
		int ans = 0;
		vector<int> right(n);
		right[n - 1] = a[n - 1] - (n - 1);
		for (int i = n - 2; i >= 0; --i)
		{
			right[i] = max(right[i + 1], a[i] - i);
		}
		int mx = a[0] + 0;
		for (int i = 1; i < n; ++i)
		{
			ans = max(ans, mx + right[i]);
			mx = max(mx, a[i] + i);
		}
		return ans;
	}

	//1023. Binary String With Substrings Representing 1 To N
	bool queryString(string S, int N) {
		int Max = 1e5;
		mt19937 e;
		while (Max)
		{
			int v = e() % N + 1;
			string s;
			while (v)
			{
				s.push_back(v % 2 + '0');
				v /= 2;
			}
			reverse(s.begin(), s.end());
			if (S.find(s) == string::npos) {
				return false;
			}
			Max--;
		}
		return true;
	}

	//1017. Convert to Base -2
	string baseNeg2(int N) {
		if (N == 0) return "0";
		string ans;
		int n = N, u = 0;
		while (n)
		{
			if (n & 1)
			{
				ans.push_back('1');
				if (u % 2 == 1)
					n += 2;
			}
			else ans.push_back('0');
			n >>= 1;
			u++;
		}
		reverse(ans.begin(), ans.end());
		return ans;
	}

	//1020. Number of Enclaves
	int numEnclaves(vector<vector<int>>& A) {
		int n = A.size(), m = A[0].size();
		queue<int> q;
		int ans = 0;
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j) ans += (A[i][j] == 1);
		for (int i = 0; i < n; ++i)
		{
			if (A[i][0] == 1) q.push(i * m), A[i][0] = 0;
			if (m && A[i][m - 1] == 1) q.push(i * m + m - 1), A[i][m - 1] = 0;
		}
		for (int j = 1; j < m - 1; ++j)
		{
			if (A[0][j]) q.push(0 * m + j), A[0][j] = 0;
			if (n && A[n - 1][j]) q.push((n - 1) * m + j), A[n - 1][j] = 0;
		}
		int dr[] = { 0, 1, 0, -1 };
		int dc[] = { 1, 0, -1, 0 };
		ans -= q.size();
		while (!q.empty())
		{
			int x = q.front() / m, y = q.front() % m; q.pop();
			// cout << x << " " << y << endl;
			for (int d = 0; d < 4; ++d)
			{
				int nx = x + dr[d], ny = y + dc[d];
				if (0 <= nx && nx < n && 0 <= ny && ny < m && A[nx][ny] == 1)
				{
					A[nx][ny] = 0;
					// cout << nx << " " << ny << endl;
					q.push(nx * m + ny);
					ans--;
				}
			}
		}

		return ans;
	}

	//1019. Next Greater Node In Linked List
	vector<int> nextLargerNodes(ListNode* head) {
		int n = 0;
		auto u = head;
		while (u)
		{
			u = u->next;
			n++;
		}
		vector<int> ans(n);
		stack<pair<int, int>> stk;
		u = head;
		int i = 0;
		while (u)
		{
			while (!stk.empty() && stk.top().first < u->val)
			{
				ans[stk.top().second] = u->val;
				stk.pop();
			}
			stk.push({ u->val, i });
			u = u->next;
			i++;
		}
		return ans;
	}

	//1018. Binary Prefix Divisible By 5
	vector<bool> prefixesDivBy5(vector<int>& A) {
		int n = A.size();
		vector<bool> ans(n);
		int s = 0, t = 1;
		for (int i = 0; i < n; ++i)
		{
			s = (s * 2 + A[i]) % 5;
			if (s == 0) ans[i] = 1;
		}
		return ans;
	}

	long long dfs_5017(TreeNode* u, int val)
	{
		static int const mod = 1e9 + 7;

		if (!u) return 0;
		long long cur = (val * 2ll + u->val) % mod;
		if (!u->left && !u->right) return cur;
		return dfs_5017(u->left, cur) + dfs_5017(u->right, cur);
	}

	//1022. Sum of Root To Leaf Binary Numbers
	int sumRootToLeaf(TreeNode * root) {
		return dfs_5017(root, 0);
	}


	bool match_5018(string& s, string& p)
	{
		int j = 0;
		for (char& c : s)
		{
			if (j < p.size() && p[j] == c)
			{
				j++;
			}
			else if ('A' <= c && c <= 'Z') return false;
		}
		return j == p.size();
	}

	//1023. Camelcase Matching
	vector<bool> camelMatch(vector<string> & queries, string pattern) {
		int n = queries.size();
		vector<bool> ans(n);
		for (int i = 0; i < n; ++i) ans[i] = match_5018(queries[i], pattern);
		return ans;
	}

	//1024. Video Stitching
	int videoStitching(vector<vector<int>>& clips, int T) {
		int n = clips.size();
		sort(clips.begin(), clips.end());
		int t = 0, nt = 0, ans = 0;
		for (int i = 0; i < n; ++i)
		{
			if (clips[i][0] <= t)
			{
				nt = max(nt, clips[i][1]);
				if (nt >= T) return ans + 1;
			}
			else
			{
				if (clips[i][0] <= nt)
				{
					ans++;
					t = nt;
					nt = clips[i][1];
					if (t >= T) return ans;
				}
				else
				{
					return -1;
				}
			}
		}
		if (nt >= T) return ans + 1;
		return -1;
	}

	//1021. Remove Outermost Parentheses
	string removeOuterParentheses(string S) {
		string ans;
		int cnt = 0;
		for (char c : S)
		{
			int pre = cnt;
			if (c == '(') cnt++;
			else cnt--;
			if (pre == 0 && cnt == 1 || pre == 1 && cnt == 0)
			{
				continue;
			}
			ans += c;
		}
		return ans;
	}
	/*
		//1027. Longest Arithmetic Sequence
		int longestArithSeqLength(vector<int>& a) {
			int n = a.size();
			vector<map<int, int>> dp(n);
			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < i; ++j)
				{
					int d = a[i] - a[j];
					if (dp[j].count(d)) dp[i][d] = max(dp[i][d], dp[j][d] + 1);
					else dp[i][d] = max(dp[i][d], 2);
				}
			}
			int ans = 0;
			for (int i = 0; i < n; ++i)
			{
				for (auto& p : dp[i])
				{
					ans = max(ans, p.second);
				}
			}
			return ans;
		}
	*/
	
	//1027. Longest Arithmetic Sequence
	int longestArithSeqLength(vector<int>& a) {
		int n = a.size();
		map<int, vector<int>> idx;
		for (int i = 0; i < n; ++i) idx[a[i]].push_back(i);

		vector<vector<int>> dp(n, vector<int>(n));

		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < i; ++j)
			{
				int x = a[j] * 2 - a[i];
				dp[j][i] = 2;
				if (idx.count(x))
				{
					auto it = upper_bound(idx[x].begin(), idx[x].end(), j);
					if (it == idx[x].begin()) continue;
					--it;
					// for (auto k = idx[x].begin(); k != it; ++k)
					// {
					dp[j][i] = max(dp[j][i], dp[*it][j] + 1);
					// }
				}
			}
		}

		// for (int i = 0; i < n; ++i)
		// {
		//     for (int j = 0; j < n; ++j)
		//         cout << dp[i][j] << ' ';
		//     cout << endl;
		// }

		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < i; ++j)
			{
				ans = max(ans, dp[j][i]);
			}
		}
		return ans;
	}

	TreeNode* build_1028(int& i, int dep, string& s)
	{
		if (i == s.size()) return nullptr;
		int u = i, cur = 0;
		while (u < s.size() && s[u] == '-') u++;
		cur = u - i;
		if (cur == dep)
		{
			int val = 0;
			while (u < s.size() && '0' <= s[u] && s[u] <= '9') val = val * 10 + s[u++] - '0';
			auto ret = new TreeNode(val);
			i = u;
			ret->left = build_1028(i, dep + 1, s);
			ret->right = build_1028(i, dep + 1, s);
			return ret;
		}
		else return nullptr;
	}

	//1028. Recover a Tree From Preorder Traversal
	TreeNode* recoverFromPreorder(string s) {
		int i = 0;
		return build_1028(i, 0, s);
	}

	int dfs_1026(TreeNode* u, int pmax, int pmin)
	{
		if (!u) return 0;
		int ans = 0;
		ans = max(ans, abs(pmax - u->val));
		ans = max(ans, abs(pmin - u->val));
		ans = max(ans, abs(pmin - u->val));
		pmax = max(pmax, u->val);
		pmin = min(pmin, u->val);
		ans = max(ans, dfs_1026(u->left, pmax, pmin));
		ans = max(ans, dfs_1026(u->right, pmax, pmin));
		return ans;
	}
	//1026. Maximum Difference Between Node and Ancestor
	int maxAncestorDiff(TreeNode* root) {
		if (!root) return 0;
		return dfs_1026(root, root->val, root->val);
	}

	//1025. Divisor Game
	bool divisorGame(int N) {
		//vector<bool> dp(N + 1, 1);
		//dp[1] = 0;
		//for (int i = 2; i <= N; ++i)
		//{
		//	bool f = false;
		//	for (int j = 1; j < i; ++j)
		//	{
		//		if (i % j == 0 && dp[i - j] == false)
		//		{
		//			f = true;
		//			break;
		//		}
		//	}
		//	dp[i] = f;
		//}
		//return dp[N];
		return N % 2 == 0;
	}

	//1029. Two City Scheduling
	int twoCitySchedCost(vector<vector<int>>& a) {
		int n = a.size();
		auto cmp = [](vector<int>& a, vector<int>& b) {
			return a[0] - a[1] < b[0] - b[1];
		};
		sort(a.begin(), a.end(), cmp);
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			if (i < n / 2) ans += a[i][0];
			else ans += a[i][1];
		}
		return ans;
	}

	//1030. Matrix Cells in Distance Order
	vector<vector<int>> allCellsDistOrder(int R, int C, int r0, int c0) {
		queue<pair<int, int>> q;
		q.push({ r0, c0 });
		static int dr[] = { 0, 1, 0, -1 };
		static int dc[] = { 1, 0, -1, 0 };
		vector<vector<int>> ans(R * C, vector<int>(2));
		vector<vector<bool>> vis(R, vector<bool>(C));
		vis[r0][c0] = 1;
		int ansi = 0;
		ans[ansi++] = { r0, c0 };
		while (!q.empty())
		{
			int size = q.size();
			while (size--)
			{
				auto x = q.front().first, y = q.front().second;
				q.pop();
				for (int d = 0; d < 4; ++d)
				{
					int nx = x + dr[d], ny = y + dc[d];
					if (0 <= nx && nx < R && 0 <= ny && ny < C && !vis[nx][ny])
					{
						q.emplace(nx, ny);
						vis[nx][ny] = 1;
						ans[ansi][0] = nx;
						ans[ansi][1] = ny;
						ansi++;
					}
				}
			}
		}
		return ans;
	}

	// 1031. Maximum Sum of Two Non - Overlapping Subarrays
	int maxSumTwoNoOverlap(vector<int>& a, int L, int M) {
		int ans = 0;
		int n = a.size();
		vector<int> sum(n + 1);
		for (int i = 0; i < n; ++i) sum[i + 1] = sum[i] + a[i];
		int pre = 0;
		for (int i = L - 1; i < n; ++i)
		{
			pre = max(pre, sum[i + 1] - sum[i - L + 1]);
			if (i + 1 + M <= n)
			{
				ans = max(ans, pre + sum[i + 1 + M] - sum[i + 1]);
			}
		}

		pre = 0;
		for (int i = M - 1; i < n; ++i)
		{
			pre = max(pre, sum[i + 1] - sum[i - M + 1]);
			if (i + 1 + L <= n)
			{
				ans = max(ans, pre + sum[i + 1 + L] - sum[i + 1]);
			}
		}

		return ans;
	}

	//1033. Moving Stones Until Consecutive
	vector<int> numMovesStones(int a, int b, int c) {
		vector<int> x = { a, b, c };
		sort(x.begin(), x.end());
		int mi = 2, mx = x[1] - x[0] - 1 + x[2] - x[1] - 1;


		if (x[0] == x[1] - 1) mi--;
		if (x[1] == x[2] - 1) mi--;

		if (x[0] + 2 == x[1] || x[1] + 2 == x[2]) mi = 1;

		return { mi, mx };
	}

	//1034. Coloring A Border
	vector<vector<int>> colorBorder(vector<vector<int>>& g, int r0, int c0, int color) {
		if (g[r0][c0] == color) return g;
		int dr[] = { 0, 1, -1, 0 };
		int dc[] = { 1, 0, 0, -1 };
		int n = g.size();
		int m = g[0].size();
		queue<pair<int, int>> q;
		q.push({ r0, c0 });
		int c = g[r0][c0];
		g[r0][c0] = 0;
		while (!q.empty())
		{
			int x = q.front().first, y = q.front().second;
			q.pop();
			for (int d = 0; d < 4; ++d)
			{
				int nx = x + dr[d], ny = y + dc[d];
				if (0 <= nx && nx < n && 0 <= ny && ny < m)
				{
					if (g[nx][ny] == c)
					{
						q.push({ nx, ny });
						g[nx][ny] = 0;
					}
				}
			}
		}


		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				if (g[i][j] == 0)
				{
					bool f = false;
					for (int d = 0; !f && d < 4; ++d)
					{
						int nx = i + dr[d], ny = j + dc[d];
						if (nx < 0 || nx >= n || ny < 0 || ny >= m)
						{
							f = true;
						}
						else  if (g[nx][ny] != 0) f = true;
					}
					if (f)
					{
						q.push({ i, j });
					}
				}
			}
		}

		for (auto& row : g)
			for (auto& e : row)
				if (e == 0) e = c;

		while (!q.empty())
		{
			int x = q.front().first, y = q.front().second;
			q.pop();
			g[x][y] = color;
		}
		return g;
	}

	//1035. Uncrossed Lines
	int maxUncrossedLines(vector<int>& a, vector<int>& b) {
		int n = a.size(), m = b.size();
		vector<vector<int>> dp(n + 1, vector<int>(m + 1));
		for (int i = 1; i <= n; ++i)
		{
			for (int j = 1; j <= m; ++j)
			{
				if (a[i - 1] == b[j - 1])
				{
					dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i][j]);
				}
				else
				{
					dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
		return dp[n][m];
	}

	//1036. Escape a Large Maze
	bool isEscapePossible(vector<vector<int>>& a, vector<int>& st, vector<int>& ed) {
		int n = a.size();
		set<pair<int, int>> block;
		for (auto& e : a)
		{
			if (e[0] == ed[0] && e[1] == ed[1]) return false;
			block.emplace(e[0], e[1]);
		}

		queue<pair<int, int>> q;
		q.push({ st[0], st[1] });

		set<pair<int, int>> vis;
		vis.emplace(st[0], st[1]);

		int dr[] = { 0, 1, 0, -1 };
		int dc[] = { 1, 0, -1, 0 };

		int const BOUND = 1e6;
		int const max_level = 50 + 10;
		int level = 0;

		while (!q.empty() && level <= max_level)
		{
			int size = q.size();
			level++;
			while (size--)
			{
				auto x = q.front().first, y = q.front().second;
				q.pop();
				for (int d = 0; d < 4; ++d)
				{
					int nx = x + dr[d], ny = y + dc[d];
					if (nx >= 0 && nx < BOUND && 0 <= ny && ny < BOUND && !vis.count({ nx, ny }) && !block.count({ nx, ny }))
					{
						if (ed[0] == nx && ed[1] == ny) return true;
						vis.emplace(nx, ny);
						q.emplace(nx, ny);
					}
				}
			}
		}
		if (level >= max_level) return true;
		return false;
	}

	//1041. Robot Bounded In Circle
	bool isRobotBounded(string instructions) {
		vector<int> d(4);
		int u = 0;
		for (auto c : instructions)
		{
			if (c == 'G')
			{
				d[u]++;
			}
			else if (c == 'L')
			{
				u = (u + 1) % 4;
			}
			else
			{
				u = (u - 1 + 4) % 4;
			}
		}
		return d[0] == d[2] && d[1] == d[3] || u != 0;
	}

	bool dfs_1042(int u, vector<int>& color, vector<vector<int>>& g)
	{
		vector<int> cnt(4 + 1);
		for (auto& v : g[u])
		{
			if (color[v] != 0) cnt[color[v]] ++;
		}
		for (int i = 1; i <= 4; ++i)
		{
			if (cnt[i] == 0)
			{
				color[u] = i;

				bool f = true;
				for (auto& v : g[u])
				{
					if (!f) break;
					if (color[v] == 0)
					{
						if (dfs_1042(v, color, g) == false)
						{
							f = false;
						}
					}
				}
				if (f) return true;
				color[u] = 0;
			}
		}
		return false;
	}

	//1042. Flower Planting With No Adjacent
	vector<int> gardenNoAdj(int N, vector<vector<int>>& paths) {
		vector<vector<int>> g(N + 1);
		for (auto& e : paths)
		{
			int u = e[0], v = e[1];
			g[u].push_back(v);
			g[v].push_back(u);
		}
		vector<int> color(N + 1);
		for (int i = 1; i <= N; ++i)
		{
			if (color[i] == 0)
			{
				dfs_1042(i, color, g);
			}
		}
		color.erase(color.begin());
		return color;
	}

	//1043. Partition Array for Maximum Sum
	int maxSumAfterPartitioning(vector<int>& a, int k) {
		int n = a.size();
		vector<int> dp(n);
		for (int i = 0; i < n; ++i)
		{
			int mx = 0;
			for (int j = i; i - j + 1 <= k && j >= 0; --j)
			{
				mx = max(mx, a[j]);
				dp[i] = max(dp[i], mx * (i - j + 1) + (j - 1 >= 0 ? dp[j - 1] : 0));
			}
		}
		return dp[n - 1];
	}

	int getcommon(int i, int j, string& s)
	{
		int k = 0;
		while (i + k < s.size() && j + k < s.size() && s[i + k] == s[j + k])
		{
			k++;
		}
		return k;
	}

	//1044. Longest Duplicate Substring
	string longestDupSubstring(string s) {
		int n = s.size();

		bool same = true;
		for (int i = 0; i < n; ++i)
		{
			if (s[i] != s[0])
			{
				same = false;
				break;
			}
		}

		if (same) return s.substr(1);


		auto cmp = [&](int i, int j) {
			int k = 0;
			while (i + k < n && j + k < n)
			{
				if (s[i + k] == s[j + k]) k++;
				else return s[i + k] < s[j + k];
			}
			if (j + k < n) return true;
			return false;
		};
		vector<int> suff(n);
		for (int i = 0; i < n; ++i)
		{
			suff[i] = i;
		}
		sort(suff.begin(), suff.end(), cmp);

		int len = 0;
		int pos = 0;
		for (int i = 0; i + 1 < n; ++i)
		{
			int cur = getcommon(suff[i], suff[i + 1], s);
			if (cur > len)
			{
				len = cur;
				pos = suff[i];
			}
		}
		return s.substr(pos, len);
	}

	//1046. Last Stone Weight
	int lastStoneWeight(vector<int>& a) {
		priority_queue<int> pq(a.begin(), a.end());
		while (pq.size() > 1)
		{
			int x = pq.top(); pq.pop();
			int y = pq.top(); pq.pop();
			if (x == y) continue;
			pq.push(x - y);
		}
		if (pq.empty()) return 0;
		return pq.top();
	}

	//1047. Remove All Adjacent Duplicates In String
	string removeDuplicates(string s) {
		string ans;
		for (auto& c : s)
		{
			if (ans.empty()) ans.push_back(c);
			else
			{
				if (ans.back() == c) ans.pop_back();
				else ans.push_back(c);
			}
		}
		return ans;
	}

	bool judge_1048(string& a, string& b)
	{
		if (b.size() != a.size() + 1) return false;
		int cnt = 0;
		int i = 0;
		for (int j = 0; j < b.size(); ++j)
		{
			if (i < a.size() && a[i] == b[j])
			{
				i++;
			}
			else
			{
				cnt++;
			}
		}
		return i == a.size() && cnt == 1;
	}

	int dfs_1048(int u, vector<vector<int>>& g, vector<int>& cnt)
	{
		if (cnt[u] != -1) return cnt[u];
		if (g[u].empty()) return cnt[u] = 1;
		int ans = 0;
		for (auto& v : g[u])
		{
			ans = max(ans, dfs_1048(v, g, cnt) + 1);
		}
		cnt[u] = ans;
		return ans;
	}

	//1048. Longest String Chain
	int longestStrChain(vector<string>& a) {
		int n = a.size();
		vector<vector<int>> g(n);
		auto cmp = [](string& a, string& b) {
			return a.size() < b.size();
		};
		sort(a.begin(), a.end(), cmp);
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (i == j) continue;
				if (judge_1048(a[i], a[j]))
				{
					g[i].push_back(j);
				}
			}
		}
		vector<int> cnt(n, -1);
		for (int i = 0; i < n; ++i)
		{
			if (cnt[i] == -1)
			{
				dfs_1048(i, g, cnt);
			}
		}
		int ans = 0;
		for (auto& e : cnt) ans = max(ans, e);
		return ans;
	}

	//1040. Moving Stones Until Consecutive II
	vector<int> numMovesStonesII(vector<int>& a) {
		sort(a.begin(), a.end());
		int n = a.size();
		queue<int> q;
		int l = 0, r = 0;
		int mx = max(a[n - 1] - n + 2 - a[1], a[n - 2] - a[0] - n + 2), mi = n;
		for (int i = 0; i < n; ++i)
		{
			q.push(a[i]);
			while (q.size() >= 2 && q.back() - q.front() + 1 > n) q.pop();
			if (q.size() == n - 1 && q.back() - q.front() == n - 2)
				mi = min(mi, 2);
			else
				mi = min(mi, n - (int)q.size());
		}
		return { mi, mx };
	}
};
int main()
{
    Solution sol;
    cout << sol.divisorGame(10) << endl;
	return 0;
}
