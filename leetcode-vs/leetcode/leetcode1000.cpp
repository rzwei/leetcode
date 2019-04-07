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

class Solution {
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

	TreeNode* build(int l, int r, vector<int> & a)
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
		ret->left = build(l + 1, f - 1, a);
		ret->right = build(f, r, a);
		return ret;
	}

	//1008. Construct Binary Search Tree from Preorder Traversal
	TreeNode* bstFromPreorder(vector<int> & preorder) {
		return build(0, preorder.size() - 1, preorder);
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


	int dfs(bitset<10>& vis, int i, int f, vector<int>& num, map<vector<long long>, int> &memo)
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
					ans += dfs(vis, i + 1, 0, num, memo);
					vis[j] = 0;
				}
			}
			if (!vis[num[i]])
			{
				vis[num[i]] = 1;
				ans += dfs(vis, i + 1, f, num, memo);
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
					ans += dfs(vis, i + 1, 0, num, memo);
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
				ans += dfs(vis, i + 1, 1, num, memo);
				vis[num[i]] = 0;
				for (int j = 1; j < num[i]; ++j)
				{
					vis[j] = 1;
					ans += dfs(vis, i + 1, 0, num, memo);
					vis[j] = 0;
				}
			}
			else
			{
				for (int j = 1; j <= 9; ++j)
				{
					vis[j] = 1;
					ans += dfs(vis, i + 1, 0, num, memo);
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

	//5017. Sum of Root To Leaf Binary Numbers
	int sumRootToLeaf(TreeNode * root) {
		return dfs_5017(root, 0);
	}


	bool match5018(string& s, string& p)
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

	//5018. Camelcase Matching
	vector<bool> camelMatch(vector<string> & queries, string pattern) {
		int n = queries.size();
		vector<bool> ans(n);
		for (int i = 0; i < n; ++i) ans[i] = match5018(queries[i], pattern);
		return ans;
	}

	//5019. Video Stitching
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

	//5016. Remove Outermost Parentheses
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
};
int main()
{
	return 0;
}