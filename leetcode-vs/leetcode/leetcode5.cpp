#include "stdafx.h"
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
using namespace std;
class Solution
{
public:
	int fun(int N, int K) {
		if (N == 1 && K == 1)
			return 0;
		if (K > (1 << N - 2))
			return !fun(N - 1, K - (1 << N - 2));
		else return fun(N - 1, K);
	}
	//779. K-th Symbol in Grammar
	int kthGrammar(int N, int K) {
		return fun(N, K);
	}

	TreeNode* travel(TreeNode *root, int v, vector<TreeNode*> ans)
	{
		if (!root) return NULL;
		if (ans[0] == nullptr && root->val <= v)
		{
			ans[0] = root;
			auto r = root->right;
			root->right = NULL;
			return r;
		}
		else root->left = travel(root->left, v, ans);
		return root;
	}
	//776. Split BST 
	vector<TreeNode*> splitBST(TreeNode* root, int V) {
		vector<TreeNode *> ans(2);
		ans[1] = travel(root, V, ans);
		return ans;
	}
	//777. Swap Adjacent in LR String
	bool canTransform(string start, string end) {
		int i = 0, j = 0, Len = start.size();
		while (true)
		{
			while (i < Len&&start[i] == 'X')
				i++;
			while (j < Len&&end[j] == 'X')
				j++;
			if (i >= Len && j >= Len)
				break;
			if (i < Len && j < Len)
			{
				if (start[i++] != end[j++])
					return false;
			}
			else return false;
		}

		j = 0;
		for (i = 0; i < Len; i++)
		{
			if (start[i] == 'L')
			{
				while (end[j] != 'L') j++;
				if (i < j++) return false;
			}
		}
		j = 0;
		for (i = 0; i < Len; i++)
		{
			if (start[i] == 'R')
			{
				while (end[j] != 'R') j++;
				if (i > j++) return false;
			}
		}
		return true;
	}

	//778. Swim in Rising Water
	int swimInWater(vector<vector<int>>& grid) {
		int ans = 0, Len = grid.size();
		priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> pq;
		pq.push({ grid[0][0],0,0 });
		unordered_set<int> vis;
		static vector<vector<int>> dirs{ {0,1},{0,-1},{1,0},{-1,0} };
		while (!pq.empty())
		{
			auto c = pq.top();
			pq.pop();
			ans = max(ans, c[0]);
			if (c[1] == Len - 1 && c[2] == Len - 1)
				return ans;
			for (auto &d : dirs)
			{
				int nx = c[1] + d[0], ny = c[2] + d[1];
				if (0 <= nx && nx < Len && 0 <= ny && ny < Len && !vis.count(nx*Len + ny))
				{
					pq.push({ grid[nx][ny],nx,ny });
					vis.insert(nx*Len + ny);
				}
			}
		}
		return ans;
	}
	void travel_min(TreeNode *p, int &pre, int &m)
	{
		if (!p) return;
		travel_min(p->left, pre, m);
		if (pre == INT_MIN)
			pre = p->val;
		else {
			m = min(m, abs(p->val - pre));
			pre = p->val;
		}
		travel_min(p->right, pre, m);
	}
	//783. Minimum Distance Between BST Nodes
	int minDiffInBST(TreeNode* root) {
		int pre = INT_MIN, ans = INT_MAX;
		travel_min(root, pre, ans);
		return ans;
	}
	//781. Rabbits in Forest
	int numRabbits(vector<int>& answers) {
		unordered_map<int, int> m;
		int ans = 0;
		for (auto c : answers)
			if (c == 0)
				ans++;
			else
				m[c]++;
		for (auto p : m)
		{
			int k = p.first, n = p.second;
			ans += (n / (k + 1))*(k + 1);
			if (n % (k + 1))
				ans += k + 1;
		}
		return ans;
	}
	//780. Reaching Points
	bool reachingPoints(int sx, int sy, int tx, int ty) {
		if (tx == ty) return sx == sy && sx == tx;
		while (sx != tx || sy != ty) {
			if (tx < sx || ty < sy) return false;
			if (tx > ty) tx -= max((tx - sx) / ty, 1) * ty;
			else         ty -= max((ty - sy) / tx, 1) * tx;
		}
		return true;
	}

	void swap_cols(vector<vector<int>> &board, int c1, int c2) {
		int n = board.size();
		for (int i = 0; i < n; i++) {
			swap(board[i][c1], board[i][c2]);
		}
	}

	void swap_rows(vector<vector<int>> &board, int r1, int r2) {
		int n = board.size();
		for (int i = 0; i < n; i++) {
			swap(board[r1][i], board[r2][i]);
		}
	}

	bool verify(vector<vector<int>> &board) {
		int n = board.size();
		int b = board[0][0];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if ((i + j) % 2 == 0 && board[i][j] != b) return false;
				if ((i + j) % 2 != 0 && board[i][j] == b) return false;
			}
		}

		return true;
	}

	int can_cols_swap(vector<vector<int>> board, int black) {
		int n = board.size();
		vector<int> blks, whites;
		int moves = 0;
		for (int i = 0; i < n; i++) {
			if (board[0][i] == black && i % 2 != 0) blks.push_back(i);
			if (board[0][i] != black && i % 2 == 0) whites.push_back(i);
		}

		if (blks.size() == whites.size()) {
			moves += blks.size();

			for (int i = 0; i < blks.size(); i++) {
				swap_cols(board, blks[i], whites[i]);
			}
			if (!verify(board)) moves = INT_MAX;
		}
		else moves = INT_MAX;

		return moves;
	}

	int can_rows_swap(vector<vector<int>> board, int black) {
		int moves = 0, n = board.size();
		vector<int> blks, whites;
		for (int i = 0; i < n; i++) {
			if (board[i][0] == black && i % 2 != 0) blks.push_back(i);
			if (board[i][0] != black && i % 2 == 0) whites.push_back(i);
		}

		if (blks.size() == whites.size()) {
			moves += blks.size();

			for (int i = 0; i < blks.size(); i++) {
				swap_rows(board, blks[i], whites[i]);
			}

			int col_moves = min(can_cols_swap(board, 0), can_cols_swap(board, 1));
			if (col_moves == INT_MAX) moves = INT_MAX;
			else moves += col_moves;
		}
		else moves = INT_MAX;

		return moves;
	}
	//782. Transform to Chessboard
	int movesToChessboard(vector<vector<int>>& board) {
		int ans = min(can_rows_swap(board, 0), can_rows_swap(board, 1));
		return ans == INT_MAX ? -1 : ans;
	}
	void dfs_784(string cur, int i, string &pattern, vector<string> &ans)
	{
		while (i < pattern.size() && isdigit(pattern[i]))
			cur += pattern[i++];
		if (i == pattern.size())
		{
			ans.push_back(cur);
			return;
		}
		dfs_784(cur + char(tolower(pattern[i])), i + 1, pattern, ans);
		dfs_784(cur + char(toupper(pattern[i])), i + 1, pattern, ans);

	}
	// 784. Letter Case Permutation
	vector<string> letterCasePermutation(string S) {
		vector<string> ans;
		dfs_784("", 0, S, ans);
		return ans;
	}
	//785. Is Graph Bipartite?
	bool isBipartite(vector<vector<int>>& graph) {
		int len = graph.size();
		vector<int>m(len);
		int const Left = 100, Right = ~Left;
		for (int i = 0; i < len; i++)
		{
			if (m[i] == 0) m[i] = Left;
			for (auto j : graph[i])
			{
				if (m[j] == m[i]) return false;
				m[j] = ~m[i];
			}
		}
		return true;
	}

	//787. Cheapest Flights Within K Stops 
	int findCheapestPrice_floyd(
		int n,
		vector<vector<int>>& flights,
		int src, int dst, int K) {

		vector<vector<int>> dp(n, vector<int>(n, -1));
		for (int i = 0; i < flights.size(); i++)
		{
			int u = flights[i][0], v = flights[i][1], w = flights[i][2];
			dp[u][v] = w;
		}
		for (int k = 1; k <= K; k++)
		{
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					for (int ki = 0; ki < n; ki++)
					{
						if (dp[i][ki] == -1 || dp[ki][j] == -1)
							continue;
						int t = dp[i][ki] + dp[ki][j];
						if (dp[i][j] == -1 || dp[i][j] > t)
							dp[i][j] = t;
					}
				}
			}
		}
		return dp[src][dst];
	}
	int findCheapestPrice(
		int n,
		vector<vector<int>>& flights,
		int src, int dst, int K) {

		vector<int>dis(n, INT_MAX / 2), pre(n, INT_MAX / 2);
		dis[src] = pre[src] = 0;
		for (int i = 0; i <= K; i++)
		{
			for (auto &edge : flights)
			{
				int u = edge[0], v = edge[1], w = edge[2];
				dis[v] = min(dis[v], pre[u] + w);
			}
			pre = dis;
		}
		return dis[dst] < INT_MAX / 2 ? dis[dst] : -1;
	}

	////786. K-th Smallest Prime Fraction
	//vector<int> kthSmallestPrimeFraction(vector<int>& A, int K) {
	//	int len = A.size();
	//	function<bool(vector<int>&, vector<int>&)> fun = [&](vector<int> &a, vector<int> &b) {
	//		//return  A[a[0]] / float(A[a[1]]) > A[b[0]] / float(A[b[1]]);
	//		bool r = A[a[0]] * A[b[1]] > A[a[1]] * A[b[0]];
	//		return r;
	//	};
	//	priority_queue<vector<int>, vector<vector<int>>, decltype(fun)> pq(fun);
	//	for (int i = 0; i < len - 1; i++)
	//	{
	//		pq.push({ i,len - 1 });
	//	}
	//	while (--K)
	//	{
	//		auto top = pq.top();
	//		//cout << A[top[0]] << " " << A[top[1]] << endl;
	//		pq.pop();
	//		if (top[1] - 1 > top[0])
	//			pq.push({ top[0],top[1] - 1 });
	//	}
	//	auto r = pq.top();
	//	return { A[r[0]],A[r[1]] };
	//}
	//786. K-th Smallest Prime Fraction
	vector<int> kthSmallestPrimeFraction(vector<int> &A, int K) {
		int p = 0, q = 1;
		double l = 0, r = 1;

		for (int n = A.size(), cnt = 0; true; cnt = 0, p = 0) {
			double m = (l + r) / 2;

			for (int i = 0, j = n - 1; i < n; i++) {
				while (j >= 0 && A[i] > m * A[n - 1 - j]) j--;
				cnt += (j + 1);

				if (j >= 0 && p * A[n - 1 - j] < q * A[i]) {
					p = A[i];
					q = A[n - 1 - j];
				}
			}

			if (cnt < K) {
				l = m;
			}
			else if (cnt > K) {
				r = m;
			}
			else {
				return { p, q };
			}
		}
	}

	bool vailid(int n, unordered_map<char, char>&m)
	{
		string t = to_string(n);
		string s2 = t;
		for (int i = 0; i < t.size(); i++)
		{
			if (!m.count(t[i]))
				return false;
			s2[i] = m[t[i]];
		}
		return s2 != t;
	}
	int rotatedDigits(int N) {
		unordered_map<char, char> m;
		m['0'] = '0';
		m['1'] = '1';
		m['8'] = '8';
		m['2'] = '5';
		m['5'] = '2';
		m['6'] = '9';
		m['9'] = '6';
		int ans = 0;
		for (int i = 1; i <= N; i++)
			if (vailid(i, m)) ans++;
		return ans;
	}

	bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target) {
		int l = abs(target[0]) + abs(target[1]);
		for (auto &c : ghosts)
		{
			int dis = abs(c[0] - target[0]) + abs(c[1] - target[1]);
			if (dis <= l) return false;
		}
		return true;
	}
	string customSortString(string S, string T) {
		char m[26];
		for (int i = 0; i < S.size(); i++)
			m[S[i] - 'a'] = i;
		auto cmp = [&](char a, char b) {
			return m[a - 'a'] < m[b - 'a'];
		};
		sort(T.begin(), T.end(), cmp);
		return T;
	}
	int numTilings(int N) {
		vector<int> dp(N + 1);;
		dp[1] = 1;
		dp[2] = 2;
		dp[3] = 5;
		int const MOD = 1000000007;
		for (int i = 4; i <= N; i++)
			dp[i] = (dp[i - 3] % MOD + (dp[i - 1] % MOD) * 2 % MOD) % MOD;
		return dp[N];
	}

};

int main()
{
	Solution sol;
	cout << sol.numTilings(30) << endl;
	//string S = "cba", T = "abcd";
	//cout << sol.customSortString(S, T) << endl;
	//vector<vector<int>> ghosts;
	//ghosts = { {1,0},{0,3} };
	//vector<int> target{ 0,1 };
	//target = { 1,0 };
	//ghosts = { {2,0} };
	//cout << sol.escapeGhosts(ghosts, target) << endl;
	//cout << sol.rotatedDigits(10) << endl;
	//vector<int> A;
	//A = { 1,2,3,5 };
	//auto r = sol.kthSmallestPrimeFraction(A, 5);
	//A = { 1,7 };
	//auto r = sol.kthSmallestPrimeFraction(A, 1);
	//cout << r[0] << " " << r[1] << endl;
	//vector<vector<int>> edges;
	//edges = { { 1,2,10 },{ 2,0,7 },{ 1,3,8 },{ 4,0,10 },{ 3,4,2 },{ 4,2,10 },{ 0,3,3 },{ 3,1,6 },{ 2,4,5 } };
	//cout << sol.findCheapestPrice(5, edges, 0, 4, 1) << endl;
	//edges = { { 0,1,100 },{ 1,2,100 },{ 0,2,500 } };
	//edges = { { 0,1,2 },{ 1,2,1 },{ 2,0,10} };
	//cout << sol.findCheapestPrice(3, edges, 0, 2, 1) << endl;
	//cout << sol.findCheapestPrice(3, edges, 1, 2, 1) << endl;
	//vector<vector<int>> graph;
	//graph = { { 1,3 },{ 0,2 },{ 1,3 },{ 0,2 } };
	//graph = { { 1,2,3 },{ 0,2 },{ 0,1,3 },{ 0,2 } };
	//cout << sol.isBipartite(graph) << endl;
	//auto r = sol.letterCasePermutation("a1b2");
	//for (auto &c : r)
	//	cout << c << endl;
	//cout << sol.reachingPoints(1, 1, 3, 5) << endl;
	//cout << sol.reachingPoints(1, 1, 2, 2) << endl;
	//cout << sol.reachingPoints(9, 5, 12, 8) << endl;
	//vector<int> nums;
	//nums = { 10,10,10 };
	//nums = { 3,3,3,3,3 };
	//nums = { 1,1,0,0,0 };
	//cout << sol.numRabbits(nums) << endl;
	//cout << sol.canTransform("RXXLRXRXL", "XRLXXRRLX") << endl;
	//vector<vector<int>> grid;
	//grid = { { 0,1,2,3,4 },{ 24,23,22,21,5 },{ 12,13,14,15,16 },{ 11,17,18,19,20 },{ 10,9,8,7,6 } };
	//cout << sol.swimInWater(grid) << endl;
	//cout << sol.kthGrammar(2, 1) << endl;
	//cout << sol.kthGrammar(4, 5) << endl;
	//vector<int>nums1{ 1,2 }, nums2{ 3,4 };
	//cout << sol.findMedianSortedArrays(nums1, nums2) << endl;
	//cout << sol.lengthOfLongestSubstring("abcabcbb") << endl;
	//cout << sol.hammingDistance(1, 4) << endl;
	//vector<int>nums;
	//nums = { 1,2,3 };
	//cout << sol.combinationSum4(nums, 4) << endl;
	return 0;
}