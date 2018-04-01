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

class DSU {
	vector<int> parent;
	vector<int> rank;
	vector<int> sz;
public:
	DSU(int N) :parent(N), rank(N), sz(N, 1) {
		for (int i = 0; i < N; ++i)
			parent[i] = i;
	}

	int find(int x) {
		if (parent[x] != x) parent[x] = find(parent[x]);
		return parent[x];
	}

	void Union(int x, int y) {
		int xr = find(x), yr = find(y);
		if (xr == yr) return;

		if (rank[xr] < rank[yr]) {
			int tmp = yr;
			yr = xr;
			xr = tmp;
		}
		if (rank[xr] == rank[yr])
			rank[xr]++;

		parent[yr] = xr;
		sz[xr] += sz[yr];
	}

	int size(int x) {
		return sz[find(x)];
	}

	int top() {
		return size(sz.size() - 1) - 1;
	}
};
class Trie {
public:
	char v;
	std::unordered_map<char, Trie *> next;

	Trie(char _v) : v(_v) {};

	static void add(Trie *root, std::string &word) {
		auto cur = root;
		for (char c : word) {
			if (!cur->next.count(c))
				cur->next[c] = new Trie(c);
			cur = cur->next[c];
		}
	}

	static bool find(Trie *root, std::string &word) {
		auto cur = root;
		for (auto c : word) {
			if (!cur->next.count(c)) return false;
			cur = cur->next[c];
		}
		return true;
	}

};

class Solution {
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

	TreeNode *travel(TreeNode *root, int v, vector<TreeNode *> ans) {
		if (!root) return NULL;
		if (ans[0] == nullptr && root->val <= v) {
			ans[0] = root;
			auto r = root->right;
			root->right = NULL;
			return r;
		}
		else root->left = travel(root->left, v, ans);
		return root;
	}

	//776. Split BST
	vector<TreeNode *> splitBST(TreeNode *root, int V) {
		vector<TreeNode *> ans(2);
		ans[1] = travel(root, V, ans);
		return ans;
	}

	//777. Swap Adjacent in LR String
	bool canTransform(string start, string end) {
		int i = 0, j = 0, Len = start.size();
		while (true) {
			while (i < Len && start[i] == 'X')
				i++;
			while (j < Len && end[j] == 'X')
				j++;
			if (i >= Len && j >= Len)
				break;
			if (i < Len && j < Len) {
				if (start[i++] != end[j++])
					return false;
			}
			else return false;
		}

		j = 0;
		for (i = 0; i < Len; i++) {
			if (start[i] == 'L') {
				while (end[j] != 'L') j++;
				if (i < j++) return false;
			}
		}
		j = 0;
		for (i = 0; i < Len; i++) {
			if (start[i] == 'R') {
				while (end[j] != 'R') j++;
				if (i > j++) return false;
			}
		}
		return true;
	}

	//778. Swim in Rising Water
	int swimInWater(vector<vector<int>> &grid) {
		int ans = 0, Len = grid.size();
		priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> pq;
		pq.push({ grid[0][0], 0, 0 });
		unordered_set<int> vis;
		static vector<vector<int>> dirs{ {0,  1},
										{0,  -1},
										{1,  0},
										{-1, 0} };
		while (!pq.empty()) {
			auto c = pq.top();
			pq.pop();
			ans = max(ans, c[0]);
			if (c[1] == Len - 1 && c[2] == Len - 1)
				return ans;
			for (auto &d : dirs) {
				int nx = c[1] + d[0], ny = c[2] + d[1];
				if (0 <= nx && nx < Len && 0 <= ny && ny < Len && !vis.count(nx * Len + ny)) {
					pq.push({ grid[nx][ny], nx, ny });
					vis.insert(nx * Len + ny);
				}
			}
		}
		return ans;
	}

	void travel_min(TreeNode *p, int &pre, int &m) {
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
	int minDiffInBST(TreeNode *root) {
		int pre = INT_MIN, ans = INT_MAX;
		travel_min(root, pre, ans);
		return ans;
	}

	//781. Rabbits in Forest
	int numRabbits(vector<int> &answers) {
		unordered_map<int, int> m;
		int ans = 0;
		for (auto c : answers)
			if (c == 0)
				ans++;
			else
				m[c]++;
		for (auto p : m) {
			int k = p.first, n = p.second;
			ans += (n / (k + 1)) * (k + 1);
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
			else ty -= max((ty - sy) / tx, 1) * tx;
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
	int movesToChessboard(vector<vector<int>> &board) {
		int ans = min(can_rows_swap(board, 0), can_rows_swap(board, 1));
		return ans == INT_MAX ? -1 : ans;
	}

	void dfs_784(string cur, int i, string &pattern, vector<string> &ans) {
		while (i < pattern.size() && isdigit(pattern[i]))
			cur += pattern[i++];
		if (i == pattern.size()) {
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
	bool isBipartite(vector<vector<int>> &graph) {
		int len = graph.size();
		vector<int> m(len);
		int const Left = 100, Right = ~Left;
		for (int i = 0; i < len; i++) {
			if (m[i] == 0) m[i] = Left;
			for (auto j : graph[i]) {
				if (m[j] == m[i]) return false;
				m[j] = ~m[i];
			}
		}
		return true;
	}

	//787. Cheapest Flights Within K Stops
	int findCheapestPrice_floyd(
		int n,
		vector<vector<int>> &flights,
		int src, int dst, int K) {

		vector<vector<int>> dp(n, vector<int>(n, -1));
		for (int i = 0; i < flights.size(); i++) {
			int u = flights[i][0], v = flights[i][1], w = flights[i][2];
			dp[u][v] = w;
		}
		for (int k = 1; k <= K; k++) {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					for (int ki = 0; ki < n; ki++) {
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
		vector<vector<int>> &flights,
		int src, int dst, int K) {

		vector<int> dis(n, INT_MAX / 2), pre(n, INT_MAX / 2);
		dis[src] = pre[src] = 0;
		for (int i = 0; i <= K; i++) {
			for (auto &edge : flights) {
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

	bool vailid(int n, unordered_map<char, char> &m) {
		string t = to_string(n);
		string s2 = t;
		for (int i = 0; i < t.size(); i++) {
			if (!m.count(t[i]))
				return false;
			s2[i] = m[t[i]];
		}
		return s2 != t;
	}
	//788. Rotated Digits
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
	//789. Escape The Ghosts
	bool escapeGhosts(vector<vector<int>> &ghosts, vector<int> &target) {
		int l = abs(target[0]) + abs(target[1]);
		for (auto &c : ghosts) {
			int dis = abs(c[0] - target[0]) + abs(c[1] - target[1]);
			if (dis <= l) return false;
		}
		return true;
	}
	//791. Custom Sort String
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
	//790. Domino and Tromino Tiling
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

	bool valid_794(vector<string> &board) {
		for (int i = 0; i < 3; i++) {
			int t = board[i][0], j;
			for (j = 0; t != ' ' && j < 3; j++)
				if (board[i][j] != t) break;
			if (j == 3) return true;
			t = board[0][i];
			for (j = 0; t != ' ' && j < 3; j++)
				if (board[j][i] != t) break;
			if (j == 3) return true;
		}
		int t = board[0][0], j;
		for (j = 0; j < 3 && t != ' '; j++)
			if (board[j][j] != t)break;
		if (j == 3) return true;
		t = board[0][2];
		for (j = 0; t != ' ' && j < 3; j++)
			if (board[j][2 - j] != t) break;
		if (j == 3) return true;
		return false;
	}

	bool dfs_794(int cur, vector<string> &board, vector<vector<int>> &vis, bool old) {
#ifdef __DEBUG
		for (auto c : board)
			cout << c << endl;
		cout << "---------------" << endl;
#endif // __DEBUG
		int f = 0;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				if (board[i][j] != ' ') {
					f = 1;
					break;
				}
			}
		if (f == 0) return true;

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				if (board[i][j] == cur && vis[i][j] == 0) {
					board[i][j] = ' ';
					vis[i][j] = 1;
					bool s = false;
					if (old == true)
						s = valid_794(board);
					bool r = dfs_794(cur == 'X' ? 'O' : 'X', board, vis, s);
					board[i][j] = cur;
					vis[i][j] = 0;
					if (r && (old == true ? s == false : true))
						return true;
				}
			}
		return false;
	}

	//794. Valid Tic-Tac-Toe State
	bool validTicTacToe(vector<string> &board) {
		vector<vector<int>> vis(3, vector<int>(3));
		int x = 0, o = 0;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				if (board[i][j] == ' ') continue;
				if (board[i][j] == 'X') x++;
				else if (board[i][j] == 'O')o++;
			}
		if (o > x || x - o > 1) return false;
		return dfs_794(x > o ? 'X' : 'O', board, vis, valid_794(board));
	}

	//792. Number of Matching Subsequences
	int numMatchingSubseq(string S, vector<string> &words) {
		vector<vector<string>> heads(26);
		for (auto &word : words)
			heads[word[0] - 'a'].push_back(word.substr(1));
		int len = S.length(), ans = 0;
		for (int i = 0; i < len; i++) {
			vector<string> t;
			swap(t, heads[S[i] - 'a']);

			heads[S[i] - 'a'].clear();
			for (auto &c : t) {
				if (c == "") ans++;
				else heads[c[0] - 'a'].push_back(c.substr(1));
			}
		}
		return ans;
	}

	int numSubarrayBoundedMax_(vector<int> &A, int L, int R) {
		return count(A, R) - count(A, L - 1);
	}

	int count(vector<int> &A, int bound) {
		int ans = 0, cur = 0;
		for (int x : A) {
			cur = x <= bound ? cur + 1 : 0;
			ans += cur;
		}
		return ans;
	}

	//795. Number of Subarrays with Bounded Maximum
	int numSubarrayBoundedMax(vector<int> &A, int L, int R) {
		int res = 0, heads = 0, tails = 0;
		for (int val : A) {
			if (L <= val && val <= R) {
				// val is a head. All tails promoted to heads
				heads += tails + 1;
				tails = 0;
				res += heads;
			}
			else if (val < L) {
				// val is a tail, can extend existing subarrays
				tails++;
				res += heads;
			}
			else {
				// combo breaker
				heads = 0;
				tails = 0;
			}
		}
		return res;
	}

	//793. Preimage Size of Factorial Zeroes Function
	int preimageSizeFZF(int K) {
		int lo = K, hi = 10 * K + 1;
		while (lo < hi) {
			int mi = lo + (hi - lo) / 2;
			int zmi = zeta(mi);
			if (zmi == K) return 5;
			else if (zmi < K) lo = mi + 1;
			else hi = mi;
		}
		return 0;
	}

	int zeta(int x) {
		if (x == 0) return 0;
		return x / 5 + zeta(x / 5);
	}
	//796. Rotate String
	bool rotateString(string A, string B) {
		if (A.size() != B.size()) return false;
		if (A.size() == 0) return true;
		int len = A.size();
		for (int i = 0; i < len; i++)
		{
			if (B[i] == A[0])
			{
				int j = 0;
				int k = i;
				while (j < len)
				{
					if (B[k] != A[j]) break;
					j++;
					k = k + 1 == len ? 0 : k + 1;
				}
				if (j == len) return true;
			}
		}
		return false;
	}

	void dfs_allPathsSourceTarget(int cur, vector<int> &path, vector<int> &vis, int target, vector<vector<int>> &grid, vector<vector<int>> &ans)
	{
		if (cur == target) ans.push_back(path);
		for (int v : grid[cur])
		{
			if (vis[v]) continue;
			vis[v] = 1;
			path.push_back(v);
			dfs_allPathsSourceTarget(v, path, vis, target, grid, ans);
			path.pop_back();
			vis[v] = 0;
		}
	}
	// 797. All Paths From Source to Target
	vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
		vector<int> path, vis(graph.size());
		vector<vector<int>> ans;
		vis[0] = 1;
		path.push_back(0);
		dfs_allPathsSourceTarget(0, path, vis, graph.size() - 1, graph, ans);
		return ans;
	}
	//799. Champagne Tower
	double champagneTower(int poured, int query_row, int query_glass) {
		vector<double>pre(100), cur(100);
		if (query_row == 0) return poured >= 1 ? 1 : 0;

		pre[0] = poured;
		int f = 1;
		for (int i = 1; i < 100 && f; i++)
		{
			for (int k = 0; k <= i; k++)
				cur[k] = 0;
			f = 0;
			for (int j = 0; j <= i; j++)
			{
				if (pre[j] <= 1) continue;
				double m = pre[j] - 1;
				pre[j] = 1;
				cur[j] += m / 2;
				cur[j + 1] += m / 2;
				f = 1;
			}
			if (i - 1 == query_row) return pre[query_glass];
			swap(pre, cur);
		}
		return pre[query_glass];
	}
	//798. Smallest Rotation with Highest Score
	int bestRotation(vector<int>& A) {
		int N = A.size();
		vector<int> bad(N);
		for (int i = 0; i < N; ++i) {
			int left = (i - A[i] + 1 + N) % N;
			int right = (i + 1) % N;
			bad[left]--;
			bad[right]++;
			if (left > right)
				bad[0]--;
		}

		int best = -N;

		int ans = 0, cur = 0;
		for (int i = 0; i < N; ++i) {
			cur += bad[i];
			if (cur > best) {
				best = cur;
				ans = i;
			}
		}
		return ans;
	}

	inline char toChar(int v)
	{
		if (v < 10)
			return v + '0';
		else
			return v - 10 + 'a';
	}

	inline int value(char c)
	{
		if ('0' <= c && c <= '9') return c - '0';
		else return c - 'a' + 10;
	}
	//800. Similar RGB Color 
	string similarRGB(string color) {
		int mins = INT_MIN;
		int r = (value(color[1]) << 4) + value(color[2]);
		int g = (value(color[3]) << 4) + value(color[4]);
		int b = (value(color[5]) << 4) + value(color[6]);
		int ansi = 0, ansj = 0, ansk = 0;
		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
				for (int k = 0; k < 16; k++) {
					int dis = -(r - (i << 4) - i)*(r - (i << 4) - i)
						- (g - (j << 4) - j)*(g - (j << 4) - j)
						- (b - (k << 4) - k)*(b - (k << 4) - k);
					if (dis > mins)
					{
						ansi = i;
						ansj = j;
						ansk = k;
						mins = dis;
					}
				}
			}
		}
		string ans;
		ans.push_back('#');
		ans.push_back(toChar(ansi));
		ans.push_back(toChar(ansi));
		ans.push_back(toChar(ansj));
		ans.push_back(toChar(ansj));
		ans.push_back(toChar(ansk));
		ans.push_back(toChar(ansk));
		return ans;
	}
	//801. Minimum Swaps To Make Sequences Increasing 
	int minSwap(vector<int> &A, vector<int> &B) {
		int len = A.size();
		vector<int[2]> dp(len);

		dp[0][0] = 0;
		dp[0][1] = A[0] == B[0] ? 0 : 1;

		for (int i = 1; i < len; i++)
		{
			if (A[i] == B[i])
			{
				dp[i][0] = min(dp[i - 1][0], dp[i - 1][1]);
				dp[i][1] = dp[i][0];
				continue;
			}

			dp[i][0] = INT_MAX;
			dp[i][1] = INT_MAX;

			if (A[i] > B[i - 1] && B[i] > A[i - 1])
			{
				dp[i][0] = min(dp[i - 1][1], dp[i][0]);
				dp[i][1] = min(dp[i - 1][0] + 1, dp[i][1]);
			}
			if (A[i] > A[i - 1] && B[i] > B[i - 1])
			{
				dp[i][0] = min(dp[i - 1][0], dp[i][0]);
				dp[i][1] = min(dp[i - 1][1] + 1, dp[i][1]);
			}
		}
		return min(dp[len - 1][0], dp[len - 1][1]);
	}

	bool dfs_SafeNodes(int u, vector<int> &color, vector<vector<int>> &graph)
	{
		if (color[u] != 0)
			return color[u] == 2;
		color[u] = 1;
		for (auto ni : graph[u])
		{
			if (color[ni] == 2)
				continue;
			if (color[ni] == 1 || !dfs_SafeNodes(ni, color, graph))
				return false;
		}
		color[u] = 2;
		return true;
	}

	vector<int> eventualSafeNodes_dfs(vector<vector<int>>& graph) {
		int N = graph.size();
		vector<int> color(N), ans;
		for (int i = 0; i < N; i++)
			if (dfs_SafeNodes(i, color, graph))
				ans.push_back(i);
		return ans;
	}

	//802. Find Eventual Safe States 
	vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
		int N = graph.size();
		vector<vector<int>> outgoing(N);
		vector<int> diag(N);

		for (int i = 0; i < N; i++)
			for (auto j : graph[i])
			{
				outgoing[j].push_back(i);
				diag[i]++;
			}
		vector<int> q;
		for (int i = 0; i < N; i++)
			if (diag[i] == 0)
				q.push_back(i);
		while (!q.empty())
		{
			int d = q.back();
			q.pop_back();

			for (auto j : outgoing[d])
			{
				diag[j]--;
				if (diag[j] == 0)
					q.push_back(j);
			}
		}
		vector<int> ans;
		for (int i = 0; i < N; i++)
			if (diag[i] == 0)
				ans.push_back(i);
		return ans;
	}

	//void dfs_falling(int i, int j, int color, vector<vector<int>> &grid, vector<vector<bool>> &vis, int m, int n)
	//{
	//	static int dirs[4][2] = { {0,1},{1,0},{0,-1},{-1,0} };
	//	vis[i][j] = color;
	//	for (auto &d : dirs)
	//	{
	//		int x = i + d[0], y = j + d[1];
	//		if (x < 0 || x >= m || y < 0 || y >= n || vis[x][y] != 0 || grid[x][y] == 0)
	//			continue;
	//		dfs_falling(x, y, color, grid, vis, m, n);
	//	}
	//}

	//int falling(vector<vector<int>> &grid)
	//{
	//	int m = grid.size(), n = grid[0].size();
	//	vector<vector<bool>> vis(m, vector<bool>(n, 0));
	//	for (int i = 0; i < n; i++)
	//	{
	//		if (grid[0][i] == 1 && vis[0][i] == 0)
	//			dfs_falling(0, i, 1, grid, vis, m, n);
	//	}
	//	int ret = 0;
	//	for (int i = 0; i < m; i++)
	//		for (int j = 0; j < n; j++)
	//		{
	//			if (vis[i][j] == 0 && grid[i][j] != 0)
	//			{
	//				ret++;
	//				grid[i][j] = 0;
	//			}
	//		}
	//	return ret;
	//}

	//803. Bricks Falling When Hit 
	vector<int> hitBricks(vector<vector<int>>& grid, vector<vector<int>>& hits) {
		int R = grid.size(), C = grid[0].size();
		vector<int> dr = { 1, 0, -1, 0 };
		vector<int> dc = { 0, 1, 0, -1 };

		vector<vector<int>> A(R, vector<int>(C));
		for (int r = 0; r < R; ++r)
			A[r] = grid[r];
		for (auto hit : hits)
			A[hit[0]][hit[1]] = 0;

		DSU dsu(R*C + 1);
		for (int r = 0; r < R; ++r) {
			for (int c = 0; c < C; ++c) {
				if (A[r][c] == 1) {
					int i = r * C + c;
					if (r == 0)
						dsu.Union(i, R*C);
					if (r > 0 && A[r - 1][c] == 1)
						dsu.Union(i, (r - 1) *C + c);
					if (c > 0 && A[r][c - 1] == 1)
						dsu.Union(i, r * C + c - 1);
				}
			}
		}
		int t = hits.size();
		vector<int> ans(t--);

		while (t >= 0) {
			int r = hits[t][0];
			int c = hits[t][1];
			int preRoof = dsu.top();
			if (grid[r][c] == 0) {
				t--;
			}
			else {
				int i = r * C + c;
				for (int k = 0; k < 4; ++k) {
					int nr = r + dr[k];
					int nc = c + dc[k];
					if (0 <= nr && nr < R && 0 <= nc && nc < C && A[nr][nc] == 1)
						dsu.Union(i, nr * C + nc);
				}
				if (r == 0)
					dsu.Union(i, R*C);
				A[r][c] = 1;
				ans[t--] = max(0, dsu.top() - preRoof - 1);
			}
		}
		return ans;
	}
	//806. Number of Lines To Write String
	vector<int> numberOfLines(vector<int> &widths, string S) {
		int curl = 0, curpos = 0;
		for (auto c : S) {
			if (curpos + widths[c - 'a'] < 100) {
				curpos += widths[c - 'a'];
			}
			else if (curpos + widths[c - 'a'] == 100) {
				curl += 1;
				curpos = 0;
			}
			else {
				curl += 1;
				curpos = widths[c - 'a'];
			}
		}
		return { curl + 1, curpos };
	}

	//804. Unique Morse Code Words
	int uniqueMorseRepresentations(vector<string> &words) {
		static vector<string> dict = { ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-",
			".-..",
			"--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--",
			"-..-",
			"-.--", "--.." };
		unordered_set<string> ans;
		for (auto &word : words) {
			string morse;
			for (auto c : word) {
				morse += dict[c - 'a'];
			}
			if (!ans.count(morse)) {
				ans.insert(morse);
			}
		}
		return ans.size();
	}

	//807. Max Increase to Keep City Skyline
	int maxIncreaseKeepingSkyline(vector<vector<int>> &grid) {
		int m = grid.size(), n = grid[0].size();
		int ans = 0;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				int m1 = 0;
				for (int k = 0; k < n; ++k) {
					m1 = max(m1, grid[i][k]);
				}
				int m2 = 0;
				for (int k = 0; k < m; ++k) {
					m2 = max(m2, grid[k][j]);
				}
				ans += min(m1, m2) - grid[i][j];
			}
		}
		return ans;
	}

	bool dfs_805(int len1, int i, int s1, int s, vector<int> &A) {
		if (i == A.size()) return false;
		if (len1 > A.size() / 2) {
			return false;
		}
		if (s1 * (A.size() - len1) == (s - s1) * len1) {
			return true;
		}
		return dfs_805(len1, i + 1, s1, s, A) || dfs_805(len1 + 1, i + 1, s1 + A[i], s, A);
	}

	bool subsetWithk(int i, int ks, int kn, vector<int> &A) {
		if (ks == 0 && kn == 0) return true;
		if (kn == 0) return false;
		for (int j = i; j < A.size(); ++j) {
			if (A[j] <= ks) {
				if (subsetWithk(j + 1, ks - A[j], kn - 1, A)) {
					return true;
				}
			}
		}
		return false;
	}

	//805. Split Array With Same Average
	bool splitArraySameAverage(vector<int> &A) {
		int s = 0, len = A.size();
		for (auto ai : A) {
			s += ai;
		}
		for (int k = 1; k <= len / 2; ++k) {
			if (k * s % len) {
				continue;
			}
			int tk = k * s / len;
			if (subsetWithk(0, tk, k, A)) {
				return true;
			}
		}
		return false;
	}
	//811. Subdomain Visit Count
	vector<string> subdomainVisits(vector<string> &cpdomains) {
		unordered_map<string, int> count;
		for (auto &cpdomain : cpdomains) {
			int n = 0, i = 0, len = cpdomain.size();
			while (i < len) {
				if (isdigit(cpdomain[i])) {
					n = n * 10 + cpdomain[i] - '0';
					i++;
				}
				else {
					break;
				}
			}
			int j = len - 1;
			string sub;
			while (j > i) {
				if (cpdomain[j] != '.') {
					sub = cpdomain[j] + sub;
				}
				else {
					count[sub] += n;
					sub = "." + sub;
				}
				j--;
			}
			count[sub] += n;
		}
		vector<string> ans;
		for (auto &p : count) {
			ans.push_back(to_string(p.second) + " " + p.first);
		}
		return ans;
	}
	bool match(string &s, string &p)
	{
		int lens = s.size(), lenp = p.size(), i = 0, j = 0;
		while (i < lens && j < lenp)
		{
			if (s[i] != p[j]) return false;
			int t = i, n = 0;
			while (t < lens && s[i] == s[t])
			{
				n++;
				t++;
			}
			int n2 = 0, t2 = j;
			while (t2 < lenp && p[t2] == p[j])
			{
				t2++;
				n2++;
			}
			if (!(n == n2 || n >= 3 && n > n2))
				return false;
			if (t == lens)
			{
				return t2 == lenp;
			}
			j = t2;
			i = t;
		}
		return false;
	}
	//809. Expressive Words
	int expressiveWords(string S, vector<string>& words) {
		int ans = 0;
		for (auto &pi : words)
		{
			if (match(S, pi))
				ans++;
		}
		return ans;
	}
	//810. Chalkboard XOR Game
	bool xorGame(vector<int>& nums) {
		int n = 0;
		for (auto &i : nums)
			n ^= i;
		if (n == 0) return true;
		return nums.size() % 2 == 0;
	}
	function<unsigned long(const pair<int, int> &)> hasFun = [](const pair<int, int> &a) {
		return a.first * 101 + a.second;
	};
	double emptyA(int na, int nb, unordered_map<pair<int, int>, double, decltype(hasFun)> &memo)
	{
		if (na <= 0 && nb <= 0) return 0.5;
		if (na <= 0) return 1;
		if (nb <= 0) return 0;
		auto key = make_pair(na, nb);
		if (memo.count(key)) return memo[key];
		double ans = 0;
		ans += emptyA(na - 100, nb, memo);
		ans += emptyA(na - 75, nb - 25, memo);
		ans += emptyA(na - 50, nb - 50, memo);
		ans += emptyA(na - 25, nb - 75, memo);
		ans *= .25;
		memo[key] = ans;
		return ans;
	}
	//808. Soup Servings
	double soupServings(int N) {
		unordered_map<pair<int, int>, double, decltype(hasFun)> memo(0, hasFun);
		return emptyA(N, N, memo);
	}
};
int main() {
	Solution sol;
	vector<int> A;
	A = { 1, 2, 3, 4, 5, 6, 7, 8,10 };
	A = { 1, 3,2 };
	A = { 33,86,88,78,21,76,19,20,88,76,10,25,37,97,58,89,65,59,98,57,50,30,58,5,61,72,23,6 };
	cout << sol.splitArraySameAverage(A) << endl;
	//vector<vector<int>> grid, hits;
	//grid = { { 1,0,0,0 },{ 1,1,1,0 } };
	//hits = { {1,0} };
	//grid = { {1,0,0,0},{1,1,0,0} };
	//hits = { {1,1},{1,0} };
	//auto r = sol.hitBricks(grid, hits);
	//for (auto i : r)
	//	cout << i << " ";
	//cout << endl;
	//cout << sol.champagneTower(1, 1, 1) << endl;
	//vector<int> nums = { 2,3,1,4,0 };
	//cout << sol.bestRotation(nums) << endl;
	//nums = { 4,1,4,0,0 };
	//cout << sol.champagneTower(200, 15, 11) << endl;
	//cout << sol.champagneTower(4, 2, 1) << endl;
	//cout << sol.champagneTower(4, 2, 2) << endl;
	//vector<vector<int>> graph;
	//cout << sol.rotateString("abcde", "cdecab") << endl;
	//vector<int> nums = { 2, 1, 4, 3 };
	//nums = { 2,9,2,5,6 };
	//cout << sol.numSubarrayBoundedMax(nums, 2, 8) << endl;
	//vector<string> words = {"a", "bb", "acd", "ace"};
	//string S = "abcde";
	//cout << sol.numMatchingSubseq(S, words) << endl;
	//vector<string> board = { "XOX", " X ", "   " };
	//cout << sol.validTicTacToe(board) << endl;
	//board = { "XOX", " X ", "   " };
	//cout << sol.validTicTacToe(board) << endl;
	//board = { "XOX", "O O", "XOX" };
	//cout << sol.validTicTacToe(board) << endl;
	//board = { "O  ", " X ", "   " };
	//board = { "XOX","OXO","XXO" };
	//cout << sol.validTicTacToe(board) << endl;
	//cout << sol.numTilings(30) << endl;
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