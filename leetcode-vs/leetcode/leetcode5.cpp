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
	int findCheapestPrice(
		int n,
		vector<vector<int>>& flights,
		int src, int dst, int K) {
		vector<vector<int>> graph(n, vector<int>(n, INT_MAX));
		for (int i = 0; i < n; i++)
		{
			int u = flights[i][0], v = flights[i][1], w = flights[i][2];
			graph[u][v] = w;
		}
	}
};

int main()
{
	Solution sol;
	vector<vector<int>> graph;
	graph = { { 1,3 },{ 0,2 },{ 1,3 },{ 0,2 } };
	graph = { { 1,2,3 },{ 0,2 },{ 0,1,3 },{ 0,2 } };
	cout << sol.isBipartite(graph) << endl;
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