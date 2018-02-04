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
};

int main()
{
	Solution sol;
	//cout << sol.canTransform("RXXLRXRXL", "XRLXXRRLX") << endl;
	vector<vector<int>> grid;
	grid = { { 0,1,2,3,4 },{ 24,23,22,21,5 },{ 12,13,14,15,16 },{ 11,17,18,19,20 },{ 10,9,8,7,6 } };
	cout << sol.swimInWater(grid) << endl;
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