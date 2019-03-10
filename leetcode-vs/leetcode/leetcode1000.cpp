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
};
int main()
{
	return 0;
}