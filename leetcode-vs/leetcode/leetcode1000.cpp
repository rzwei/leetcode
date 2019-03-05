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

class Solution {
	int const maxn = INT_MAX / 8;
	int dfs_1000(int l, int r, int m, vector<int> &sums, int k, vector<vector<vector<int>>> &memo)
	{
		if ((r - l + 1 - m) % (k - 1)) return maxn;
		if (memo[l][r][m] != -1) return memo[l][r][m];
		if (l == r) return m == 1 ? 0 : maxn;
		if (m == 1) return dfs_1000(l, r, k, sums, k, memo) + sums[r + 1] - sums[l];
		int ans = maxn;
		for (int i = l; i < r; i += k - 1)
		{
			ans = min(ans, dfs_1000(l, i, 1, sums, k, memo) + dfs_1000(i + 1, r, m - 1, sums, k, memo));
		}
		memo[l][r][m] = ans;
		return ans;
	}

	//1000. Minimum Cost to Merge Stones
	int mergeStones(vector<int>& a, int k) {
		int n = a.size();
		vector<int> sums(n + 1);
		for (int i = 0; i < n; ++i)
		{
			sums[i + 1] = sums[i] + a[i];
		}
		vector<vector<vector<int>>> memo(n + 1, vector<vector<int>>(n + 1, vector<int>(n + 1, -1)));
		return dfs_1000(0, n - 1, 1, sums, k, memo);
	}
};
int main()
{
	return 0;
}