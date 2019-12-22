#pragma once
#include <vector>
#include <algorithm>
#include <queue>
class BIT
{
	int lowbit(int x)
	{
		return x & (-x);
	}
	int n;
public:
	std::vector<int> a;
	BIT(int n) : a(n + 1, 0), n(n)
	{

	}
	int add(int i, int v)
	{
		while (i <= n)
		{
			a[i] = std::max(a[i], v);
			i += lowbit(i);
		}
		return 0;
	}
	int query(int i)
	{
		int ans = 0;
		while (i > 0)
		{
			ans = std::max(ans, a[i]);
			i -= lowbit(i);
		}
		return ans;
	}
};


struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class NumMatrix {
public:
	std::vector<std::vector<int>> sums;
	int n, m;
	NumMatrix(std::vector<std::vector<int>> a) {
		n = a.size();
		if (n == 0)
		{
			a.push_back({ 0 });
			return;
		}
		m = a[0].size();
		sums.assign(n + 1, std::vector<int>(m + 1));
		for (int i = 1; i <= n; ++i)
		{
			int cur = 0;
			for (int j = 1; j <= m; ++j)
			{
				cur += a[i - 1][j - 1];
				sums[i][j] = sums[i - 1][j] + cur;
			}
		}
	}

	int sumRegion(int row1, int col1, int row2, int col2) {
		return sums[row2 + 1][col2 + 1] - sums[row2 + 1][col1] - sums[row1][col2 + 1] + sums[row1][col1];
	}
};


struct ListNode {
	int val;
	ListNode* next;
	ListNode(int x) : val(x), next(NULL) {}
};
