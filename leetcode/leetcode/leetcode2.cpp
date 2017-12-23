#include "stdafx.h"
#include <sstream>
#include <math.h>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <queue>
using namespace std;
class Solution
{
public:
	int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
		if (matrix.empty())
			return 0;
		int row = matrix.size(), col = matrix[0].size();
		int res = -2147483647;
		for (int l = 0; l < col; l++)
		{
			vector<int>sums(row, 0);
			for (int r = l; r < col; r++) {
				for (int i = 0; i < row; i++)
					sums[i] += matrix[i][r];
				set<int>accSet;
				accSet.insert(0);
				int curMax = -2147483647, curSum = 0;
				for (int sum : sums)
				{
					curSum += sum;
					auto it = accSet.lower_bound(curSum - k);
					if (it != accSet.end())
						curMax = max(curMax, curSum - *it);
					accSet.insert(curSum);
				}
				res = max(res, curMax);
			}
		}
		return res;
	}
	int longestSubstring(string s, int k) {
		return dv_longestSubstring(s, 0, s.length(), k);
	}
	int dv_longestSubstring(string &S, int start, int end, int k)
	{
		if (end - start < k)
			return 0;
		int m[26] = { 0 };
		for (int i = start; i < end; i++)
			m[S[i] - 'a']++;

		for (int i = start; i < end;)
		{
			while (i < end && m[S[i] - 'a'] < k)
				i++;
			if (i == end)
				return 0;

			int j = i;
			while (j < end && m[S[j] - 'a'] >= k)
				j++;

			if (j == end)
				return j - i + 1;
			return max(dv_longestSubstring(S, start, i, k), dv_longestSubstring(S, i + 1, end, k));
		}
		return end - start;
	}
};


int main()
{
	Solution sol;
	cout << sol.longestSubstring("bbaaacbd", 3) << endl;
	return 0;
}