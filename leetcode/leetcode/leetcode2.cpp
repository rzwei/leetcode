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
		int res = -2147483648;
		for (int l = 0; l < col; l++)
		{
			vector<int>sums(row, 0);
			for (int r = l; r < col; r++) {
				for (int i = 0; i < row; i++)
					sums[i] += matrix[i][r];
				set<int>accSet;
				accSet.insert(0);
				int curMax = -2147483648, curSum = 0;
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
};


int main()
{
	return 0;
}