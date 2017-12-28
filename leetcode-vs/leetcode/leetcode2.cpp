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
		//cout << start << " " << end << endl;
		for (int i = start; i < end; i++)
		{
			if (m[S[i] - 'a'] > 0 && m[S[i] - 'a'] < k)
				return max(dv_longestSubstring(S, start, i, k), dv_longestSubstring(S, i + 1, end, k));
		}
		return end - start;
	}
	int gcd(int a, int b)
	{
		if (a < b)
			swap(a, b);
		int t;
		while (b != 0)
		{
			t = a % b;
			a = b;
			b = t;
		}
		return a;
	}
	bool canMeasureWater(int x, int y, int z) {
		if (x + y < z)
			return false;
		if (x == z || y == z || x + y == z)
			return true;
		return z % gcd(x, y) == 0;
	}
	int trap(vector<int>& height) {
		if (height.size() == 0)
			return 0;
		int left = 0, right = height.size() - 1;
		int leftMax = height[left], rightMax = height[right];
		int res = 0;
		while (left <= right)
		{

			if (leftMax < rightMax)
			{
				if (height[left] > leftMax)
					leftMax = height[left];
				else
					res += leftMax - height[left];
				left++;
			}
			else
			{
				if (height[right] > rightMax)
					rightMax = height[right];
				else
					res += rightMax - height[right];
				right--;
			}
		}
		return res;
	}
	vector<double> medianSlidingWindow(vector<int>& nums, int k) {
		// 480. Sliding Window Median
		multiset<int> window();
	}
};


int main()
{
	Solution sol;
	//cout << sol.longestSubstring("bbaaacbd", 3) << endl;
	//Solution sol;
	//vector<vector<int>> nums(4, vector<int>());
	//nums[0] = vector<int>{ 1,4,3,1,3,2 };
	//nums[1] = vector<int>{ 3,2,1,3,2,4 };
	//nums[2] = vector<int>{ 2,3,3,2,3,1 };
	//cout << sol.trapRainWater(nums) << endl;


	//vector<int>nums{ 0,1,0,2,1,0,1,3,2,1,2,1 };
	//vector<int>nums{ 3,2,1,3,2,4 };
	//cout << sol.trap(nums) << endl;
	return 0;
}