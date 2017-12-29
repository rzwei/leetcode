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
			cout << left << " " << leftMax << " " << right << " " << rightMax << endl;

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
	//int reversePairs(vector<int> &nums) {
	//	//        int ans = 0, len = nums.size();
	//	//        vector<int64_t> vs;
	//	//        multiset<int64_t> prev;
	//	//        for (int i = 0; i < len; i++) {
	//	//            int64_t v = (int64_t) nums[i] * 2;
	//	//            auto it = upper_bound(vs.begin(), vs.end(), v);
	//	//            ans += vs.end() - it;
	//	//            vs.insert(lower_bound(vs.begin(), vs.end(), nums[i]), nums[i]);
	//	//        }
	//	//        return ans;
	//	return merge_reversePairs(nums, 0, nums.size() - 1);
	//}

	//int merge_reversePairs(vector<int> nums, int s, int e) {
	//	if (s >= e)
	//		return 0;
	//	int mid = (s + e) / 2;
	//	int count = merge_reversePairs(nums, s, mid) + merge_reversePairs(nums, mid + 1, e);
	//	int i = s, j = mid + 1, p = mid + 1, k = 0;
	//	vector<int> merge(e - s + 1, 0);
	//	while (i <= mid) {
	//		while (p <= e && nums[i] > 2 * nums[p])p++;
	//		count += p - mid - 1;
	//		while (j <= e && nums[i] >= nums[j])
	//			merge[k++] = nums[j++];
	//		merge[k++] = nums[i++];
	//	}
	//	while (j <= e) merge[k++] = nums[j++];
	//	for (i = s; i <= e; i++)
	//		nums[i] = merge[i - s];
	//	for (auto i : nums)
	//		cout << i << " ";
	//	cout << endl;
	//	return count;
	//}
	//493. Reverse Pairs
	int reversePairs(vector<int> &nums) {

		vector<int> cache(nums.size(), 0);
		return merge_reversePairs(nums.begin(), nums.end());
	}

	//    int merge_reversePairs(vector<int> &nums, int s, int e, vector<int> &merge) {
	//        if (s >= e)
	//            return 0;
	//        int mid = (s + e) / 2;
	//        int count = merge_reversePairs(nums, s, mid, merge) + merge_reversePairs(nums, mid + 1, e, merge);
	//        int i = s, j = mid + 1, p = mid + 1, k = 0;
	//        while (i <= mid) {
	//            while (p <= e && nums[i] > 2L * nums[p])p++;
	//            count += p - mid - 1;
	//            while (j <= e && nums[i] >= nums[j])
	//                merge[k++] = nums[j++];
	//            merge[k++] = nums[i++];
	//        }
	//        while (j <= e) merge[k++] = nums[j++];
	//        for (i = s; i <= e; i++)
	//            nums[i] = merge[i - s];
	//        return count;
	//    }
	int merge_reversePairs(vector<int>::iterator begin, vector<int>::iterator end) {
		if (end - begin <= 1)
			return 0;
		auto mid = begin + (end - begin) / 2;
		int count = merge_reversePairs(begin, mid) + merge_reversePairs(mid, end);
		auto i = begin, p = mid;
		while (i < mid) {
			while (p < end && *i > 2L * *p) p++;
			count += p - mid;
			i++;
		}
		inplace_merge(begin, mid, end);
		return count;
	}
};


int main()
{
	Solution sol;
	vector<int> nums{ 1, 3, 2, 3, 1 };

	//vector<int> nums{ 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647 };
	cout << sol.reversePairs(nums) << endl;
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