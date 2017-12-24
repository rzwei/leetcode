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
	int trapRainWater(vector<vector<int>>& heightMap) {
		if (heightMap.size() == 0)
			return 0;
		int row = heightMap.size(), col = heightMap[0].size(), res = 0;
		if (row <= 3)
			return 0;
		for (int k = 1; k < row - 1; k++)
		{
			int l = 0, r = col - 1, h1 = heightMap[k][l], h2 = heightMap[k][r];
			while (l <= r)
			{
				if (h1 < h2)
				{
					int h = min(h1, heightMap[k - 1][l]);
					h = min(h, heightMap[k + 1][l]);

					cout << max(0, h - heightMap[k][l]) << endl;

					res += max(0, h - heightMap[k][l]);

					h1 = max(h1, heightMap[k][l]);
					l++;
				}
				else {
					int h = min(h2, heightMap[k - 1][r]);
					h = min(h, heightMap[k + 1][r]);

					cout << max(0, h - heightMap[k][r]) << endl;

					res += max(0, h - heightMap[k][r]);
					h2 = max(h2, heightMap[k][r]);
					r--;
				}
			}
		}
		return res;
	}
	int dfs_openlock(string cur, int n, set<string> &visited, string &target)
	{
		if (cur == target)
			return n;
		if (visited.find(cur) != visited.end())
			return -1;
		cout << cur << endl;
		visited.insert(cur);
		int ret = -1;
		for (int i = 0; i < 4; i++)
		{
			char old = cur[i];
			char u = ((cur[i] - '0') + 1 + 10) % 10 + '0';
			char l = ((cur[i] - '0') - 1 + 10) % 10 + '0';

			cur[i] = u;
			int r = dfs_openlock(cur, n + 1, visited, target);
			if (r != -1)
			{
				if (ret == -1)
					ret = r;
				else
					ret = min(ret, r);
			}
			cur[i] = l;
			r = dfs_openlock(cur, n + 1, visited, target);
			if (r != -1)
			{
				if (ret == -1)
					ret = r;
				else
					ret = min(ret, r);
			}
		}
		return ret;
	}
	int openLock(vector<string>& deadends, string target) {
		//set<string> visited;
		//for (auto i : deadends)
		//	visited.insert(i);
		//return dfs_openlock("0000", 0, visited, target);

		set<string> deads;
		for (auto i : deadends)
			deads.insert(i);
		queue<string> q;
		set<string> visited;
		q.push("0000");
		int res = 0;
		while (!q.empty())
		{
			queue<string> p;
			int size = q.size();
			while (size-- > 0)
			{
				string cur = q.front();
				q.pop();
				visited.insert(cur);
				if (cur == target)
					return res;
				for (int i = 0; i < 4; i++)
				{

					char u = ((cur[i] - '0') + 1 + 10) % 10 + '0';
					char l = ((cur[i] - '0') - 1 + 10) % 10 + '0';

					char old = cur[i];

					cur[i] = u;
					if (deads.find(cur) == deads.end() && visited.find(cur) == visited.end())
						q.push(cur);

					cur[i] = l;
					if (deads.find(cur) == deads.end() && visited.find(cur) == visited.end())
						q.push(cur);

					cur[i] = old;
				}
			}
			res++;
		}
		return -1;
	}

};


int main()
{
	Solution sol;
	vector<string> deads{ "0201","0101","0102","1212","2002" };
	cout << sol.openLock(deads, "0000") << endl;
	cout << sol.openLock(deads, "0202") << endl;
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