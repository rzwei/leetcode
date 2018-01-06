#include "stdafx.h"
#include <sstream>
#include <functional>
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

class Solution {
public:
	int maxSumSubmatrix(vector<vector<int>> &matrix, int k) {
		if (matrix.empty())
			return 0;
		int row = matrix.size(), col = matrix[0].size();
		int res = -2147483647;
		for (int l = 0; l < col; l++) {
			vector<int> sums(row, 0);
			for (int r = l; r < col; r++) {
				for (int i = 0; i < row; i++)
					sums[i] += matrix[i][r];
				set<int> accSet;
				accSet.insert(0);
				int curMax = -2147483647, curSum = 0;
				for (int sum : sums) {
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

	int dv_longestSubstring(string &S, int start, int end, int k) {
		if (end - start < k)
			return 0;
		int m[26] = { 0 };
		for (int i = start; i < end; i++)
			m[S[i] - 'a']++;
		//cout << start << " " << end << endl;
		for (int i = start; i < end; i++) {
			if (m[S[i] - 'a'] > 0 && m[S[i] - 'a'] < k)
				return max(dv_longestSubstring(S, start, i, k), dv_longestSubstring(S, i + 1, end, k));
		}
		return end - start;
	}

	int gcd(int a, int b) {
		if (a < b)
			swap(a, b);
		int t;
		while (b != 0) {
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

	int trap(vector<int> &height) {
		if (height.size() == 0)
			return 0;
		int left = 0, right = height.size() - 1;
		int leftMax = height[left], rightMax = height[right];
		int res = 0;
		while (left <= right) {
			cout << left << " " << leftMax << " " << right << " " << rightMax << endl;

			if (leftMax < rightMax) {
				if (height[left] > leftMax)
					leftMax = height[left];
				else
					res += leftMax - height[left];
				left++;
			}
			else {
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

	//736. Parse Lisp Expression
	int evaluate(string expression) {
		unordered_map<string, int> myMap;
		return help_evaluate(expression, myMap);
	}

	int help_evaluate(string expression, unordered_map<string, int> myMap) {
		if ((expression[0] == '-') || (expression[0] >= '0' && expression[0] <= '9'))
			return stoi(expression);
		else if (expression[0] != '(')
			return myMap[expression];
		//to get rid of the first '(' and the last ')'
		string s = expression.substr(1, expression.size() - 2);
		int start = 0;
		string word = parse_evaluate(s, start);
		if (word == "let") {
			while (true) {
				string variable = parse_evaluate(s, start);
				//if there is no more expression, simply evaluate the variable
				if (start > s.size())
					return help_evaluate(variable, myMap);
				string temp = parse_evaluate(s, start);
				myMap[variable] = help_evaluate(temp, myMap);
			}
		}
		else if (word == "add")
			return help_evaluate(parse_evaluate(s, start), myMap) + help_evaluate(parse_evaluate(s, start), myMap);
		else if (word == "mult")
			return help_evaluate(parse_evaluate(s, start), myMap) * help_evaluate(parse_evaluate(s, start), myMap);
	}

	//function to seperate each expression
	string parse_evaluate(string &s, int &start) {
		int end = start + 1, temp = start, count = 1;
		if (s[start] == '(') {
			while (count != 0) {
				if (s[end] == '(')
					count++;
				else if (s[end] == ')')
					count--;
				end++;
			}
		}
		else {
			while (end < s.size() && s[end] != ' ')
				end++;
		}
		start = end + 1;
		return s.substr(temp, end - temp);
	}

	//738. Monotone Increasing Digits
	int monotoneIncreasingDigits(int N) {
		if (N < 10)
			return N;
		string s = move(to_string(N));
		int n = s.length(), idx = -1, v;
		for (int i = 0; i < n - 1; i++)
			if (s[i] > s[i + 1])
			{
				idx = i;
				v = s[i];
				break;
			}
		if (idx == -1)
			return N;
		while (idx > 0 && s[idx] == v)
			idx--;
		if (s[idx] != v)
			idx++;
		s[idx] -= 1;
		for (int i = idx + 1; i < n; i++)
			s[i] = '9';
		return stoi(s);
	}

	bool dfs_pyramid(string &bottom, string &cur, int idx, set<string> &invalid, map<string, vector<char>> &allowed)
	{
		cout << bottom << " " << cur << " " << idx << endl;
		if (bottom.size() < 2)
			return true;
		if (idx == cur.size())
		{
			string n(cur.size() - 1, 'A');
			return dfs_pyramid(cur, n, 0, invalid, allowed);
		}

		if (idx == 0)
		{
			string key(2, 'A');
			for (int i = 0; i < bottom.size() - 1; i++)
			{
				key[0] = bottom[i];
				key[1] = bottom[i + 1];
				if (allowed.find(key) == allowed.end())
				{
					invalid.insert(bottom);
					return false;
				}
			}
		}
		string key(2, 'A');
		key[0] = bottom[idx];
		key[1] = bottom[idx + 1];
		for (char c : allowed[key])
		{
			cur[idx] = c;
			if (dfs_pyramid(bottom, cur, idx + 1, invalid, allowed))
				return true;
		}
		invalid.insert(bottom);
		return false;
	}

	//756. Pyramid Transition Matrix
	bool pyramidTransition(string bottom, vector<string>& allowed) {
		map<string, vector<char>> states;
		for (auto &i : allowed)
			states[i.substr(0, 2)].push_back(i[2]);
		set<string> invalid;
		string cur(bottom.size() - 1, 'A');
		return dfs_pyramid(bottom, cur, 0, invalid, states);
	}
	//12. Integer to Roman
	string intToRoman(int num) {
		char* c[4][10] = {
		{ "","I","II","III","IV","V","VI","VII","VIII","IX" },
		{ "","X","XX","XXX","XL","L","LX","LXX","LXXX","XC" },
		{ "","C","CC","CCC","CD","D","DC","DCC","DCCC","CM" },
		{ "","M","MM","MMM" }
		};
		string roman;
		roman.append(c[3][num / 1000 % 10]);
		roman.append(c[2][num / 100 % 10]);
		roman.append(c[1][num / 10 % 10]);
		roman.append(c[0][num % 10]);

		return roman;
	}
	//13. Roman to Integer
	int romanToInt(string s) {
		unordered_map<char, int> T = {
			{ 'I' , 1 },
		{ 'V' , 5 },
		{ 'X' , 10 },
		{ 'L' , 50 },
		{ 'C' , 100 },
		{ 'D' , 500 },
		{ 'M' , 1000 }
		};

		int sum = T[s.back()];
		for (int i = s.length() - 2; i >= 0; --i)
		{
			if (T[s[i]] < T[s[i + 1]])
			{
				sum -= T[s[i]];
			}
			else
			{
				sum += T[s[i]];
			}
		}

		return sum;
	}

	//388. Longest Absolute File Path
	int lengthLongestPath(string input) {
		//string cur;
		//int level = -1, Len = input.size(), i = 0, ans = 0, curLen = 0, n = 0;
		//vector<string > path;
		//while (i <= Len)
		//{
		//	if (i < Len &&input[i] != '\n' && input[i] != '\t')
		//		cur.push_back(input[i++]);
		//	else {
		//		if (n - level == 1 || level == -1)
		//		{
		//			curLen += cur.length();
		//			path.push_back(cur);
		//			if (cur.find('.') != string::npos)
		//				ans = max(ans, curLen + n);
		//			cur.clear();
		//			level++;
		//		}
		//		else {
		//			while (level >= n)
		//			{
		//				curLen -= path.back().length();
		//				path.pop_back();
		//				level--;
		//			}
		//			path.push_back(cur);
		//			curLen += cur.length();
		//			if (cur.find('.') != string::npos)
		//				ans = max(ans, curLen + n);
		//			cur.clear();
		//			level++;
		//		}
		//		//for (auto &i : path)
		//		//	cout << i << " ";
		//		//cout << endl;
		//		if (i == Len)
		//			break;
		//		if (i < Len && input[i] == '\n')
		//		{
		//			n = 0;
		//			while (i < Len && input[++i] == '\t')
		//				n++;
		//		}

		//	}
		//}
		//return ans;
		int maxi = 0, count = 0, ln = 1;
		bool isFile = false;
		vector<int> level(200);
		level[0] = 0;
		for (int i = 0, fin = input.size(); i < fin; ++i) {
			//find which level
			while (input[i] == '\t') {
				++ln; ++i;
			}
			//read file name
			while (input[i] != '\n'&&i < fin) {
				if (input[i] == '.')isFile = true;
				++count; ++i;
			}
			//calculate
			if (isFile) {
				maxi = max(maxi, level[ln - 1] + count);
			}
			else {
				level[ln] = level[ln - 1] + count + 1;// 1 means '/'
			}
			//reset
			count = 0; ln = 1; isFile = false;
		}
		return maxi;
	}
};


int main() {
	Solution sol;
	cout << sol.lengthLongestPath("dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext") << endl;
	//cout << sol.lengthLongestPath("a") << endl;
	//vector<string> allowed{ "XYD", "YZE", "DEA", "FFF" };
	//allowed = { "XXX", "XXY", "XYX", "XYY", "Y1XZ" };
	//cout << sol.pyramidTransition("XXYX", allowed) << endl;
	//cout << sol.pyramidTransition("XYZ", allowed) << endl;
	//cout << sol.monotoneIncreasingDigits(110) << endl;
	//cout << sol.evaluate("(add 1 2)") << endl;
	//    vector<int> nums{1, 3, 2, 3, 1};
		//vector<int> nums{ 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647 };
	//    cout << sol.reversePairs(nums) << endl;
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