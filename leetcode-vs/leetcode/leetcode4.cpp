//
// Created by ren on 18-1-23.
//
#include "stdafx.h"
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

#define __DEBUG
using namespace std;

//// This is the interface that allows for creating nested lists.
//// You should not implement it, or speculate about its implementation
//class NestedInteger {
//public:
//	// Constructor initializes an empty nested list.
//	NestedInteger();
//
//	// Constructor initializes a single integer.
//	NestedInteger(int value) {
//	}
//
//	// Return true if this NestedInteger holds a single integer, rather than a nested list.
//	bool isInteger() const;
//
//	// Return the single integer that this NestedInteger holds, if it holds a single integer
//	// The result is undefined if this NestedInteger holds a nested list
//	int getInteger() const;
//
//	// Set this NestedInteger to hold a single integer.
//	void setInteger(int value);
//
//	// Set this NestedInteger to hold a nested list and adds a nested integer to it.
//	void add(const NestedInteger &ni);
//
//	// Return the nested list that this NestedInteger holds, if it holds a nested list
//	// The result is undefined if this NestedInteger holds a single integer
//	const vector<NestedInteger> &getList() const;
//};

class Solution {
public:
	//227. Basic Calculator II
	int calculateII(string s) {
		int curval = 0, res = 0, Len = s.length(), prev = 0;
		char sign = '+';
		for (int i = 0; i < Len;) {
			curval = 0;
			while (isspace(s[i])) {
				i++;
			}
			while (isdigit(s[i])) {
				curval = curval * 10 + s[i] - '0';
				i++;
			}
			if (sign == '+') {
				res += prev;
				prev = curval;
			}
			if (sign == '-') {
				res += prev;
				prev = -curval;
			}
			if (sign == '*') {
				prev = prev * curval;
			}
			if (sign == '/') {
				prev = prev / curval;
			}
			if (i < Len) {
				sign = s[i];
				i++;
			}
		}
		res += prev;
		return res;
	}

	int dfs_741(int i, int j, int f, vector<vector<int>> &mat,
		vector<vector<vector<int>>> &memo) {
#ifdef __DEBUG
		cout << i << " " << j << " " << f << " " << endl;
#endif
		int N = mat[0].size(), res = -1;
		if (mat[i][j] == -1) {
			return -1;
		}
		if (f == 0 && memo[f][i][j] != -1) {
			return memo[f][i][j];
		}
		if (i == 0 && j == 0 && f == 1) {
			return mat[i][j];
		}
		else if (i == N - 1 && j == N - 1) {
			if (f == 0)
				return dfs_741(i, j, 1, mat, memo);
		}
		int t = mat[i][j];
		mat[i][j] = 0;
		int F = f == 0 ? 1 : -1;
		vector<vector<int>> dirs{ {1, 0},
								 {0, 1} };
		for (auto d : dirs) {
			int nx = i + F * d[0], ny = j + F * d[1];
			if (0 <= nx && nx < N && 0 <= ny && ny < N && mat[i][j] != -1) {
				int r = dfs_741(nx, ny, f, mat, memo);
				if (r >= 0) {
					res = max(res, r + t);
				}
			}
		}
		mat[i][j] = t;
		if (f == 0)
			memo[f][i][j] = res;
		return res;
	}

	//741. Cherry Pickup wa
	int cherryPickup(vector<vector<int>> &grid) {
		int N = grid.size();
		vector<vector<vector<int>>> memo(2, vector<vector<int>>(N, vector<int>(N, -1)));
		int r = dfs_741(0, 0, 0, grid, memo);
		return r == -1 ? 0 : r;
	}

	//126. Word Ladder II
	vector<vector<string> >
		findLadders(string beginWord, string endWord, vector<string> &dictwords) {
		unordered_set<string> dict(dictwords.begin(), dictwords.end());

		vector<vector<string> > paths;
		if (!dict.count(endWord)) {
			return paths;
		}
		vector<string> path(1, beginWord);
		if (beginWord == endWord) {
			paths.push_back(path);
			return paths;
		}
		unordered_set<string> words1, words2;
		words1.insert(beginWord);
		words2.insert(endWord);
		unordered_map<string, vector<string> > nexts;
		bool words1IsBegin = false;
		if (findLaddersHelper(words1, words2, dict, nexts, words1IsBegin))
			getPath(beginWord, endWord, nexts, path, paths);
		return paths;
	}

	bool findLaddersHelper(
		unordered_set<string> &words1,
		unordered_set<string> &words2,
		unordered_set<string> &dict,
		unordered_map<string, vector<string> > &nexts,
		bool &words1IsBegin) {
		words1IsBegin = !words1IsBegin;
		if (words1.empty())
			return false;
		if (words1.size() > words2.size())
			return findLaddersHelper(words2, words1, dict, nexts, words1IsBegin);
		for (auto it = words1.begin(); it != words1.end(); ++it)
			dict.erase(*it);
		for (auto it = words2.begin(); it != words2.end(); ++it)
			dict.erase(*it);
		unordered_set<string> words3;
		bool reach = false;
		for (auto it = words1.begin(); it != words1.end(); ++it) {
			string word = *it;
			for (auto ch = word.begin(); ch != word.end(); ++ch) {
				char tmp = *ch;
				for (*ch = 'a'; *ch <= 'z'; ++(*ch))
					if (*ch != tmp)
						if (words2.find(word) != words2.end()) {
							reach = true;
							words1IsBegin ? nexts[*it].push_back(word) : nexts[word].push_back(*it);
						}
						else if (!reach && dict.find(word) != dict.end()) {
							words3.insert(word);
							words1IsBegin ? nexts[*it].push_back(word) : nexts[word].push_back(*it);
						}
						*ch = tmp;
			}
		}
		return reach || findLaddersHelper(words2, words3, dict, nexts, words1IsBegin);
	}

	void getPath(
		string beginWord,
		string &endWord,
		unordered_map<string, vector<string> > &nexts,
		vector<string> &path,
		vector<vector<string> > &paths) {
		if (beginWord == endWord)
			paths.push_back(path);
		else
			for (auto it = nexts[beginWord].begin(); it != nexts[beginWord].end(); ++it) {
				path.push_back(*it);
				getPath(*it, endWord, nexts, path, paths);
				path.pop_back();
			}
	}

	//456. 132 Pattern
	bool find132pattern(vector<int> &nums) {
		//        int Len = nums.size();
		//        if (Len < 3) {
		//            return false;
		//        }
		//        int m = nums[0];
		//        for (int i = 1; i < Len; i++) {
		//            for (int j = i + 1; j < Len; ++j) {
		//                if (nums[j] > m && nums[j] < nums[i]) {
		//                    return true;
		//                }
		//            }
		//            m = min(m, nums[i]);
		//        }
		//        return false;
		int s3 = INT_MIN;
		stack<int> st;
		for (int i = nums.size() - 1; i >= 0; i--) {
			if (nums[i] < s3) return true;
			else
				while (!st.empty() && nums[i] > st.top()) {
					s3 = st.top();
					st.pop();
				}
			st.push(nums[i]);
		}
		return false;
	}

	//433. Minimum Genetic Mutation
	int minMutation(string start, string end, vector<string> &bank) {
		unordered_set<string> validgenic(bank.begin(), bank.end());
		if (!validgenic.count(end)) {
			return -1;
		}
		unordered_set<string> left{ start }, right{ end }, visited;
		vector<char> chars{ 'A', 'G', 'C', 'T' };
		int ans = 0;
		while (!left.empty() && !right.empty()) {
			unordered_set<string> &l = left, &r = right;
			if (l.size() > r.size()) {
				swap(l, r);
			}
			unordered_set<string> nl;
			for (auto genic : l) {
				if (visited.count(genic)) {
					continue;
				}
				if (r.count(genic)) {
					return ans;
				}
				for (int i = 0; i < genic.size(); i++) {
					char t = genic[i];
					for (char c : chars) {
						if (c == t) {
							continue;
						}
						genic[i] = c;
						if (validgenic.count(genic)) {
							nl.insert(genic);
						}
					}
					genic[i] = t;
				}
				visited.insert(genic);
			}
			ans++;
			l = nl;
		}
		return -1;
	}


	bool dfs_678(int i, int counter, string &s, set<pair<int, int>> &memo) {
		if (memo.find({ i, counter }) != memo.end()) {
			return false;
		}
		if (i == s.size()) {
			return counter == 0;
		}
		if (counter < 0) {
			return false;
		}
		if (counter == 0 && s[i] == ')') {
			return false;
		}
		bool r = false;
		if (s[i] == '(') {
			r = dfs_678(i + 1, counter + 1, s, memo);
		}
		else if (s[i] == ')') {
			r = dfs_678(i + 1, counter - 1, s, memo);
		}
		else if (s[i] == '*') {
			r = dfs_678(i + 1, counter, s, memo) || dfs_678(i + 1, counter + 1, s, memo) ||
				dfs_678(i + 1, counter - 1, s, memo);
		}
		if (!r) {
			memo.insert({ i, counter });
		}
		return r;
	}

	//678. Valid Parenthesis String
	bool checkValidString(string s) {
		set<pair<int, int>> memo;
		return dfs_678(0, 0, s, memo);
	}


	//639. Decode Ways II
	int numDecodings(string s) {
		int Len = s.length();
		if (Len == 0 || s[0] == '0') {
			return 0;
		}
		vector<long long> dp(Len + 1);
		const int MOD = 1000000007;
		dp[0] = s[0] == '*' ? 9 : 1;
		for (int i = 1; i < Len; i++) {
			if (s[i] == '*') {
				if (s[i - 1] == '*') {
					dp[i] = 9 * dp[i - 1] + 15 * (i - 2 >= 0 ? dp[i - 2] : 1);
				}
				else if (s[i - 1] == '1') {
					dp[i] = 9 * dp[i - 1] + 9 * (i - 2 >= 0 ? dp[i - 2] : 1);
				}
				else if (s[i - 1] == '2') {
					dp[i] = 9 * dp[i - 1] + 6 * (i - 2 >= 0 ? dp[i - 2] : 1);
				}
				else {
					dp[i] = 9 * dp[i - 1];
				}
			}
			else if (s[i] != '0') {
				if (s[i - 1] == '*') {
					if ('1' <= s[i] && s[i] <= '6') {
						dp[i] = dp[i - 1] + 2 * (i - 2 >= 0 ? dp[i - 2] : 1);
					}
					else {
						dp[i] = dp[i - 1] + (i - 2 >= 0 ? dp[i - 2] : 1);
					}
				}
				else {
					int v = (s[i - 1] - '0') * 10 + s[i] - '0';
					if (10 <= v && v <= 26) {
						dp[i] = dp[i - 1] + (i - 2 >= 0 ? dp[i - 2] : 1);
					}
					else {
						dp[i] = dp[i - 1];
					}
				}
			}
			else if (s[i] == '0') {
				if (s[i - 1] == '1' || s[i - 1] == '2') {
					dp[i] = (i - 2 >= 0) ? dp[i - 2] : 1;
				}
				else if (s[i - 1] == '*') {
					dp[i] = 2 * (i - 2 >= 0 ? dp[i - 2] : 1);
				}
			}
			dp[i] %= MOD;
		}
		//        for (auto &&item : dp) {
		//            cout << item << " ";
		//        }
		//        cout << endl;
		return dp[Len - 1];
	}

	//605. Can Place Flowers
	bool canPlaceFlowers_dp(vector<int> &flowerbed, int n) {
		int Len = flowerbed.size();
		vector<int> dp(Len + 1);
		if (Len >= 2 && flowerbed[0] == 0 && flowerbed[1] == 0) {
			dp[1] = 1;
		}
		else if (Len == 1 && flowerbed[0] == 0) {
			return true;
		}
		for (int i = 2; i <= Len; i++) {
			int pre = flowerbed[i - 2], cur = flowerbed[i - 1];
			if (cur == 0) {
				if (pre == 1) {
					dp[i] = dp[i - 1];
				}
				else {
					dp[i] = max(dp[i - 1], dp[i - 2] + 1);
				}
			}
			else {
				if (pre == 1) {
					dp[i] = dp[i - 1];
				}
				else {
					dp[i] = dp[i - 2];
				}
			}
		}
#ifdef __DEBUG
		for (auto &&i : dp) {
			cout << i << " ";
		}
		cout << endl;
#endif
		return dp[Len] >= n;
	}

	//605. Can Place Flowers
	bool canPlaceFlowers(vector<int> &flowerbed, int n) {
		int cnt = n, Len = flowerbed.size();
		for (int i = 0; i < Len && cnt; i++) {
			if (flowerbed[i]) {
				continue;
			}
			int next = i + 1 < flowerbed.size() ? flowerbed[i + 1] : 0, prev = i - 1 >= 0 ? flowerbed[i - 1] : 0;
			if (!next && !prev) {
				flowerbed[i] = 1;
				cnt--;
			}
		}
		return cnt == 0;
	}

	//721. Accounts Merge
	vector<vector<string>> accountsMerge(vector<vector<string>> &acts) {
		unordered_map<string, string> owner;
		unordered_map<string, string> parents;
		function<string(string)> find = [&](string s) { return parents[s] == s ? s : find(parents[s]); };
		for (int i = 0; i < acts.size(); i++) {
			for (int j = 1; j < acts[i].size(); j++) {
				parents[acts[i][j]] = acts[i][j];
				owner[acts[i][j]] = acts[i][0];
			}
		}
		for (int i = 0; i < acts.size(); i++) {
			string p = find(acts[i][1]);
			for (int j = 2; j < acts[i].size(); j++) {
				parents[find(acts[i][j])] = p;
			}
		}
		unordered_map<string, set<string>> unions;
		for (int i = 0; i < acts.size(); i++) {
			for (int j = 1; j < acts[i].size(); j++) {
				unions[find(acts[i][j])].insert(acts[i][j]);
			}
		}
		vector<vector<string>> merged;
		for (pair<string, set<string>> p : unions) {
			vector<string> emails(p.second.begin(), p.second.end());
			emails.insert(emails.begin(), owner[p.first]);
			merged.push_back(emails);
		}
		return merged;
	}
	//166. Fraction to Recurring Decimal
	string fractionToDecimal(long long n, long long d) {
		if (n == 0)return "0";
		string res;
		if (n < 0 ^ d < 0) res += '-';
		n = abs(n), d = abs(d);
		long long r = n % d;
		res += to_string(n / d);
		if (r == 0)
			return res;
		res += '.';
		unordered_map<int, int> m;
		for (; r; r %= d)
		{
			if (m.count(r))
			{
				res.insert(res.begin() + m[r], '(');
				res += ')';
				break;
			}
			m[r] = res.size();
			r *= 10;
			res += to_string(abs(r / d));
		}
		return res;
	}


	//NestedInteger build_385(int &l, int &r, string &s)
	//{
	//	if (isdigit(s[l]) || s[l] == '-')
	//	{
	//		int f = 1, v = 0;
	//		if (s[l] == '-')
	//		{
	//			l++;
	//			f = -1;
	//		}
	//		while (l < r&&isdigit(s[l]))
	//			v = v * 10 + s[l++] - '0';
	//		return NestedInteger(f == 1 ? v : -v);
	//	}
	//	else if (s[l] == '[')
	//	{
	//		NestedInteger ret;
	//		int i = l + 1;
	//		r--;
	//		while (i < r)
	//		{
	//			if (isdigit(s[i]) || s[i] == '-')
	//				ret.add(build_385(i, r, s));
	//			else if (s[i] == '[') {
	//				int c = 1, j = i + 1;
	//				while (j < r) {
	//					if (s[j] == ']')
	//						c--;
	//					else if (s[j] == '[')
	//						c++;
	//					if (c == 0)
	//						break;
	//					j++;
	//				}
	//				ret.add(build_385(i, ++j, s));
	//				i = j + 1;
	//			}
	//			else if (s[i] == ',')
	//				i++;
	//		}
	//		return ret;
	//	}
	//	return NestedInteger();
	//}

	////385. Mini Parser
	//NestedInteger deserialize(string s) {
	//	int l = 0, r = s.length();
	//	return build_385(l, r, s);
	//}
	//475. Heaters
	int findRadius(vector<int>& houses, vector<int>& heaters) {
		sort(houses.begin(), houses.end());
		sort(heaters.begin(), heaters.end());
		int  i = 0, j = 0, ans = 0;
		while (i < houses.size())
		{
			while (j < heaters.size() - 1 && abs(heaters[j + 1] - houses[i]) <= abs(heaters[j] - houses[i]))
				j++;
			ans = max(ans, abs(heaters[j] - houses[i]));
			i++;
		}
		return ans;
	}
	int findRadius_binarysearch(vector<int>& houses, vector<int>& heaters) {
		sort(heaters.begin(), heaters.end());
		int ans = 0;
		for (int house : houses)
		{
			int idx = lower_bound(heaters.begin(), heaters.end(), house) - heaters.begin();
			int diff = INT_MAX;
			if (idx < heaters.size())
				diff = min(diff, heaters[idx] - house);
			if (idx > 0)
				diff = min(diff, house - heaters[idx - 1]);
			ans = max(ans, diff);
		}
		return ans;
	}
};


int main() {
	Solution sol;
	vector<int> houses = { 1,2,3,4 }, heaters = {1,4};
	cout << sol.findRadius_binarysearch(houses, heaters) << endl;
	//auto r=sol.deserialize("324");
	//cout << 1 << endl;
	//cout << sol.fractionToDecimal(2, 3) << endl;
	//cout << sol.fractionToDecimal(1, 7) << endl;
	//cout << sol.fractionToDecimal(-1, -2147483648ll) << endl;
	//cout << sol.fractionToDecimal(-2147483648ll, 1) << endl;
	//vector<vector<string>> accounts{{"Hanzo", "Hanzo2@m.co", "Hanzo3@m.co"},
	//                                {"Hanzo", "Hanzo4@m.co", "Hanzo5@m.co"},
	//                                {"Hanzo", "Hanzo0@m.co", "Hanzo1@m.co"},
	//                                {"Hanzo", "Hanzo3@m.co", "Hanzo4@m.co"},
	//                                {"Hanzo", "Hanzo7@m.co", "Hanzo8@m.co"},
	//                                {"Hanzo", "Hanzo1@m.co", "Hanzo2@m.co"},
	//                                {"Hanzo", "Hanzo6@m.co", "Hanzo7@m.co"},
	//                                {"Hanzo", "Hanzo5@m.co", "Hanzo6@m.co"}};
	//auto r = sol.accountsMerge(accounts);
	//for (auto &i:r) {
	//    for (auto &j:i) {
	//        cout << j << " ";
	//    }
	//    cout << endl;
	//}
//    vector<int> nums;
//    nums = {0, 0, 1, 0, 1};
//    cout << sol.canPlaceFlowers(nums, 1) << endl;
//    cout << sol.numDecodings("123") << endl;
//    cout << sol.numDecodings("1*") << endl;
//    cout << sol.numDecodings("*3") << endl;
//    cout << sol.numDecodings("*10*1") << endl;
//    cout << sol.numDecodings("*0**0") << endl;
//    cout << sol.numDecodings("1*72*") << endl;
//    cout << sol.checkValidString(
//            "()(()(*(())()*)(*)))()))*)((()(*(((()())()))()()*)((*)))()))(*)(()()(((()*()()((()))((*((*)()")
//         << endl;
//    vector<string> bank;
//    bank = {"AACCGGTA", "AACCGCTA", "AAACGGTA"};
//    bank = {"AAAACCCC", "AAACCCCC", "AACCCCCC"};
//    cout << sol.minMutation("AACCGGTT", "AAACGGTA", bank) << endl;
//    vector<int> nums{1, 2, 3, 4, 5};
//    nums = {1, 0, 1, -4, -3};
//    nums = {3, 1, 4, 2};
//    nums = {-1, 3, 2, 0};
//    nums = {3, 5, 0, 3, 4};
//    cout << sol.find132pattern(nums) << endl;
//    vector<vector<int>> mat{{0, 1, -1},
//                            {1, 0, 1},
//                            {1, 1, 1}};
//    cout << sol.cherryPickup(mat) << endl;
//    string expression;
//    while (getline(cin, expression)) {
//        cout << sol.calculateII(expression) << endl;
//    }
	return 0;
}