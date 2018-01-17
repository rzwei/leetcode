//#include "stdafx.h"
#include <sstream>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <queue>
#include <climits>
#include <stack>
#include <list>
using namespace std;

struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;

	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

void swap(int &a, int &b) {
	a ^= b;
	b ^= a;
	a ^= b;
}

//146. LRU Cache
class LRUCache {
	LRUCache(int capacity) : _capacity(capacity) {}

	int get(int key) {
		auto it = cache.find(key);
		if (it == cache.end()) return -1;
		touch(it);
		return it->second.first;
	}

	void set(int key, int value) {
		auto it = cache.find(key);
		if (it != cache.end()) touch(it);
		else {
			if (cache.size() == _capacity) {
				cache.erase(used.back());
				used.pop_back();
			}
			used.push_front(key);
		}
		cache[key] = { value, used.begin() };
	}

private:
	typedef list<int> LI;
	typedef pair<int, LI::iterator> PII;
	typedef unordered_map<int, PII> HIPII;

	void touch(HIPII::iterator it) {
		int key = it->first;
		used.erase(it->second.second);
		used.push_front(key);
		it->second.second = used.begin();
	}

	HIPII cache;
	LI used;
	int _capacity;
};


class LRUCache2 {
public:
	list<int> keys;
	unordered_map<int, pair<int, list<int>::iterator>> cache;
	typedef unordered_map<int, pair<int, list<int>::iterator>>::iterator cacheentry;
	int size;

	LRUCache2(int capacity) : size(capacity) {

	}

	int get(int key) {
		if (cache.find(key) == cache.end())
			return -1;
		auto r = cache.find(key);
		touch(r);
		return r->second.first;
	}

	void put(int key, int value) {
		auto it = cache.find(key);
		if (it != cache.end()) {
			touch(it);
			it->second.first = value;
		}
		else {
			if (cache.size() == size) {
				cache.erase(keys.back());
				keys.pop_back();
			}
			keys.push_front(key);
			cache[key] = { value, keys.begin() };
		}
	}

	void touch(cacheentry it) {
		int key = it->first;
		keys.erase(it->second.second);
		keys.push_front(key);
		it->second.second = keys.begin();
	}
};
//381. Insert Delete GetRandom O(1) - Duplicates allowed
class RandomizedCollection {
public:
	unordered_map<int, vector<int>> m;
	vector<pair<int, int>> nums;
	/** Initialize your data structure here. */
	RandomizedCollection() {

	}

	/** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
	bool insert(int val) {
		auto result = m.find(val) == m.end();
		m[val].push_back(nums.size());
		nums.push_back({ val, m[val].size() - 1 });
		return result;
	}

	/** Removes a value from the collection. Returns true if the collection contained the specified element. */
	bool remove(int val) {
		auto result = m.find(val) != m.end();
		if (result)
		{
			int idx = m[val].back();
			auto r = nums.back();
			m[r.first][r.second] = idx;
			nums[idx] = r;
			nums.pop_back();
			m[val].pop_back();
			if (m[val].empty())
				m.erase(val);
		}
		return result;
	}

	/** Get a random element from the collection. */
	int getRandom() {
		return nums[rand() % nums.size()].first;
	}
};
class Iterator {
	struct Data;
	Data *data;
public:
	Iterator(const vector<int> &nums);

	Iterator(const Iterator &iter);

	virtual ~Iterator();

	// Returns the next element in the iteration.
	int next();

	// Returns true if the iteration has more elements.
	bool hasNext() const;
};


class PeekingIterator : public Iterator {
	int m_next;
	bool m_hasnext;
public:
	PeekingIterator(const vector<int> &nums) : Iterator(nums) {
		m_hasnext = Iterator::hasNext();
		if (m_hasnext) m_next = Iterator::next();
	}

	int peek() {
		return m_next;
	}

	int next() {
		int t = m_next;
		m_hasnext = Iterator::hasNext();
		if (m_hasnext) m_next = Iterator::next();
		return t;
	}

	bool hasNext() const {
		return m_hasnext;
	}
};

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
			if (s[i] > s[i + 1]) {
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

	bool dfs_pyramid(string &bottom, string &cur, int idx, set<string> &invalid, map<string, vector<char>> &allowed) {
		cout << bottom << " " << cur << " " << idx << endl;
		if (bottom.size() < 2)
			return true;
		if (idx == cur.size()) {
			string n(cur.size() - 1, 'A');
			return dfs_pyramid(cur, n, 0, invalid, allowed);
		}

		if (idx == 0) {
			string key(2, 'A');
			for (int i = 0; i < bottom.size() - 1; i++) {
				key[0] = bottom[i];
				key[1] = bottom[i + 1];
				if (allowed.find(key) == allowed.end()) {
					invalid.insert(bottom);
					return false;
				}
			}
		}
		string key(2, 'A');
		key[0] = bottom[idx];
		key[1] = bottom[idx + 1];
		for (char c : allowed[key]) {
			cur[idx] = c;
			if (dfs_pyramid(bottom, cur, idx + 1, invalid, allowed))
				return true;
		}
		invalid.insert(bottom);
		return false;
	}

	//756. Pyramid Transition Matrix
	bool pyramidTransition(string bottom, vector<string> &allowed) {
		map<string, vector<char>> states;
		for (auto &i : allowed)
			states[i.substr(0, 2)].push_back(i[2]);
		set<string> invalid;
		string cur(bottom.size() - 1, 'A');
		return dfs_pyramid(bottom, cur, 0, invalid, states);
	}

	//    //12. Integer to Roman
	//    string intToRoman(int num) {
	//        char *c[4][10] = {
	//                {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"},
	//                {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"},
	//                {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"},
	//                {"", "M", "MM", "MMM"}
	//        };
	//        string roman;
	//        roman.append(c[3][num / 1000 % 10]);
	//        roman.append(c[2][num / 100 % 10]);
	//        roman.append(c[1][num / 10 % 10]);
	//        roman.append(c[0][num % 10]);
	//
	//        return roman;
	//    }

	//13. Roman to Integer
	int romanToInt(string s) {
		unordered_map<char, int> T = {
				{'I', 1},
				{'V', 5},
				{'X', 10},
				{'L', 50},
				{'C', 100},
				{'D', 500},
				{'M', 1000}
		};

		int sum = T[s.back()];
		for (int i = s.length() - 2; i >= 0; --i) {
			if (T[s[i]] < T[s[i + 1]]) {
				sum -= T[s[i]];
			}
			else {
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
				++ln;
				++i;
			}
			//read file name
			while (input[i] != '\n' && i < fin) {
				if (input[i] == '.')isFile = true;
				++count;
				++i;
			}
			//calculate
			if (isFile) {
				maxi = max(maxi, level[ln - 1] + count);
			}
			else {
				level[ln] = level[ln - 1] + count + 1;// 1 means '/'
			}
			//reset
			count = 0;
			ln = 1;
			isFile = false;
		}
		return maxi;
	}

	//121. Best Time to Buy and Sell Stock
	int maxProfit(vector<int> &prices) {
		//priority_queue<int> pq;
		//int ans = 0;
		//for (int i = prices.size() - 1; i >= 0; i--) {
		//	if (!pq.empty() && pq.top() > prices[i])
		//		ans = max(ans, pq.top() - prices[i]);
		//	else
		//		pq.push(prices[i]);
		//}
		if (prices.size() <= 1)
			return 0;
		int v = prices[0], ans, N = prices.size();
		for (int i = 0; i < N; i++) {
			ans = max(ans, prices[i] - v);
			v = min(v, prices[i]);
		}
		return ans;
	}

	//122. Best Time to Buy and Sell Stock II
	int maxProfit_two(vector<int> &prices) {
		int ans = 0, N = prices.size();
		for (int i = 1; i < N; i++)
			ans += max(prices[i] - prices[i - 1], 0);
		return ans;
	}

	//123. Best Time to Buy and Sell Stock III
	int maxProfit_three(vector<int> &prices) {
		int hold1 = INT_MIN, hold2 = INT_MIN;
		int release1 = 0, release2 = 0;
		for (int i : prices) {                              // Assume we only have 0 money at first
			release2 = max(release2, hold2 + i);     // The maximum if we've just sold 2nd stock so far.
			hold2 = max(hold2, release1 - i);  // The maximum if we've just buy  2nd stock so far.
			release1 = max(release1, hold1 + i);     // The maximum if we've just sold 1nd stock so far.
			hold1 = max(hold1, -i);          // The maximum if we've just buy  1st stock so far.
		}
		return release2; ///Since release1 is initiated as 0, so release2 will always higher than release1.
	}

	//int dfs_maxProfit(int i, int j, vector<int> &prices, map<pair<int, int>, int> &memo) {
	//	if (i == 0 || j == 0)
	//		return 0;

	//	if (memo.find({ i, j }) != memo.end())
	//		return memo[{i, j}];
	//	int ans = dfs_maxProfit(i, j - 1, prices, memo);
	//	for (int k = 0; k < j; k++)
	//		ans = max(ans, prices[j] - prices[k] + dfs_maxProfit(i - 1, k, prices, memo));
	//	memo[{i, j}] = ans;
	//	return ans;
	//}

	//188. Best Time to Buy and Sell Stock IV
	int maxProfit(int k, vector<int> &prices) {
		int N = prices.size();
		if (k >= prices.size() / 2) {
			int ans = 0;
			for (int i = 1; i < prices.size(); i++)
				ans += max(prices[i] - prices[i - 1], 0);
			return ans;
		}

		vector<vector<int> > dp(k + 1, vector<int>(N + 1, 0));
		for (int i = 1; i <= k; i++) {
			int tmp = -prices[0];
			for (int j = 1; j < N; j++) {
				dp[i][j] = max(dp[i][j - 1], prices[j] + tmp);
				tmp = max(tmp, dp[i - 1][j - 1] - prices[j]);
			}
		}
		return dp[k][N - 1];
		//map<pair<int, int>, int>memo;
		//return dfs_maxProfit(k, prices.size() - 1, prices, memo);
	}

	//714. Best Time to Buy and Sell Stock with Transaction Fee
	int maxProfit(vector<int> &prices, int fee) {
		int s0 = 0, s1 = INT_MIN, tmp = 0;
		for (int p : prices) {
			tmp = s0;
			s0 = max(s0, s1 + p);
			s1 = max(s1, tmp - p - fee);
		}
		return s0;
	}

	//652. Find Duplicate Subtrees
	vector<TreeNode *> findDuplicateSubtrees(TreeNode *root) {
		unordered_map<string, vector<TreeNode *>> map;
		vector<TreeNode *> dups;
		serialize(root, map);
		for (auto it = map.begin(); it != map.end(); it++)
			if (it->second.size() > 1) dups.push_back(it->second[0]);
		return dups;
	}

	string serialize(TreeNode *node, unordered_map<string, vector<TreeNode *>> &map) {
		if (!node) return "";
		string s = "(" + serialize(node->left, map) + to_string(node->val) + serialize(node->right, map) + ")";
		map[s].push_back(node);
		return s;
	}

	//758. Bold Words in String
	string boldWords(vector<string> &words, string S) {
		int N = S.length();
		vector<int> hit(N + 1);
		for (auto &word : words) {
			for (int i = 0; i <= N - word.length(); i++)
				if (S.substr(i, word.length()) == word) {
					hit[i]++;
					hit[i + word.length()]--;
				}

		}
		for (int i = 0; i < N; i++)
			hit[i + 1] += hit[i];
		string ans;
		for (int i = 0; i < N; i++) {
			if (hit[i] > 0) {
				ans += "<b>";
				while (i < N && hit[i] > 0)
					ans.push_back(S[i++]);
				i--;
				ans += "</b>";
			}
			else
				ans.push_back(S[i]);
		}
		return ans;
	}

	//    int findMinStep(string board, string hand) {
	//        int result = INT_MAX;
	//        unordered_map<int, int> dict;
	//        for (auto ch : hand)
	//            ++dict[ch];
	//
	//        result = search(board, dict);
	//        return result == INT_MAX ? -1 : result;
	//
	//    }
	//
	//    int search(string board, unordered_map<int, int> &dict) {
	//        // check the consequence after the last insertion and removal
	//        board = remove(board);
	//        if (board.empty()) {
	//            return 0;
	//        }
	//
	//        int cnt = INT_MAX, j = 0;
	//        // important, i must equal to n to include last char
	//        for (int i = 0; i <= board.size(); ++i) {
	//            if (i < board.size() && board[i] == board[j])
	//                continue;
	//            int need = 3 - (i - j);
	//            if (dict[board[j]] >= need) {
	//                // put a ball only when it can remove balls
	//                dict[board[j]] -= need;
	//                int t = search(board.substr(0, j) + board.substr(i), dict);
	//                if (t != INT_MAX) // important
	//                    cnt = min(t + need, cnt);
	//                dict[board[j]] += need;
	//            }
	//            j = i;
	//        }
	//        return cnt;
	//    }
	//
	//    string remove(string board) {
	//        // check 3 consecutive balls
	//        // after put a ball and erase 3 consecutive balls, check if there are more consecutive balls afterwards
	//        for (int i = 0, j = 0; i <= board.size(); ++i) {
	//            if (i < board.size() && board[i] == board[j])
	//                continue;
	//            if (i - j >= 3)
	//                return remove(board.substr(0, j) + board.substr(i));
	//            else j = i;
	//        }
	//        return board;
	//    }

	int dfs_zumagame(vector<char> board, map<char, int> &hands, int v) {
		int flag = 1;
		vector<char> newboard;
		while (flag) {
			int i = 0, Len = board.size();
			newboard.clear();
			flag = 0;
			while (i < Len) {
				if (i + 1 < Len && board[i] == board[i + 1]) {
					int n = 0, t = i;
					while (i < Len && board[i] == board[t]) {
						i++;
						n++;
					}
					if (n >= 3) {
						flag = 1;
						continue;
					}
					else {
						while (n--) {
							newboard.push_back(board[i - 1]);
						}
					}
				}
				else {
					newboard.push_back(board[i++]);
				}
			}
			board = newboard;
		}
		//        for (auto &i:newboard) {
		//            cout << i;
		//        }
		//        cout << endl;
		if (newboard.size() == 0) {
			return v;
		}
		int Len = newboard.size();
		int const BOUND = INT_MAX - 10000;
		int ans = BOUND;
		for (auto &it : hands) {
			if (it.second > 0) {
				int i = 0;
				while (i < Len) {
					if (newboard[i] == it.first) {
						it.second--;
						//                        cout << "insert " << it.first << endl;
						newboard.insert(newboard.begin() + i, it.first);
						int r = dfs_zumagame(newboard, hands, v + 1);
						if (r != -1) {
							ans = min(ans, r);
						}
						newboard.erase(newboard.begin() + i);
						it.second++;
						while (i < Len && newboard[i] == it.first) {
							i++;
						}
					}
					else {
						i++;
					}
				}
			}
		}
		if (ans >= BOUND) {
			return -1;
		}
		return ans;
	}

	//488. Zuma Game
	int findMinStep(string board, string hand) {
		vector<char> boards;
		for (auto i : board) {
			boards.push_back(i);
		}
		map<char, int> hands;
		for (auto i : hand) {
			hands[i]++;
		}
		return dfs_zumagame(boards, hands, 0);
	}

	//402. Remove K Digits
	string removeKdigits(string num, int k) {
		string ret;
		int len = num.length(), top = 0, i = 0;
		vector<char> st(len);
		for (int i = 0; i < len; i++) {
			char c = num[i];
			while (top > 0 && st[top - 1] > c && k) {
				top--;
				k--;
			}
			st[top++] = c;
		}
		while (i < top && st[i] == '0') {
			i++;
		}
		for (; i < top - k; i++) {
			ret.push_back(st[i]);
		}
		return ret.empty() ? "0" : ret;
	}

	//29. Divide Two Integers
	int divide(int dividend, int divisor) {
		if (!divisor || (dividend == INT_MIN && divisor == -1))
			return INT_MAX;
		int sign = ((dividend < 0) ^ (divisor < 0)) ? -1 : 1;
		long long dvd = labs(dividend);
		long long dvs = labs(divisor);
		int res = 0;
		while (dvd >= dvs) {
			long long temp = dvs, multiple = 1;
			while (dvd >= (temp << 1)) {
				temp <<= 1;
				multiple <<= 1;
			}
			dvd -= temp;
			res += multiple;
		}
		return sign == 1 ? res : -res;
	}

	//135. Candy
	int candy(vector<int> &ratings) {
		int len = ratings.size(), ans = 0;
		if (len <= 1) {
			return len;
		}
		vector<int> dp(len, 1);
		for (int i = 1; i < len; ++i) {
			if (ratings[i] > ratings[i - 1]) {
				dp[i] = dp[i - 1] + 1;
			}
		}
		for (int i = len - 1; i >= 1; i--) {
			if (ratings[i - 1] > ratings[i]) {
				dp[i - 1] = max(dp[i] + 1, dp[i - 1]);
			}
		}
		for (auto i : dp) {
			ans += i;
		}
		return ans;
	}

	bool findnext(int i, long long last, long long target, string &num) {
		cout << i << " " << last << " " << target << endl;
		int len = num.length(), k = 0;
		string ts = to_string(target);
		//if (num[i] == '0')
		//return false;
		for (k = 0; i < len && k < ts.length(); i++, k++) {
			if (num[i] == ts[k])
				continue;
			return false;
		}
		if (k == ts.length()) {
			if (i == len)
				return true;
			else
				return findnext(i, target, last + target, num);
		}
		return false;
	}

	//306. Additive Number
	bool isAdditiveNumber(string num) {
		int len = num.size();
		if (len == 0)
			return false;
		long long left = 0, right;
		for (int i = 0; i < len - 1; i++) {
			left = left * 10 + num[i] - '0';
			if (num[0] == '0' && i > 0)
				continue;
			right = 0;
			for (int j = i + 1; j < len; j++) {
				right = right * 10 + num[j] - '0';
				//cout << left << " " << right << " " << left + right << endl;
				if (num[i + 1] == '0' && j > i + 1)
					continue;
				if (findnext(j + 1, right, left + right, num))
					return true;
			}
		}
		return false;
	}

	//int cmpVector(vector<int> &num1, vector<int> &num2) {
	//	if (num1.size() == num2.size())
	//	{
	//		for (int i = 0; i < num1.size(); i++)
	//		{
	//			if (num1[i] != num2[i])
	//			{
	//				return num1[i] - num2[i];
	//			}
	//		}
	//		return 0;
	//	}
	//	return num1.size() > num2.size() ? 1 : -1;
	//}
	//vector<int> getNumber(int i, int j, int k, vector<int> &num1, vector<int>&num2, map<pair<int, pair<int, int>>, vector<int>*>&memo)
	//{
	//	if (memo.find({ i,{j,k } }) != memo.end())
	//	{
	//		return *memo[{i, { j,k }}];
	//	}
	//	if (k == 0) return vector<int>();
	//	int t = i, maxi = i, maxj = j, len1 = num1.size(), len2 = num2.size(), idxi = i, idxj = j;
	//	vector<int> ans;
	//	while (t < num1.size() && len1 - t + len2 - j >= k)
	//	{
	//		if (num1[t] > num1[maxi])
	//			maxi = t;
	//		t++;
	//	}
	//	t = j;
	//	while (t < num2.size() && len1 - i + len2 - t >= k)
	//	{
	//		if (num2[t] > num2[maxj])
	//			maxj = t;
	//		t++;
	//	}
	//	if (maxi < len1&&maxj < len2) {
	//		if (num1[maxi] > num2[maxj])
	//		{
	//			auto r = getNumber(maxi + 1, j, k - 1, num1, num2, memo);
	//			r.insert(r.begin(), num1[maxi]);
	//			ans = r;
	//		}
	//		else if (num1[maxi] < num2[maxj])
	//		{
	//			auto r = getNumber(i, maxj + 1, k - 1, num1, num2, memo);
	//			r.insert(r.begin(), num2[maxj]);
	//			ans = r;
	//		}
	//		else {
	//			auto l = getNumber(maxi + 1, j, k - 1, num1, num2, memo), r = getNumber(i, maxj + 1, k - 1, num1, num2, memo);
	//			if (cmpVector(l, r) > 0)
	//			{
	//				l.insert(l.begin(), num1[maxi]);
	//				ans = l;
	//			}
	//			else {
	//				r.insert(r.begin(), num2[maxj]);
	//				ans = r;
	//			}
	//		}
	//	}
	//	else if (maxi < len1) {
	//		auto r = getNumber(maxi + 1, j, k - 1, num1, num2, memo);
	//		r.insert(r.begin(), num1[maxi]);
	//		ans = r;
	//	}
	//	else if (maxj < len2)
	//	{
	//		auto r = getNumber(i, maxj + 1, k - 1, num1, num2, memo);
	//		r.insert(r.begin(), num2[maxj]);
	//		ans = r;
	//	}
	//	memo[{i, { j,k }}] = &ans;
	//	return ans;
	//}
	////321. Create Maximum Number
	//vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
	//	map<pair<int, pair<int, int>>, vector<int>*>memo;
	//	return getNumber(0, 0, k, nums1, nums2, memo);
	//}
	vector<int> getMaxNumber(vector<int> &nums, int k) {
		vector<int> ans;
		int len = nums.size();
		int drop = len - k;

		for (int i = 0; i < len; i++) {
			while (drop && !ans.empty() && ans.back() < nums[i]) {
				ans.pop_back();
				drop--;
			}
			ans.push_back(nums[i]);
		}
		ans.resize(k);
		return ans;
	}

	vector<int> getMaxNumber(vector<int> nums1, vector<int> nums2) {
		vector<int> ans;
		while (nums1.size() + nums2.size()) {
			vector<int> &now = nums1 > nums2 ? nums1 : nums2;
			ans.push_back(now[0]);
			now.erase(now.begin());
		}
		return ans;
	}

	//321. Create Maximum Number
	vector<int> maxNumber(vector<int> &nums1, vector<int> &nums2, int k) {
		int n1 = nums1.size(), n2 = nums2.size();
		vector<int> ans;
		for (int k1 = max(k - n2, 0); k1 <= min(k, n1); k1++)
			ans = max(ans, getMaxNumber(getMaxNumber(nums1, k1), getMaxNumber(nums2, k - k1)));
		return ans;
	}

	//324. Wiggle Sort II
	void wiggleSort(vector<int> &nums) {
		int n = nums.size();

		// Find a median.
		auto midptr = nums.begin() + n / 2;
		nth_element(nums.begin(), midptr, nums.end());
		int mid = *midptr;

		// Index-rewiring.
#define A(i) nums[(1+2*(i)) % (n|1)]

		// 3-way-partition-to-wiggly in O(n) time with O(1) space.
		int i = 0, j = 0, k = n - 1;
		while (j <= k) {
			if (A(j) > mid)
				swap(A(i++), A(j++));
			else if (A(j) < mid)
				swap(A(j), A(k--));
			else
				j++;
		}
	}

};


int main() {
	Solution sol;
	//RandomizedCollection  r;
	//r.insert(1);
	//r.remove(1);
	//r.insert(1);
	//vector<int> nums{ 1,5,1,1,6,4 };
	//sol.wiggleSort(nums);
	//for (auto i : nums)
	//	cout << i << " ";
	//cout << endl;

	//vector<int> nums1{ 3, 4, 6, 5 }, nums2{ 9, 1, 2, 5, 8, 3 };
	//nums2 = { 8,9 };
	//nums2 = { 8,9 };
	//auto r = sol.maxNumber(nums1, nums2, 5);
	//for (auto i : r)
	//	cout << i << " ";
	//cout << endl;
	//cout << sol.isAdditiveNumber("112358") << endl;
	//cout << sol.isAdditiveNumber("1991001992") << endl;
	//cout << sol.isAdditiveNumber("121474836472147483648") << endl;
	//LRUCache2 cache(2);
	//cache.put(1, 1);
	//cache.put(2, 2);
	//cout << cache.get(1) << endl;
	//cout << cache.get(2) << endl;
	//cache.put(2, 3);
	//cout << cache.get(1) << endl;
	//cout << cache.get(2) << endl;
	//cout << cache.get(3) << endl;

	//    vector<int> rating;
	//    rating = {1, 3, 4, 56, 3, 2};
	//    cout << sol.candy(rating) << endl;
	//    vector<string> words;
	//    cout << sol.divide(1230, 12) << endl;
	//	cout << sol.removeKdigits("1432219", 3) << endl;
	//	cout << sol.removeKdigits("10", 1) << endl;
	//	cout << sol.removeKdigits("112", 1) << endl;
	//    cout << sol.findMinStep("WBYGWYYGGB", "WR") << endl;
	//    cout << sol.findMinStep("WWRRBBWW", "WRBRW") << endl;
	//    cout << sol.findMinStep("G", "GGGG") << endl;
	//    cout << sol.findMinStep("RBYYBBRRB", "YRBGB") << endl;
	//words = { "ab","bc","cd" };
	//    cout << sol.boldWords(words, "aabcd") << endl;
	//cout << sol.boldWords(words, "aabcd") << endl;
	//vector<int> prices{ 3,3,5,0,0,3,1,4 };
	//prices = { 1,2,4 };
	//prices = { 7,1,5,3,6,4 };
	//cout << sol.maxProfit(prices) << endl;
	//cout << sol.maxProfit_two(prices) << endl;
	//cout << sol.lengthLongestPath("dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext") << endl;
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