//
// Created by ren on 18-1-23.
//
#include "stdafx.h"
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

using namespace std;
#define __DEBUG

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
struct Trie {
    vector<int> words; // index of words
    vector<Trie *> children;

    Trie() {
        children = vector<Trie *>(26, nullptr);
    }

    // Thanks to @huahualeetcode for adding this in case of memory leak
    ~Trie() {
        for (int i = 0; i < 26; i++) {
            if (children[i]) {
                delete children[i];
            }
        }
    }

    void add(const string &word, size_t begin, int index) {
        words.push_back(index);
        if (begin < word.length()) {
            if (!children[word[begin] - 'a']) {
                children[word[begin] - 'a'] = new Trie();
            }
            children[word[begin] - 'a']->add(word, begin + 1, index);
        }
    }

    vector<int> find(const string &prefix, size_t begin) {
        if (begin == prefix.length()) {
            return words;
        } else {
            if (!children[prefix[begin] - 'a']) {
                return {};
            } else {
                return children[prefix[begin] - 'a']->find(prefix, begin + 1);
            }
        }
    }
};

class WordFilter {
public:
    WordFilter(vector<string> words) {
        forwardTrie = new Trie();
        backwardTrie = new Trie();
        for (int i = 0; i < words.size(); i++) {
            string word = words[i];
            forwardTrie->add(word, 0, i);
            reverse(word.begin(), word.end());
            backwardTrie->add(word, 0, i);
        }
    }

    int f(string prefix, string suffix) {
        auto forwardMatch = forwardTrie->find(prefix, 0);
        reverse(suffix.begin(), suffix.end());
        auto backwardMatch = backwardTrie->find(suffix, 0);
        // search from the back
        auto fIter = forwardMatch.rbegin(), bIter = backwardMatch.rbegin();
        while (fIter != forwardMatch.rend() && bIter != backwardMatch.rend()) {
            if (*fIter == *bIter) {
                return *fIter;
            } else if (*fIter > *bIter) {
                fIter++;
            } else {
                bIter++;
            }
        }
        return -1;
    }

private:
    Trie *forwardTrie, *backwardTrie;
};

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
                        } else if (!reach && dict.find(word) != dict.end()) {
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
        unordered_set<string> left{start}, right{end}, visited;
        vector<char> chars{'A', 'G', 'C', 'T'};
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
        if (memo.find({i, counter}) != memo.end()) {
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
        } else if (s[i] == ')') {
            r = dfs_678(i + 1, counter - 1, s, memo);
        } else if (s[i] == '*') {
            r = dfs_678(i + 1, counter, s, memo) || dfs_678(i + 1, counter + 1, s, memo) ||
                dfs_678(i + 1, counter - 1, s, memo);
        }
        if (!r) {
            memo.insert({i, counter});
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
                } else if (s[i - 1] == '1') {
                    dp[i] = 9 * dp[i - 1] + 9 * (i - 2 >= 0 ? dp[i - 2] : 1);
                } else if (s[i - 1] == '2') {
                    dp[i] = 9 * dp[i - 1] + 6 * (i - 2 >= 0 ? dp[i - 2] : 1);
                } else {
                    dp[i] = 9 * dp[i - 1];
                }
            } else if (s[i] != '0') {
                if (s[i - 1] == '*') {
                    if ('1' <= s[i] && s[i] <= '6') {
                        dp[i] = dp[i - 1] + 2 * (i - 2 >= 0 ? dp[i - 2] : 1);
                    } else {
                        dp[i] = dp[i - 1] + (i - 2 >= 0 ? dp[i - 2] : 1);
                    }
                } else {
                    int v = (s[i - 1] - '0') * 10 + s[i] - '0';
                    if (10 <= v && v <= 26) {
                        dp[i] = dp[i - 1] + (i - 2 >= 0 ? dp[i - 2] : 1);
                    } else {
                        dp[i] = dp[i - 1];
                    }
                }
            } else if (s[i] == '0') {
                if (s[i - 1] == '1' || s[i - 1] == '2') {
                    dp[i] = (i - 2 >= 0) ? dp[i - 2] : 1;
                } else if (s[i - 1] == '*') {
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
        } else if (Len == 1 && flowerbed[0] == 0) {
            return true;
        }
        for (int i = 2; i <= Len; i++) {
            int pre = flowerbed[i - 2], cur = flowerbed[i - 1];
            if (cur == 0) {
                if (pre == 1) {
                    dp[i] = dp[i - 1];
                } else {
                    dp[i] = max(dp[i - 1], dp[i - 2] + 1);
                }
            } else {
                if (pre == 1) {
                    dp[i] = dp[i - 1];
                } else {
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
        for (; r; r %= d) {
            if (m.count(r)) {
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
    int findRadius(vector<int> &houses, vector<int> &heaters) {
        sort(houses.begin(), houses.end());
        sort(heaters.begin(), heaters.end());
        int i = 0, j = 0, ans = 0;
        while (i < houses.size()) {
            while (j < heaters.size() - 1 && abs(heaters[j + 1] - houses[i]) <= abs(heaters[j] - houses[i]))
                j++;
            ans = max(ans, abs(heaters[j] - houses[i]));
            i++;
        }
        return ans;
    }

    int findRadius_binarysearch(vector<int> &houses, vector<int> &heaters) {
        sort(heaters.begin(), heaters.end());
        int ans = 0;
        for (int house : houses) {
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

    void dfs_216(int s, int k, int n, vector<int> &vis, vector<int> &path, vector<vector<int>> &ans) {
        if (k == 0) {
            if (n == 0)
                ans.push_back(path);
            return;
        }
        for (int i = s; i <= 9; i++)
            if (vis[i] == 0 && n >= i) {
                vis[i] = 1;
                path.push_back(i);
                dfs_216(i + 1, k - 1, n - i, vis, path, ans);
                path.pop_back();
                vis[i] = 0;
            }
    }

    //216. Combination Sum III
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<int> vis(10), path;
        vector<vector<int>> ans;
        dfs_216(1, k, n, vis, path, ans);
        return ans;
    }

    //771. Jewels and Stones
    int numJewelsInStones(string J, string S) {
        set<char> jewels(J.begin(), J.end());
        int ans = 0;
        for (auto c : S)
            if (jewels.count(c))
                ans++;
        return ans;
    }

    //775. Global and Local Inversions
    bool isIdealPermutation(vector<int> &A) {
        int Len = A.size();
        vector<int> dp(Len + 1);
        int e = -1;
        for (int i = 0; i < Len; i++)
            if (A[i] - i < -1 || A[i] - i > 1)
                return false;
        return true;
    }

    vector<string> getSwap(string &s) {
        int x, y;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                if (s[i * 3 + j] == '0') {
                    x = i;
                    y = j;
                    break;
                }
        vector<string> ret;
        static vector<vector<int>> dirs{{0,  1},
                                        {0,  -1},
                                        {-1, 0},
                                        {1,  0}};
        for (auto &d : dirs) {
            int nx = x + d[0], ny = y + d[1];
            if (0 <= nx && nx < 2 && 0 <= ny && ny < 3) {
                string t = s;
                swap(t[nx * 3 + ny], t[x * 3 + y]);
                ret.push_back(t);
            }
        }
        return ret;
    }

    //773. Sliding Puzzle
    int slidingPuzzle(vector<vector<int>> &board) {
        unordered_set<string> left, right;
        string start;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                start.push_back(board[i][j] + '0');
        if (start == "123450")
            return 0;
        left.insert(start);
        right.insert("123450");
        unordered_set<string> vis;
        int ans = 0;
        while (!left.empty() && !right.empty()) {
            if (left.size() > right.size())
                swap(left, right);
            ans++;
            unordered_set<string> tmp;
            for (auto cur : left) {
                if (vis.count(cur))
                    continue;
                vis.insert(cur);
                for (auto &s : getSwap(cur)) {
                    if (right.count(s))
                        return ans;
                    tmp.insert(s);
                }
            }
            left = tmp;
        }
        return -1;
    }

    class CustomClass {
    public:
        CustomClass(int x) : num(x), denom(1), val(x) {}

        bool operator<(const CustomClass &o) const {
            return val < o.val;
        }

        double num;
        double denom;
        double val;
    };

    //774. Minimize Max Distance to Gas Station
    double minmaxGasDist(vector<int> &stations, int K) {
        double res = 0;
        double total = 0;
        priority_queue<CustomClass> gap;
        for (int i = 1; i < stations.size(); i++) {
            int diff = stations[i] - stations[i - 1];
            total += diff;
            gap.push(CustomClass(diff));
        }
        total /= (K + 1);
        for (int i = 0; i < K;) {
            auto top = gap.top();
            gap.pop();
            while (i < K && (top.val >= gap.top().val || top.val > total)) {
                i++;
                top.denom++;
                top.val = top.num / top.denom;
            }
            gap.push(top);
        }
        return gap.top().val;
    }

    double minmaxGasDist_(vector<int> &st, int K) {
        int count, N = st.size();
        float left = 0, right = st[N - 1] - st[0], mid;

        while (left + 0.00001 < right) {
            mid = (left + right) / 2;
            count = 0;
            for (int i = 0; i < N - 1; ++i)
                count += ceil((st[i + 1] - st[i]) / mid) - 1;
            if (count > K) left = mid;
            else right = mid;
        }
        return right;
    }

    //761. Special Binary String
    string makeLargestSpecial(string S) {
        int count = 0, i = 0;
        vector<string> res;
        for (int j = 0; j < S.size(); ++j) {
            if (S[j] == '1') count++;
            else count--;
            if (count == 0) {
                res.push_back('1' + makeLargestSpecial(S.substr(i + 1, j - i - 1)) + '0');
                i = j + 1;
            }
        }
        sort(res.begin(), res.end(), greater<string>());
        string res2 = "";
        for (i = 0; i < res.size(); ++i) res2 += res[i];
        return res2;
    }

    //741. Cherry Pickup
    int cherryPickup(vector<vector<int>> &grid) {
        int n = grid.size();
        // dp holds maximum # of cherries two k-length paths can pickup.
        // The two k-length paths arrive at (i, k - i) and (j, k - j),
        // respectively.
        vector<vector<int>> dp(n, vector<int>(n, -1));

        dp[0][0] = grid[0][0]; // length k = 0

        // maxK: number of steps from (0, 0) to (n-1, n-1).
        const int maxK = 2 * (n - 1);

        for (int k = 1; k <= maxK; k++) { // for every length k
            vector<vector<int>> curr(n, vector<int>(n, -1));

            // one path of length k arrive at (i, k - i)
            for (int i = 0; i < n && i <= k; i++) {
                if (k - i >= n) continue;
                // another path of length k arrive at (j, k - j)
                for (int j = 0; j < n && j <= k; j++) {
                    if (k - j >= n) continue;
                    if (grid[i][k - i] < 0 || grid[j][k - j] < 0) { // keep away from thorns
                        continue;
                    }

                    int cherries = dp[i][j]; // # of cherries picked up by the two (k-1)-length paths.

                    // See the figure below for an intuitive understanding
                    if (i > 0) cherries = std::max(cherries, dp[i - 1][j]);
                    if (j > 0) cherries = std::max(cherries, dp[i][j - 1]);
                    if (i > 0 && j > 0) cherries = std::max(cherries, dp[i - 1][j - 1]);

                    // No viable way to arrive at (i, k - i)-(j, k-j).
                    if (cherries < 0) continue;

                    // Pickup cherries at (i, k - i) and (j, k -j ) if i != j.
                    // Otherwise, pickup (i, k-i).
                    cherries += grid[i][k - i] + (i == j ? 0 : grid[j][k - j]);

                    curr[i][j] = cherries;
                }
            }
            dp = std::move(curr);
        }
        return std::max(dp[n - 1][n - 1], 0);
    }

    //741. Cherry Pickup
    int cherryPickup_(vector<vector<int>> &grid) {
        int N = grid.size();
        vector<vector<int>> dp(N, vector<int>(N, -1));
        dp[0][0] = grid[0][0];
        for (int k = 1; k <= 2 * (N - 1); k++) {
            vector<vector<int>> curr(N, vector<int>(N, -1));
            for (int i = 0; i < N; i++) {
                if (k - i >= N || k - i < 0)
                    continue;
                for (int j = 0; j < N; j++) {
                    if (k - j >= N || k - j < 0)
                        continue;
                    if (grid[i][k - i] < 0 || grid[j][k - j] < 0)
                        continue;

                    int cherry = dp[i][j];
                    if (j > 0)cherry = max(cherry, dp[i][j - 1]);
                    if (i > 0)cherry = max(cherry, dp[i - 1][j]);
                    if (i > 0 && j > 0)cherry = max(cherry, dp[i - 1][j - 1]);
                    if (cherry < 0)
                        continue;
                    int val = grid[i][k - i];
                    if (i != j)
                        val += grid[j][k - j];
                    curr[i][j] = cherry + val;
                }
            }
            dp = move(curr);
        }
        return dp[N - 1][N - 1];
    }

    int dp_741(int r1, int c1, int r2, vector<vector<int>> &grid, vector<vector<vector<int>>> &memo) {
        int c2 = r1 + c1 - r2;
        int N = grid.size();
        if (r1 == N || r2 == N || c1 == N || c2 == N || grid[r1][c1] == -1 || grid[r2][c2] == -1)
            return INT_MIN;

        if (r1 == N - 1 && c1 == N - 1 && r2 == N - 1)
            return grid[r1][c1];

        if (memo[r1][c1][r2] != INT_MIN)
            return memo[r1][c1][r2];

        int val = grid[r1][c1];
        if (r1 != r2)
            val += grid[r2][c2];
        int ret = dp_741(r1 + 1, c1, r2, grid, memo);
        ret = max(ret, dp_741(r1 + 1, c1, r2 + 1, grid, memo));
        ret = max(ret, dp_741(r1, c1 + 1, r2, grid, memo));
        ret = max(ret, dp_741(r1, c1 + 1, r2 + 1, grid, memo));
        if (ret == INT_MIN)
            ret++;
        else
            ret += val;
        memo[r1][c1][r2] = ret;
        return ret;
    }

    //741. Cherry Pickup
    int cherryPickup_dfs(vector<vector<int>> &grid) {
        int N = grid.size();
        vector<vector<vector<int>>> memo(N, vector<vector<int>>(N, vector<int>(N, INT_MIN)));
        return max(dp_741(0, 0, 0, grid, memo), 0);
    }

    const string validIPv6Chars = "0123456789abcdefABCDEF";

    bool isValidIPv4Block(string &block) {
        int num = 0;
        if (block.size() > 0 && block.size() <= 3) {
            for (int i = 0; i < block.size(); i++) {
                char c = block[i];
                // special case: if c is a leading zero and there are characters left
                if (!isalnum(c) || (i == 0 && c == '0' && block.size() > 1))
                    return false;
                else {
                    num *= 10;
                    num += c - '0';
                }
            }
            return num <= 255;
        }
        return false;
    }

    bool isValidIPv6Block(string &block) {
        if (block.size() > 0 && block.size() <= 4) {
            for (int i = 0; i < block.size(); i++) {
                char c = block[i];
                if (validIPv6Chars.find(c) == string::npos)
                    return false;
            }
            return true;
        }
        return false;
    }

    //468. Validate IP Address
    string validIPAddress(string IP) {
        string ans[3] = {"IPv4", "IPv6", "Neither"};
        stringstream ss(IP);
        string block;
        // ipv4 candidate
        if (IP.substr(0, 4).find('.') != string::npos) {
            for (int i = 0; i < 4; i++) {
                if (!getline(ss, block, '.') || !isValidIPv4Block(block))
                    return ans[2];
            }
            return ss.eof() ? ans[0] : ans[2];
        }
            // ipv6 candidate
        else if (IP.substr(0, 5).find(':') != string::npos) {
            for (int i = 0; i < 8; i++) {
                if (!getline(ss, block, ':') || !isValidIPv6Block(block))
                    return ans[2];
            }
            return ss.eof() ? ans[1] : ans[2];
        }

        return ans[2];
    }

    //420. Strong Password Checker
    int strongPasswordChecker(string s) {
        int deleteTarget = max(0, (int) s.length() - 20), addTarget = max(0, 6 - (int) s.length());
        int toDelete = 0, toAdd = 0, toReplace = 0, needUpper = 1, needLower = 1, needDigit = 1;

        for (int l = 0, r = 0; r < s.length(); r++) {
            if (isupper(s[r])) { needUpper = 0; }
            if (islower(s[r])) { needLower = 0; }
            if (isdigit(s[r])) { needDigit = 0; }

            if (r - l == 2) {
                if (s[l] == s[l + 1] && s[l + 1] == s[r]) {
                    if (toAdd < addTarget) { toAdd++, l = r; }
                    else { toReplace++, l = r + 1; }
                } else { l++; }
            }
        }
        if (s.length() <= 20) { return max(addTarget + toReplace, needUpper + needLower + needDigit); }

        toReplace = 0;
        vector<unordered_map<int, int>> lenCnts(3);
        for (int l = 0, r = 0, len; r <= s.length(); r++) {
            if (r == s.length() || s[l] != s[r]) {
                if ((len = r - l) > 2) { lenCnts[len % 3][len]++; }
                l = r;
            }
        }

        for (int i = 0, numLetters, dec; i < 3; i++) {
            for (auto it = lenCnts[i].begin(); it != lenCnts[i].end(); it++) {
                if (i < 2) {
                    numLetters = i + 1, dec = min(it->second, (deleteTarget - toDelete) / numLetters);
                    toDelete += dec * numLetters, it->second -= dec;
                    if (it->first - numLetters > 2) { lenCnts[2][it->first - numLetters] += dec; }
                }
                toReplace += (it->second) * ((it->first) / 3);
            }
        }

        int dec = (deleteTarget - toDelete) / 3;
        toReplace -= dec, toDelete -= dec * 3;
        return deleteTarget + max(toReplace, needUpper + needLower + needDigit);
    }

    //466. Count The Repetitions
    int getMaxRepetitions(string s1, int n1, string s2, int n2) {
        int i = 0, j = 0, ans = 0, c2 = n2, c1 = 0, Len1 = s1.size(), Len2 = s2.size();
        while (c1 < n1) {
            if (s1[i] == s2[j]) {
                i++;
                j++;
            } else
                i++;

            if (i == Len1) {
                i = 0;
                c1++;
            }
            if (j == Len2) {
                c2--;
                j = 0;
                if (c2 == 0) {
                    c2 = n2;
                    ans++;
                }
            }
        }
        return ans;
    }

    //479. Largest Palindrome Product
    int largestPalindrome(int n) {
        if (n == 1)
            return 9;
        int maxNum = pow(10, n) - 1;
        int minNum = maxNum / 10;
        for (int cur = maxNum; cur > minNum; cur--) {
            string curStr = to_string(cur);
            string revStr(curStr);
            reverse(revStr.begin(), revStr.end());
            long long toCheck = stoll(curStr + revStr);
            // cout<<toCheck<<endl;
            for (long long otherCur = maxNum; otherCur * otherCur >= toCheck; otherCur--)
                if (toCheck % otherCur == 0) {
                    // cout<<toCheck<<endl;
                    return toCheck % 1337;
                }
        }
        return 0;
    }

    //685. Redundant Connection II
    vector<int> findRedundantDirectedConnection(vector<vector<int>> &edges) {
        int Len = edges.size();
        vector<int> graph(Len + 1), candA, candB;
        for (auto &edge : edges) {
            int &l = edge[0], &r = edge[1];
            if (graph[r] == 0)
                graph[r] = l;
            else {
                candA = {graph[r], r};
                candB = edge;
                r = 0;
            }
        }
        for (int i = 1; i <= Len; i++)
            graph[i] = i;
        function<int(int)> fun = [&](int v) { return v == graph[v] ? v : fun(graph[v]); };
        for (auto &edge : edges) {
            if (edge[1] == 0)
                continue;
            int u = edge[0], v = edge[1], pu = fun(u);
            if (pu == v) {
                if (candA.empty()) return edge;
                return candA;
            }
            graph[v] = pu;
        }
        return candB;
    }

    //564. Find the Closest Palindrome
    string nearestPalindromic(string n) {
        int l = n.size();
        set<long> candidates;
        // biggest, one more digit, 10...01
        candidates.insert(long(pow(10, l)) + 1);
        // smallest, one less digit, 9...9 or 0
        candidates.insert(long(pow(10, l - 1)) - 1);
        // the closest must be in middle digit +1, 0, -1, then flip left to right
        long prefix = stol(n.substr(0, (l + 1) / 2));
        for (long i = -1; i <= 1; ++i) {
            string p = to_string(prefix + i);
            string pp = p + string(p.rbegin() + (l & 1), p.rend());
            candidates.insert(stol(pp));
        }
        long num = stol(n), minDiff = LONG_MAX, diff, minVal;
        candidates.erase(num);
        for (long val : candidates) {
            diff = abs(val - num);
            if (diff < minDiff) {
                minDiff = diff;
                minVal = val;
            } else if (diff == minDiff) {
                minVal = min(minVal, val);
            }
        }
        return to_string(minVal);
    }

    int calstep(long long n, long long n1, long long n2) {
        int steps = 0;
        while (n1 <= n) {
            steps += min(n + 1, n2) - n1;
            n1 *= 10;
            n2 *= 10;
        }
        return steps;
    }

    //440. K - th Smallest in Lexicographical Order
    int findKthNumber(int n, int k) {
        int curr = 1;
        k = k - 1;
        while (k > 0) {
            int steps = calstep(n, curr, curr + 1);
            if (steps <= k) {
                curr += 1;
                k -= steps;
            } else {
                curr *= 10;
                k -= 1;
            }
        }
        return curr;
    }

    int bfs_675(int i, int j, int ti, int tj, vector<vector<int>> &forest) {
        if (i == ti && j == tj)
            return 0;
        int m = forest.size(), n = forest[0].size(), ans = 0;
        queue<pair<int, int>> q;
        static vector<vector<int>> dirs{{1,  0},
                                        {-1, 0},
                                        {0,  1},
                                        {0,  -1}};
        vector<vector<int>> vis(m, vector<int>(n));
        q.push({i, j});
        vis[i][j] = 1;
        while (!q.empty()) {
            int qsize = q.size();
            ans += 1;
            while (qsize--) {
                int x = q.front().first, y = q.front().second;
                q.pop();
                for (auto &d : dirs) {
                    int nx = x + d[0], ny = y + d[1];
                    if (nx < 0 || nx >= m || ny < 0 || ny >= n || vis[nx][ny] || forest[nx][ny] == 0)
                        continue;
                    if (nx == ti && ny == tj)
                        return ans;
                    q.push({nx, ny});
                    vis[nx][ny] = 1;
                }
            }
        }
        return -1;
    }

    //675. Cut Off Trees for Golf Event
    int cutOffTree(vector<vector<int>> &forest) {
        int m = forest.size(), n = forest[0].size();
        vector<vector<int>> pq;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (forest[i][j] > 1)
                    pq.push_back({forest[i][j], i, j});
        sort(pq.begin(), pq.end());
        int ans = 0, x = 0, y = 0;
        for (int i = 0; i < pq.size(); i++) {
            auto t = pq[i];
            int r = bfs_675(x, y, t[1], t[2], forest);
            if (r == -1)
                return -1;
            x = t[1];
            y = t[2];
            ans += r;
        }
        return ans;
    }

    //749. Contain Virus
    int containVirus(vector<vector<int>> &grid) {
        int ans = 0;
        while (true) {
            int walls = process(grid);
            if (walls == 0) break; // No more walls to build
            ans += walls;
        }
        return ans;
    }

    int process(vector<vector<int>> &grid) {
        int m = grid.size(), n = grid[0].size();
        // cnt is max area to be affected by a single virus region; ans is corresponding walls
        int cnt = 0, ans = 0, color = -1, row = -1, col = -1;
        // visited virus as 1, visited 0 using different color to indicate being affected by different virus
        vector<vector<int>> visited(m, vector<int>(n, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 && visited[i][j] == 0) {
                    int walls = 0, area = dfs(grid, visited, i, j, color, walls);
                    if (area > cnt) {
                        ans = walls;
                        cnt = area;
                        row = i;
                        col = j;
                    }
                    color--;
                }
            }
        }
        // set this virus region inactive
        buildWall(grid, row, col);
        // propagate other virus by 1 step
        visited = vector<vector<int>>(m, vector<int>(n, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 && visited[i][j] == 0)
                    spread(grid, visited, i, j);
            }
        }
        return ans;
    }

    int dfs(vector<vector<int>> &grid, vector<vector<int>> &visited, int row, int col, int color, int &walls) {
        int m = grid.size(), n = grid[0].size(), ans = 0;
        if (row < 0 || row >= m || col < 0 || col >= n) return 0;
        if (grid[row][col] == 0) {
            walls++;
            if (visited[row][col] == color) return 0;
            visited[row][col] = color;
            return 1;
        }
        // grid[row][col] could be -1, inactive virus
        if (visited[row][col] == 1 || grid[row][col] != 1) return 0;
        visited[row][col] = 1;
        vector<int> dir = {-1, 0, 1, 0, -1};
        for (int i = 0; i < 4; i++)
            ans += dfs(grid, visited, row + dir[i], col + dir[i + 1], color, walls);
        return ans;
    }

    void buildWall(vector<vector<int>> &grid, int row, int col) {
        int m = grid.size(), n = grid[0].size();
        if (row < 0 || row >= m || col < 0 || col >= n || grid[row][col] != 1) return;
        grid[row][col] = -1; //set inactive
        vector<int> dir = {-1, 0, 1, 0, -1};
        for (int i = 0; i < 4; i++)
            buildWall(grid, row + dir[i], col + dir[i + 1]);
    }

    void spread(vector<vector<int>> &grid, vector<vector<int>> &visited, int row, int col) {
        int m = grid.size(), n = grid[0].size();
        if (row < 0 || row >= m || col < 0 || col >= n || visited[row][col] == 1) return;
        if (grid[row][col] == 0) {
            grid[row][col] = 1;
            visited[row][col] = 1;
        } else if (grid[row][col] == 1) {
            visited[row][col] = 1;
            vector<int> dir = {-1, 0, 1, 0, -1};
            for (int i = 0; i < 4; i++)
                spread(grid, visited, row + dir[i], col + dir[i + 1]);
        }
    }

    //743. Network Delay Time
    int networkDelayTime(vector<vector<int>> &times, int N, int K) {
        N++;
        vector<vector<pair<int, int>>> graph(N);
        for (auto &edge : times) {
            int u = edge[0], v = edge[1], w = edge[2];
            graph[u].push_back({v, w});
        }

        vector<int> minvs(N, -1);
        minvs[K] = 0;
        vector<int> vis(N);
        vis[K] = 1;

        for (auto &edge : graph[K]) {
            int v = edge.first, w = edge.second;
            minvs[v] = w;
        }

        while (true) {
            int minv = -1;
            for (int i = 1; i < N; i++)
                if (minvs[i] != -1 && vis[i] == 0) {
                    if (minv == -1)
                        minv = i;
                    else if (minvs[i] < minvs[minv])
                        minv = i;
                }
            if (minv == -1)
                break;
            vis[minv] = 1;
            for (auto &edge : graph[minv]) {
                int v = edge.first, w = edge.second;
                if (v != K && (minvs[v] == -1 || minvs[v] > minvs[minv] + w))
                    minvs[v] = minvs[minv] + w;
            }
        }
        int ans = 0;
        for (int i = 1; i < N; i++)
            if (i != K && minvs[i] == -1)
                return -1;
            else
                ans = max(ans, minvs[i]);
        return ans;
    }

    //591. Tag Validator
    bool isValid(string code) {
        int i = 0;
        return validTag(code, i) && i == code.size();
    }

    bool validTag(string s, int &i) {
        int j = i;
        string tag = parseTagName(s, j);
        if (tag.empty()) return false;
        if (!validContent(s, j)) return false;
        int k = j + tag.size() + 2; // expecting j = pos of "</" , k = pos of '>'
        if (k >= s.size() || s.substr(j, k + 1 - j) != "</" + tag + ">") return false;
        i = k + 1;
        return true;
    }

    string parseTagName(string s, int &i) {
        if (s[i] != '<') return "";
        int j = s.find('>', i);
        if (j == string::npos || j - 1 - i < 1 || 9 < j - 1 - i) return "";
        string tag = s.substr(i + 1, j - 1 - i);
        for (char ch : tag) {
            if (ch < 'A' || 'Z' < ch) return "";
        }
        i = j + 1;
        return tag;
    }

    bool validContent(string s, int &i) {
        int j = i;
        while (j < s.size()) {
            if (!validText(s, j) && !validCData(s, j) && !validTag(s, j)) break;
        }
        i = j;
        return true;
    }

    bool validText(string s, int &i) {
        int j = i;
        while (i < s.size() && s[i] != '<') { i++; }
        return i != j;
    }

    bool validCData(string s, int &i) {
        if (s.find("<![CDATA[", i) != i) return false;
        int j = s.find("]]>", i);
        if (j == string::npos) return false;
        i = j + 3;
        return true;
    }

    //335. Self Crossing
    bool isSelfCrossing(vector<int> &x) {
        int l = x.size();
        if (l <= 3) return false;

        for (int i = 3; i < l; i++) {
            if (x[i] >= x[i - 2] && x[i - 1] <= x[i - 3]) return true;  //Fourth line crosses first line and onward
            if (i >= 4) {
                if (x[i - 1] == x[i - 3] && x[i] + x[i - 4] >= x[i - 2])
                    return true; // Fifth line meets first line and onward
            }
            if (i >= 5) {
                if (x[i - 2] - x[i - 4] >= 0 && x[i] >= x[i - 2] - x[i - 4] && x[i - 1] >= x[i - 3] - x[i - 5] &&
                    x[i - 1] <= x[i - 3])
                    return true;  // Sixth line crosses first line and onward
            }
        }
        return false;
    }

    //770. Basic Calculator IV
    vector<string> basicCalculatorIV(string expression, vector<string> &evalvars, vector<int> &evalints) {
        unordered_map<string, int> mp;
        int n = evalvars.size();
        // create a map for variable value pairs
        for (int i = 0; i < n; ++i) mp[evalvars[i]] = evalints[i];
        // helper function is recursion using implicit stack
        int pos = 0;
        unordered_map<string, int> output = helper(expression, mp, pos);
        vector<pair<string, int>> ans(output.begin(), output.end());
        // sort result based on variable degree
        sort(ans.begin(), ans.end(), mycompare);
        vector<string> res;
        for (auto &p : ans) {
            // only consider non-zero coefficient variables
            if (p.second == 0) continue;
            res.push_back(to_string(p.second));
            if (p.first != "") res.back() += "*" + p.first;
        }
        return res;
    }

    unordered_map<string, int> helper(string &s, unordered_map<string, int> &mp, int &pos) {
        // every operand is an unordered_map, including single variable or nested (a * b + a * c);
        // if the operand is a number, use pair("", number)
        vector<unordered_map<string, int>> operands;
        vector<char> ops;
        ops.push_back('+');
        int n = s.size();
        while (pos < n && s[pos] != ')') {
            if (s[pos] == '(') {
                pos++;
                operands.push_back(helper(s, mp, pos));
            } else {
                int k = pos;
                while (pos < n && s[pos] != ' ' && s[pos] != ')') pos++;
                string t = s.substr(k, pos - k);
                bool isNum = true;
                for (char c : t) {
                    if (!isdigit(c)) isNum = false;
                }
                unordered_map<string, int> tmp;
                if (isNum)
                    tmp[""] = stoi(t);
                else if (mp.count(t))
                    tmp[""] = mp[t];
                else
                    tmp[t] = 1;
                operands.push_back(tmp);
            }
            if (pos < n && s[pos] == ' ') {
                ops.push_back(s[++pos]);
                pos += 2;
            }
        }
        pos++;
        return calculate(operands, ops);
    }

    unordered_map<string, int> calculate(vector<unordered_map<string, int>> &operands, vector<char> &ops) {
        unordered_map<string, int> ans;
        int n = ops.size();
        for (int i = n - 1; i >= 0; --i) {
            unordered_map<string, int> tmp = operands[i];
            while (i >= 0 && ops[i] == '*')
                tmp = multi(tmp, operands[--i]);
            int sign = ops[i] == '+' ? 1 : -1;
            for (auto &p : tmp) ans[p.first] += sign * p.second;
        }
        return ans;
    }

    unordered_map<string, int> multi(unordered_map<string, int> &lhs, unordered_map<string, int> &rhs) {
        unordered_map<string, int> ans;
        int m = lhs.size(), n = rhs.size();
        for (auto &p : lhs) {
            for (auto &q : rhs) {
                // combine and sort the product of variables
                string t = combine(p.first, q.first);
                ans[t] += p.second * q.second;
            }
        }
        return ans;
    }

    string combine(const string &a, const string &b) {
        if (a == "") return b;
        if (b == "") return a;
        vector<string> strs = split(a, '*');
        for (auto &s : split(b, '*')) strs.push_back(s);
        sort(strs.begin(), strs.end());
        string s;
        for (auto &t : strs) s += t + '*';
        s.pop_back();
        return s;
    }

    static vector<string> split(const string &s, char c) {
        vector<string> ans;
        int i = 0, n = s.size();
        while (i < n) {
            int j = i;
            i = s.find(c, i);
            if (i == -1) i = n;
            ans.push_back(s.substr(j, i - j));
            i++;
        }
        return ans;
    }

    static bool mycompare(pair<string, int> &a, pair<string, int> &b) {
        string s1 = a.first, s2 = b.first;
        vector<string> left = split(s1, '*');
        vector<string> right = split(s2, '*');
        return left.size() > right.size() || (left.size() == right.size() && left < right);
    }

    //629. K Inverse Pairs Array
    int kInversePairs(int n, int k) {
        static int mod = pow(10, 9) + 7;
        vector<vector<int>> dp(n + 1, vector<int>(k + 1));
        dp[0][0] = 1;
        for (int i = 1; i <= n; ++i) {
            dp[i][0] = 1;
            for (int j = 1; j <= k; ++j) {
                dp[i][j] = (dp[i][j - 1] + dp[i - 1][j]) % mod;
                if (j - i >= 0) {
                    dp[i][j] = (dp[i][j] - dp[i - 1][j - i] + mod) % mod;
                    //It must + mod, If you don't know why, you can check the case 1000, 1000
                }
            }
        }
        return dp[n][k];
    }

    //391. Perfect Rectangle
    bool isRectangleCover(vector<vector<int>> &rectangles) {
        int x1 = INT_MAX, y1 = INT_MAX, x2 = INT_MIN, y2 = INT_MIN;
        unordered_set<int> points;
        int area = 0;
        for (auto &rect : rectangles) {
            x1 = min(x1, rect[0]);
            y1 = min(y1, rect[1]);
            x2 = max(x2, rect[2]);
            y2 = max(y2, rect[3]);
            area += (rect[2] - rect[0]) * (rect[3] - rect[1]);
            int v;
            v = rect[0] * 101 + rect[1];
            if (points.count(v))
                points.erase(v);
            else
                points.insert(v);

            v = rect[0] * 101 + rect[3];
            if (points.count(v))
                points.erase(v);
            else
                points.insert(v);

            v = rect[2] * 101 + rect[1];
            if (points.count(v))
                points.erase(v);
            else
                points.insert(v);

            v = rect[2] * 101 + rect[3];
            if (points.count(v))
                points.erase(v);
            else
                points.insert(v);
        }
        if (!points.count(x1 * 101 + y1) ||
            !points.count(x1 * 101 + y2) ||
            !points.count(x2 * 101 + y1) ||
            !points.count(x2 * 101 + y2) ||
            points.size() != 4)
            return false;
        return area == (x2 - x1) * (y2 - y1);
    }
};


int main() {
    Solution sol;
    vector<vector<int>> rectangles;
    rectangles = {
            {1, 1, 3, 3},
            {3, 1, 4, 2},
            {3, 2, 4, 4},
            {1, 3, 2, 4},
            {2, 3, 3, 4}
    };
    cout << sol.isRectangleCover(rectangles) << endl;
    //vector<int> nums;
    //nums = { 2,1,1,2 };
    //nums = { 1,2,2,3,4 };
    //cout << sol.isSelfCrossing(nums) << endl;
    //nums = { 1,1,2,1,1 };
    //nums = { 1,1,1,1 };
    //vector<vector<int>>times{ {1,2,1},{2,3,2},{1,3,4} };
    //times = { { 2,1,1 },{ 2,3,1 },{ 3,4,1 } };
    //times = { {1,2,1},{2,1,3} };
    //cout << sol.networkDelayTime(times, 4, 2) << endl;
    //cout << sol.networkDelayTime(times, 2, 2) << endl;
    //vector<vector<int>> forest{
    //				{ 1,2,3 },
    //				{ 0,0,4 },
    //				{ 7,6,5 }
    //};
    //cout << sol.cutOffTree(forest) << endl;
    //for (int i = 1; i <= 13; i++)
    //	cout << sol.findKthNumber(13, i) << endl;
    //cout << sol.nearestPalindromic("1283") << endl;
    //vector<vector<int > >edges{ { 1,2 },{ 1,3 },{ 2,3 } };
    //edges = { { 1,2 },{ 2,3 },{ 3,4 },{ 4,1 },{ 1,5 } };
    //auto r = sol.findRedundantDirectedConnection(edges);
    //for (auto i : r)
    //	cout << i << " ";
    //cout << endl;
    //cout << sol.largestPalindrome(3) << endl;
    //for (int i = 1; i <= 8; i++)
    //	cout << sol.largestPalindrome(i) << endl;
    //cout << sol.getMaxRepetitions("acb", 4, "ab", 2) << endl;
    //vector<vector<int>> grid{ { 0, 1, -1 },
    //						  { 1, 0, -1 },
    //						  { 1, 1,  1 } };
    //grid = { {1,1,-1},{1,-1,1},{-1,1,1} };
    //cout << sol.cherryPickup_dfs(grid) << endl;
    //    cout << sol.makeLargestSpecial("11011000") << endl;
    //	vector<int>nums{ 1,2,3,4,5,6,7,8,9,10 };
    //	cout << sol.minmaxGasDist(nums, 9) << endl;
    //nums = { 23,24,36,39,46,56,57,65,84,98 };
    //cout << sol.minmaxGasDist(nums, 1) << endl;
    //nums = { 10, 19, 25, 27, 56, 63, 70, 87, 96, 97 };
    //cout << sol.minmaxGasDist(nums, 3) << endl;
    //    vector<vector<int>> board = {{4, 1, 2},
    //                                 {5, 0, 3}};
    //	board = { {1,2,3},{5,4,0} };
    //    cout << sol.slidingPuzzle(board) << endl;
    //    vector<int> nums{ 1,0,2 };
    //nums = { 0,1 };
    //cout << sol.isIdealPermutation(nums) << endl;
    //cout << sol.numJewelsInStones("aA","aAAvvbvvvv") << endl;
    //auto r = sol.combinationSum3(3, 9);
    //for (auto &i : r)
    //{
    //	for (auto j : i)
    //		cout << j << " ";
    //	cout << endl;
    //}
    //vector<int> houses = { 1,2,3,4 }, heaters = { 1,4 };
    //cout << sol.findRadius_binarysearch(houses, heaters) << endl;
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