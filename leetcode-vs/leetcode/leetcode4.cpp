//
// Created by ren on 18-1-23.
//
#include "stdafx.h"
//#include <sstream>
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


#define __DEBUG
using namespace std;

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
        } else if (i == N - 1 && j == N - 1) {
            if (f == 0)
                return dfs_741(i, j, 1, mat, memo);
        }
        int t = mat[i][j];
        mat[i][j] = 0;
        int F = f == 0 ? 1 : -1;
        vector<vector<int>> dirs{{1, 0},
                                 {0, 1}};
        for (auto d:dirs) {
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
            for (auto genic:l) {
                if (visited.count(genic)) {
                    continue;
                }
                if (r.count(genic)) {
                    return ans;
                }
                for (int i = 0; i < genic.size(); i++) {
                    char t = genic[i];
                    for (char c:chars) {
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
        for (auto &&item : dp) {
            cout << item << " ";
        }
        cout << endl;
        return dp[Len - 1];
    }
};


int main() {
    Solution sol;
    cout << sol.numDecodings("123") << endl;
    cout << sol.numDecodings("1*") << endl;
    cout << sol.numDecodings("*3") << endl;
    cout << sol.numDecodings("*10*1") << endl;
    cout << sol.numDecodings("*0**0") << endl;
    cout << sol.numDecodings("1*72*") << endl;
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