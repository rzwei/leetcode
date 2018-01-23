//
// Created by ren on 18-1-23.
//
//#include "stdafx.h"
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

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

private:
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
};


int main() {
    Solution sol;
    vector<vector<int>> mat{{0, 1, -1},
                            {1, 0, 1},
                            {1, 1, 1}};
    cout << sol.cherryPickup(mat) << endl;
//    string expression;
//    while (getline(cin, expression)) {
//        cout << sol.calculateII(expression) << endl;
//    }
    return 0;
}