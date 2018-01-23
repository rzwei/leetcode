//
// Created by ren on 18-1-23.
//
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
#include <stack>
#include <list>
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

    int dfs_741(int i, int j, int f, vector<vector<vector<int>>> &M,
                vector<vector<vector<int>>> &memo) {
//        cout << i << " " << j << " " << f << " " << endl;
        int N = M[0].size(), res = -1;
        vector<vector<int>> &mat = M[f];
        if (mat[i][j] == -1) {
            return -1;
        }
        if (memo[f][i][j] != -1) {
            return memo[f][i][j];
        }
        if (i == N - 1 && j == N - 1) {
            if (f == 0)
                return dfs_741(0, 0, 1, M, memo);
            else {
                return mat[i][j];
            }
        }
        int t = mat[i][j];
        M[f][i][j] = 0;
        M[!f][N - 1 - i][N - 1 - j] = 0;

        vector<vector<int>> dirs{{1, 0},
                                 {0, 1}};
        for (auto d:dirs) {
            int nx = i + d[0], ny = j + d[1];
            if (0 <= nx && nx < N && 0 <= ny && ny < N && mat[i][j] != -1) {
                int r = dfs_741(nx, ny, f, M, memo);
                if (r >= 0) {
                    res = max(res, r + t);
                }
            }
        }
        M[f][i][j] = t;
        M[!f][N - 1 - i][N - 1 - j] = t;
        memo[f][i][j] = res;
        return res;
    }

    //741. Cherry Pickup
    int cherryPickup(vector<vector<int>> &grid) {
        int N = grid.size();
        vector<vector<vector<int>>> M(2, vector<vector<int>>(N, vector<int>(N)));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                M[0][i][j] = grid[i][j];
                M[1][N - 1 - i][N - 1 - j] = M[0][i][j];
            }
        }
        vector<vector<vector<int>>> memo(2, vector<vector<int>>(N, vector<int>(N, -1)));
        int r = dfs_741(0, 0, 0, M, memo);
        return r == -1 ? 0 : r;
    }
};

int main() {
    Solution sol;
    vector<vector<int>> mat{{0, 1, -1},
                            {1, 0, -1},
                            {1, 1, 1}};
    cout << sol.cherryPickup(mat) << endl;
//    string expression;
//    while (getline(cin, expression)) {
//        cout << sol.calculateII(expression) << endl;
//    }
    return 0;
}