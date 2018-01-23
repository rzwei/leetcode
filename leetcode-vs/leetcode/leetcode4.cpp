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

    //772. Basic Calculator III not ac
    int calculate(string expression) {
    }
};

int main() {
    Solution sol;
    string expression;
    while (getline(cin,expression)) {
        cout << sol.calculateII(expression) << endl;
    }
    return 0;
}