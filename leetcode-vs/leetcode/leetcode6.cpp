//
// Created by rzhon on 18/4/22.
//
#include "headers.h"
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

class Solution {
public:
    //821. Shortest Distance to a Character
    vector<int> shortestToChar(string S, char C) {
        vector<int> idx;
        int len = S.size();
        for (int i = 0; i < len; ++i) {
            if (S[i] == C) {
                idx.push_back(i);
            }
        }
        vector<int> ans(len, INT_MAX);
        int j = 0;
        for (int i = 0; i < len; ++i) {
            if (i > idx[j]) {
                if (j < idx.size() - 1)
                    j += 1;
            }
            ans[i] = abs(idx[j] - i);
        }
        j = idx.size() - 1;
        for (int i = len - 1; i >= 0; --i) {
            if (i < idx[j]) {
                if (j > 0)
                    j -= 1;
            }
            ans[i] = min(ans[i], abs(i - idx[j]));
        }
        return ans;
    }

    //822. Card Flipping Game
    int flipgame(vector<int> &fronts, vector<int> &backs) {
        int ans = INT_MAX, len = fronts.size();
        unordered_set<int> t, f;
        for (int i = 0; i < len; ++i) {
            if (fronts[i] == backs[i]) {
                f.insert(fronts[i]);
            } else {
                t.insert(fronts[i]);
                t.insert(backs[i]);
            }
        }
        for (auto n:t) {
            if (!f.count(n))
                ans = min(n, ans);
        }
        return ans == INT_MAX ? 0 : ans;
    }

    class Tire {
    public:
        Tire *next[26];

        Tire() : next{nullptr} {}
    };

    //820. Short Encoding of Words
    int minimumLengthEncoding(vector<string> &words) {
        auto cmp = [](string &a, string &b) {
            return a.length() > b.length();
        };
        sort(words.begin(), words.end(), cmp);
        int ans = 0;
        Tire *root = new Tire();
        for (auto &word:words) {
            auto cur = root;
            int f = 1;
            for (int i = word.size() - 1; i >= 0; --i) {
                if (!cur->next[word[i] - 'a']) {
                    cur->next[word[i] - 'a'] = new Tire();
                    if (f)
                        f = 0;
                }
                cur = cur->next[word[i] - 'a'];
            }
            if (!f) ans += word.size() + 1;
        }
        return ans;
    }

    //823. Binary Trees With Factors
    int numFactoredBinaryTrees(vector<int> &A) {
        int const mod = 1e9 + 7;
        sort(A.begin(), A.end());
        unordered_map<long long, int> idx;
        int len = A.size(), ans = 0;
        vector<int> dp(len);
        for (int i = 0; i < len; ++i) {
            idx[A[i]] = i;
            dp[i] = 1;
        }
        for (int i = 0; i < len; ++i) {
            for (int j = 0; j <= i; ++j) {
                long long v = (long long) (A[i]) * A[j];
                if (idx.count(v)) {
                    int n = (long long) (dp[i]) * dp[j] % mod;
                    if (i != j) {
                        n = (n + n) % mod;
                    }
                    int ii = idx[v];
                    dp[ii] = (dp[ii] + n) % mod;
                }
            }
        }
        for (auto &p : dp) {
            ans = (ans + p) % mod;
        }
        return ans;
    }
};

int main() {
    return 0;
}