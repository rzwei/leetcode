//
// Created by rzhon on 17/12/28.
//
//#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <queue>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<double> medianSlidingWindow(vector<int> &nums, int k) {
        //480. Sliding Window Median
        multiset<int> window(nums.begin(), nums.begin() + k);
        auto mid = next(window.begin(), k / 2);
        vector<double> medians;
        for (int i = k;; i++) {
            medians.push_back((double(*mid) + *prev(mid, 1 - k % 2)) / 2);
            if (i == nums.size())
                return medians;
            window.insert(nums[i]);
            if (nums[i] < *mid)
                mid--;
            if (nums[i - k] <= *mid)
                mid++;
            window.erase(window.lower_bound(nums[i - k]));
        }
    }

    //514. Freedom Trail
    int findRotateSteps(string ring, string key) {
        int n = ring.length(), m = key.length();
        vector<vector<int>> dp(m + 1, vector<int>(n, 0));
        for (int i = m - 1; i >= 0; i--) {
            for (int j = 0; j < n; ++j) {
                dp[i][j] = INT32_MAX;
                for (int k = 0; k < n; ++k) {
                    if (ring[k] == key[i]) {
                        int diff = abs(j - k);
                        int step = min(diff, n - diff);
                        dp[i][j] = min(dp[i][j], step + dp[i + 1][k]);
                    }
                }
            }
        }
        return dp[0][0] + m;
    }

    int dfs_findRotateSteps(string &ring, string &key, int i, int j, map<pair<int, int>, int> &memo,
                            map<char, vector<int>> &idx) {
        if (memo.find(make_pair(i, j)) != memo.end()) {
            return memo[make_pair(i, j)];
        }
//        cout << i << " " << j << endl;
        if (i == key.length()) {
            return key.length();
        }
        int diff, step;
        int ret = INT32_MAX;
        for (int k:idx[key[i]]) {
            diff = abs(k - j);
            step = diff < ring.length() - diff ? diff : ring.length() - diff;
            ret = min(ret, dfs_findRotateSteps(ring, key, i + 1, k, memo, idx) + step);

        }
//        for (int k = 0; k < ring.length(); k++) {
//            if (ring[k] == key[i]) {
//                diff = abs(k - j);
//                step = diff < ring.length() - diff ? diff : ring.length() - diff;
//                ret = min(ret, dfs_findRotateSteps(ring, key, i + 1, k, memo, idx) + step);
//            }
//        }
        memo[make_pair(i, j)] = ret;
        return ret;
    }

    int findRotateSteps_dfs(string ring, string key) {
        map<pair<int, int>, int> memo;
        map<char, vector<int>> idx;
        for (int i = 0; i < ring.length(); ++i) {
            if (idx.find(ring[i]) == idx.end()) {
                idx[ring[i]] = vector<int>{i};
            } else {
                idx[ring[i]].push_back(i);
            }
        }
        return dfs_findRotateSteps(ring, key, 0, 0, memo, idx);
    }

    //493. Reverse Pairs
    int reversePairs(vector<int> &nums) {

    }
};

int main() {
    Solution sol;
    cout << sol.findRotateSteps_dfs("godding", "gd") << endl;
//    vector<int> nums{1, 3, -1, -3, 5, 3, 6, 7};
//    auto r = sol.medianSlidingWindow(nums, 3);
//    for (double i:r) {
//        cout << i << " ";
//    }
//    cout << endl;
    return 0;
}
