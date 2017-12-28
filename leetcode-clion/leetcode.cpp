//
// Created by rzhon on 17/12/28.
//
//#include <bits/stdc++.h>
#include <iostream>
#include <vector>
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
};

int main() {
//    cout << "hello world!" << endl;
    Solution sol;
    cout << sol.findRotateSteps("godding", "gd") << endl;
//    vector<int> nums{1, 3, -1, -3, 5, 3, 6, 7};
//    auto r = sol.medianSlidingWindow(nums, 3);
//    for (double i:r) {
//        cout << i << " ";
//    }
//    cout << endl;
    return 0;
}
