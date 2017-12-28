//
// Created by rzhon on 17/12/28.
//
#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<double> medianSlidingWindow(vector<int> &nums, int k) {
        multiset<int> window(nums.begin(), nums.begin() + k);
        auto mid = next(window.begin(), k / 2);
        vector<double> medians  ;
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
};

int main() {
    cout << "hello world!" << endl;
    Solution sol;
    vector<int> nums{1, 3, -1, -3, 5, 3, 6, 7};
    auto r = sol.medianSlidingWindow(nums, 3);
    for (double i:r) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
