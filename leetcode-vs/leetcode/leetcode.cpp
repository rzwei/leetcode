//
// Created by rzhon on 17/12/28.
//
//#include <bits/stdc++.h>
#ifdef WINVER
#include "stdafx.h"
#endif

#include <iostream>
#include <list>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <queue>
#include <algorithm>

using namespace std;

class BSTNode {
public:
    int val, cnt;
    BSTNode *left, *right;

    BSTNode(int _val) {
        val = _val;
        cnt = 1;
        left = nullptr;
        right = nullptr;
    }

    static int search(BSTNode *root, long val) {
        if (root == nullptr) {
            return 0;
        }
        if (val == root->val) {
            return root->cnt;
        } else if (val < root->val) {
            return root->cnt + search(root->left, val);
        } else {
            return search(root->right, val);
        }
    }

    static BSTNode *insert(BSTNode *root, int val) {
        if (root == nullptr) {
            return new BSTNode(val);
        } else if (val == root->val) {
            root->cnt++;
        } else if (val < root->val) {
            root->left = insert(root->left, val);
        } else {
            root->cnt++;
            root->right = insert(root->right, val);
        }
        return root;
    }
};

class BIT {
public:
    static int search(vector<int> &bit, int i) {
        int s = 0;
        while (i < bit.size()) {
            s += bit[i];
            i += i & -i;
        }
        return s;
    }

    static void insert(vector<int> &bit, int i) {
        while (i > 0) {
            bit[i] += 1;
            i -= i & -i;
        }
    }

    static int index(vector<int> &arr, long val) {
        int l = 0, r = arr.size() - 1, m = 0;
        while (l <= r) {
            m = (l + r) / 2;
            if (arr[m] >= val) {
                r = m - 1;
            } else {
                l = m + 1;
            }
        }
        return l + 1;
    }
};

int binary_lower(vector<int> &nums, int target) {
    int i = 0, j = nums.size(), m;
    while (i < j) {
        m = (i + j) / 2;
        if (nums[m] < target) {
            i = m + 1;
        } else {
            j = m;
        }
    }
    return i;
}

int binary_upper(vector<int> &nums, int target) {
    int i = 0, j = nums.size(), m;
    while (i < j) {
        m = (i + j) / 2;
        if (target < nums[m]) {
            j = m;
        } else {
            i = m + 1;
        }
    }
    return i;
}

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
        vector<int> cache(nums.size(), 0);
        return merge_reversePairs(nums, 0, nums.size() - 1, cache);
    }

    int merge_reversePairs(vector<int> &nums, int s, int e, vector<int> &merge) {
        if (s >= e)
            return 0;
        int mid = (s + e) / 2;
        int count = merge_reversePairs(nums, s, mid, merge) + merge_reversePairs(nums, mid + 1, e, merge);
        int i = s, j = mid + 1, p = mid + 1, k = 0;
        while (i <= mid) {
            while (p <= e && nums[i] > 2L * nums[p])p++;
            count += p - mid - 1;
            while (j <= e && nums[i] >= nums[j])
                merge[k++] = nums[j++];
            merge[k++] = nums[i++];
        }
        while (j <= e) merge[k++] = nums[j++];
        for (i = s; i <= e; i++)
            nums[i] = merge[i - s];
        return count;
    }

//    //493. Reverse Pairs
//    int reversePairs(vector<int> &nums) {
//        vector<int> cache(nums.size(), 0);
//        return merge_reversePairs_iterator(nums.begin(), nums.end());
//    }

    int merge_reversePairs_iterator(vector<int>::iterator begin, vector<int>::iterator end) {
        if (end - begin <= 1)
            return 0;
        auto mid = begin + (end - begin) / 2;
        int count = merge_reversePairs_iterator(begin, mid) + merge_reversePairs_iterator(mid, end);
        auto i = begin, p = mid;
        while (i < mid) {
            while (p < end && *i > 2L * *p) p++;
            count += p - mid;
            i++;
        }
        inplace_merge(begin, mid, end);
        return count;
    }

    //493. Reverse Pairs
    int reversePairs_bst(vector<int> &nums) {
//        vector<int> cache(nums.size(), 0);
//        return merge_reversePairs_iterator(nums.begin(), nums.end());
        BSTNode *root = nullptr;
        int ans = 0;
        for (int i:nums) {
            ans += BSTNode::search(root, 2L * i + 1);
            root = BSTNode::insert(root, i);
        }
        return ans;
    }

    int reversePairs_bit(vector<int> &nums) {
        vector<int> copy(nums.begin(), nums.end());
        sort(copy.begin(), copy.end());
        vector<int> bit(nums.size() + 1, 0);
        int ans = 0;
        for (auto n:nums) {
            ans += BIT::search(bit, BIT::index(copy, 2L * n + 1));
            BIT::insert(bit, BIT::index(copy, n));
        }
        return ans;
    }

    //658. Find K Closest Elements
    vector<int> findClosestElements(vector<int> &arr, int k, int x) {
//        int max_diff = 0;
//
//        max_diff = max(max(max_diff, abs(x - arr[0])), abs(x - arr[arr.size() - 1]));
//
//        int l = 0, r = max_diff;
//        while (l < r) {
//            int m = (l + r) / 2;
//            int n = upper_bound(arr.begin(), arr.end(), x + m) - lower_bound(arr.begin(), arr.end(), x - m);
//            if (n >= k) {
//                r = m;
//            } else {
//                l = m + 1;
//            }
//        }
//        cout << r << endl;
//        int lidx = lower_bound(arr.begin(), arr.end(), x - r) - arr.begin(), uidx =
//                upper_bound(arr.begin(), arr.end(), x + r) - arr.begin();
//        uidx -= 1;
//        while (uidx - lidx + 1 > k) {
//            if (abs(arr[uidx] - x) >= abs(arr[lidx] - x)) {
//                uidx--;
//            } else {
//                lidx++;
//            }
//        }
//
//        vector<int> ret(arr.begin() + lidx, arr.begin() + uidx + 1);
//        return ret;
        int lo = 0, hi = arr.size() - k, m;
        while (lo < hi) {
            m = (lo + hi) / 2;
            if (x - arr[m] > arr[m + k] - x) {
                lo = m + 1;
            } else {
                hi = m;
            }
        }
        return vector<int>(arr.begin() + lo, arr.begin() + lo + k);
    }

    //667. Beautiful Arrangement II
    vector<int> constructArray(int n, int k) {
        vector<int> res;
        for (int i = 1, j = n; i <= j;) {
            if (k > 1) {
                res.push_back(k-- % 2 ? i++ : j--);
            } else {
                res.push_back(i++);
            }
        }
        return res;
    }
};

int main() {
    Solution sol;
    vector<int> nums{1, 2, 3, 4, 5};
//    vector<int> nums{0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 6, 7, 9, 9, 10, 10, 11, 11, 12, 13, 14, 14, 15, 17, 19, 19, 22, 24,
//                     24, 25, 25, 27, 27, 29, 30, 32, 32, 33, 33, 35, 36, 38, 39, 41, 42, 43, 44, 44, 46, 47, 48, 49, 52,
//                     53, 53, 54, 54, 57, 57, 58, 59, 59, 59, 60, 60, 60, 61, 61, 62, 64, 66, 68, 68, 70, 72, 72, 74, 74,
//                     74, 75, 76, 76, 77, 77, 80, 80, 82, 83, 85, 86, 87, 87, 92, 93, 94, 96, 96, 97, 98, 99};
    auto r = sol.findClosestElements(nums, 4, 3);
//    auto r = sol.findClosestElements(nums, 25, 90);
//    for (auto i:r) {
//        cout << i << " ";
//    }
//    cout << endl;
//    vector<int> nums{1, 3, 2, 3, 1};
//    vector<int> nums{2, 4, 3, 5, 1};
//    vector<int> nums{1, 1, 1, 1, 1, 1, 1};
//    vector<int> nums{2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647};
//    cout << sol.reversePairs_bit(nums) << endl;
//    cout << sol.findRotateSteps_dfs("godding", "gd") << endl;
//    vector<int> nums{1, 3, -1, -3, 5, 3, 6, 7};
//    auto r = sol.medianSlidingWindow(nums, 3);
//    for (double i:r) {
//        cout << i << " ";
//    }
//    cout << endl;
    return 0;
}
