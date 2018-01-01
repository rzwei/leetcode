//
// Created by rzhon on 17/12/28.
//
//#include <bits/stdc++.h>
//#ifdef WINVER
//#include "stdafx.h"
//#endif
#include <cmath>
#include <functional>
#include <iostream>
#include <list>
#include <stack>
#include <vector>
#include <map>
#include <unordered_map>
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
        for (int k : idx[key[i]]) {
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
        for (int i : nums) {
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
        for (auto n : nums) {
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


    //546. Remove Boxes
    int removeBoxes(vector<int> &boxes) {
        int len = boxes.size();
        vector<vector<vector<int>>> dp(len, vector<vector<int>>(len, vector<int>(len)));
        return subproblem_removeBoxes(dp, 0, len - 1, 0, boxes);
    }

    int subproblem_removeBoxes(vector<vector<vector<int>>> &dp, int i, int j, int k, vector<int> &boxes) {
        if (i > j)
            return 0;
        int &res = dp[i][j][k];
        if (res > 0)
            return res;
        for (; i + 1 <= j && boxes[i + 1] == boxes[i]; i++, k++);
        res = (k + 1) * (k + 1) + subproblem_removeBoxes(dp, i + 1, j, 0, boxes);
        for (int m = i + 1; m <= j; m++)
            if (boxes[m] == boxes[i])
                res = max(res, subproblem_removeBoxes(dp, i + 1, m - 1, 0, boxes) +
                               subproblem_removeBoxes(dp, m, j, k + 1, boxes));
        return res;
    }


    bool drop_pourwater(vector<int> &height, int cur, int f, int sheight) {
        if (cur < 0 || cur >= height.size() || height[cur] > sheight) {
            return false;
        }
        int next = cur + f;
        if (drop_pourwater(height, next, f, height[cur])) {
            return true;
        }
        if (height[cur] < sheight) {
            height[cur]++;
            return true;
        }
        return false;
    }

    //755. Reach a Number
    int reachNumber(int target) {
        target = abs(target);
        long long n = ceil((-1.0 + sqrt(1 + 8.0 * target)) / 2);
        long long sum = n * (n + 1) / 2;
        if (sum == target) return n;
        long long res = sum - target;
        if ((res & 1) == 0)
            return n;
        else
            return n + ((n & 1) ? 2 : 1);
    }

    //756. Pour Water
    vector<int> pourWater(vector<int> &heights, int V, int K) {
        while (V--) {
            if (!drop_pourwater(heights, K - 1, -1, heights[K]) && !drop_pourwater(heights, K + 1, 1, heights[K])) {
                heights[K]++;
            }
        }
        return heights;
    }


    //757. Pyramid Transition Matrix not ac tle
    bool dfs_pyramidTransition(string &prev, int previ, string &bottom, set<string> &allowed,
                               set<pair<string, string>> &memo) {
        //        cout << prev << "," << previ << "," << bottom << endl;
        if (prev.length() == 1) {
            return true;
        }
        if (memo.find(make_pair(prev, bottom)) != memo.end()) {
            return false;
        }
        if (bottom.length() == prev.length() - 1) {
            string n;
            return dfs_pyramidTransition(bottom, 0, n, allowed, memo);
        }
        string tri, tri2;
        for (int i = 0; i < 7; ++i) {
            tri.clear();
            tri += prev[previ];
            tri += prev[previ + 1];
            tri += char('A' + i);

            bottom += char('A' + i);
            if (allowed.find(tri) != allowed.end()) {
                int f = 1;
                if (bottom.length() >= 2) {
                    tri2.clear();
                    tri2 += bottom[bottom.length() - 2];
                    tri2 += bottom[bottom.length() - 1];
                    if (!judge_pyramidTransition(tri2, allowed)) {
                        f = 0;
                    }
                }
                if (f && dfs_pyramidTransition(prev, previ + 1, bottom, allowed, memo))
                    return true;
            }
            bottom.pop_back();
        }
        memo.insert(make_pair(prev, bottom));
        return false;
    }

    bool judge_pyramidTransition(string prefix, set<string> &allowed) {
        for (int i = 0; i < 7; ++i) {
            prefix += char('A' + i);
            if (allowed.find(prefix) != allowed.end()) {
                return true;
            }
            prefix.pop_back();
        }
        return false;
    }

    bool pyramidTransition(string bottom, vector<string> &allowed) {
        set<string> allowed_set(allowed.begin(), allowed.end());
        set<pair<string, string>> memo;
        string new_bottom;
        return dfs_pyramidTransition(bottom, 0, new_bottom, allowed_set, memo);
    }

    //759. Set Intersection Size At Least Two
    int intersectionSizeTwo(vector<vector<int>> &intervals) {
        sort(intervals.begin(), intervals.end(), [](vector<int> &a, vector<int> &b) {
            if (a[0] == b[0]) {
                return a[1] > b[1];
            }
            return a[0] < b[0];
        });
//        for (auto &i : intervals) {
//            cout << i[0] << " " << i[1] << ",";
//        }
//        cout << endl;

        stack<vector<int>> st;

        for (auto &i : intervals) {
            while (!st.empty() && st.top()[1] >= i[1]) {
                st.pop();
            }
            st.push(i);
        }
        int n = st.size();

        vector<vector<int>> a(n, vector<int>(2));
        for (int i = n - 1; i >= 0; --i) {
            a[i][0] = st.top()[0];
            a[i][1] = st.top()[1];
            st.pop();
        }

        int ans = 2;

        int p1 = a[0][1] - 1, p2 = a[0][1];

        for (int i = 1; i < n; ++i) {
            bool bo1 = (p1 >= a[i][0] && p1 <= a[i][1]), bo2 = (p2 >= a[i][0] && p2 <= a[i][1]);
            if (bo1 && bo2)
                continue;
            if (bo2) {
                p1 = p2;
                p2 = a[i][1];
                ans++;
                continue;
            }
            p1 = a[i][1] - 1;
            p2 = a[i][1];
            ans += 2;
        }

        return ans;
    }

//    int remove(vector<int> &boxes, int i, int j, int k, vector<vector<vector<int>>> &dp) {
//        if (i > j) {
//            return 0;
//        }
//        if (dp[i][j][k] > 0)
//            return dp[i][j][k];
//        for (; i + 1 <= j && boxes[i + 1] == boxes[i]; i++, k++);
//        int res = (k + 1) * (k + 1) + remove(boxes, i + 1, j, 0, dp);
//
//        for (int m = i + 1; m <= j; ++m) {
//            if (boxes[m] == boxes[i]) {
//                res = max(res, remove(boxes, i + 1, m - 1, 0, dp) + remove(boxes, m, j, k + 1, dp));
//            }
//        }
//        dp[i][j][k] = res;
//        return res;
//    }
//
//    int removeBoxes2(vector<int> &boxes) {
//        int len = boxes.size();
//        vector<vector<vector<int>>> dp(len, vector<vector<int>>(len, vector<int>(len)));
//        return remove(boxes, 0, len - 1, 0, dp);
//    }

    int subproblem_printer(string &s, int i, int j, int k, map<pair<int, pair<int, int>>, int> &dp) {
        if (i > j) {
            return 0;
        }
        auto key = make_pair(i, make_pair(j, k));
        if (dp.find(key) != dp.end()) {
            return dp[key];
        }
        dp[key] = 0;
        int &res = dp[key];
        if (res > 0) {
            return res;
        }
        for (; i + 1 <= j && s[i] == s[i + 1]; i++, k++);
        res = 1 + subproblem_printer(s, i + 1, j, 0, dp);

        for (int m = i + 1; m <= j; ++m) {
            if (s[m] == s[i]) {
                res = min(res, subproblem_printer(s, i + 1, m - 1, 0, dp) + subproblem_printer(s, m, j, k + 1, dp));
            }
        }
        return res;
    }

    int subproblem_printer2(string &s, int i, int j, map<pair<int, int>, int> &dp) {
        if (i > j) {
            return 0;
        }
        auto key = make_pair(i, j);
        if (dp.find(key) != dp.end()) {
            return dp[key];
        }

        dp[key] = 0;
        int &res = dp[key];
        res = subproblem_printer2(s, i + 1, j, dp) + 1;
        for (int k = i + 1; k <= j; ++k) {
            if (s[k] == s[i]) {
                res = min(res, subproblem_printer2(s, i, k - 1, dp) + subproblem_printer2(s, k + 1, j, dp));
            }
        }
        return res;
    }

    //664. Strange Printer
    int strangePrinter(string s) {
        int n = s.length();
//        map<pair<int, pair<int, int>>, int> dp;
        map<pair<int, int>, int> dp;
//        return subproblem_printer(s, 0, s.length() - 1, 0, dp);
        return subproblem_printer2(s, 0, s.length() - 1, dp);
    }

    int strangePrinter_(string s) {
        int n = s.size();
        if (n == 0)return 0;
        vector<vector<int> > dp(n, vector<int>(n, 0));
        dp[0][0] = 1;
        for (int k = 1; k <= n; k++) {
            for (int i = 0; i < n; i++) {
                int last = i + k - 1;
                if (last >= n) continue;
                if (last > 0) dp[i][last] = dp[i][last - 1] + 1;
                for (int j = i; j < last; j++) {
                    if (s[j] == s[last]) dp[i][last] = min(dp[i][last], dp[i][j] + dp[j + 1][last - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }

    //672. Bulb Switcher II
    int flipLights(int n, int m) {
        if (m == 0) return 1;
        if (n == 1) return 2;
        if (n == 2 && m == 1) return 3;
        if (n == 2) return 4;
        if (m == 1) return 4;
        if (m == 2) return 7;
        if (m >= 3) return 8;
        return 8;
    }

    //659. Split Array into Consecutive Subsequences
    bool isPossible(vector<int> &nums) {
        unordered_map<int, int> dict, temp;
        for (auto &ele: nums) dict[ele]++;
        for (auto &ele: nums) {
            if (dict[ele] == 0)   //if the ele is already used in some sequence
                continue;
            else if (temp[ele] > 0) {  //if the ele can be added in the last consecutive sequence
                dict[ele]--;
                temp[ele]--;
                temp[ele + 1]++;

            } else if (dict[ele + 1] > 0 && dict[ele + 2] > 0) {
                //this ele should form a consecutive sequence by itself since it cannot be appended to a previous sequence
                dict[ele]--;
                dict[ele + 1]--;
                dict[ele + 2]--;
                temp[ele + 3]++;
            } else //doesn't belong to any consecutive sequence
                return false;
        }
        return true;
    }
};

int main() {
    Solution sol;
    vector<int> nums{1, 2, 3, 3};

    cout << sol.isPossible(nums) << endl;
//    cout << sol.strangePrinter("aaabbb") << endl;
//    cout << sol.strangePrinter("abcba") << endl;
//    cout << sol.strangePrinter(
//            "dvdamcpqesjzyzhgfpkgodvctchzukuvqrrpectqmnqhunnkuwoyomhxtyylmuprgrjfiprjqrizrjgnvnwfjztshqairnierpvw")
//         << endl;
    //vector<int> nums{ 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 4, 5, 6 };
    //sort(nums.begin(), nums.end(), [](const int &a, const int &b) {return a < b; });
    //for (int i = 0; i < nums.size(); ++i) {
    //	cout << nums[i] << " ";
    //}
    //cout << endl;
//    vector<vector<int>> intervals{{1, 3},
//                                  {1, 4},
//                                  {2, 5},
//                                  {3, 5}};

//    for (auto &i : intervals)
//        cout << i[0] << " " << i[1] << ",";
//    cout << endl;
//    cout << sol.intersectionSizeTwo(intervals) << endl;
    //    vector<string> allowed = {"XYD", "YZE", "DEA", "FFF"};
    //    vector<string> allowed = {"XXX", "XXY", "XYX", "XYY", "YXZ"};
    //    vector<string> allowed = {"ACC", "ACB", "ACA", "AAC", "ACD", "BCD", "BCC", "BAB", "CAC", "CCD", "CCA", "CCB", "DAD",
    //                              "DAC", "CAD", "ABB", "ABC", "ABD", "BDB", "BBD", "BBC", "BBB", "ADD", "ADB", "ADA", "CDB",
    //                              "DDA", "CDD", "CBC", "CDA", "CBD", "DBD"};
    //    cout << sol.pyramidTransition("XYZ", allowed) << endl;
    //    cout << sol.pyramidTransition("DBDBCBCDAB", allowed) << endl;
    //    vector<int> height{1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1};
    //    vector<int> height{1, 2, 3, 4};
    //    auto r = sol.pourWater(height, 10, 2);
    //    for (int i : r) {
    //        cout << i << " ";
    //    }
    //    cout << endl;
//    vector<int> boxes{1, 3, 2, 2, 2, 3, 4, 3, 1};
//    vector<int> boxes{3, 8, 8, 5, 5, 3, 9, 2, 4, 4, 6, 5, 8, 4, 8, 6, 9, 6, 2, 8, 6, 4, 1, 9, 5, 3, 10, 5, 3, 3, 9, 8,
//                      8, 6, 5, 3, 7, 4, 9, 6, 3, 9, 4, 3, 5, 10, 7, 6, 10, 7};
//    cout << sol.removeBoxes(boxes) << endl;
    //vector<int> nums{ 1, 2, 3, 4, 5 };
    //    vector<int> nums{0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 6, 7, 9, 9, 10, 10, 11, 11, 12, 13, 14, 14, 15, 17, 19, 19, 22, 24,
    //                     24, 25, 25, 27, 27, 29, 30, 32, 32, 33, 33, 35, 36, 38, 39, 41, 42, 43, 44, 44, 46, 47, 48, 49, 52,
    //                     53, 53, 54, 54, 57, 57, 58, 59, 59, 59, 60, 60, 60, 61, 61, 62, 64, 66, 68, 68, 70, 72, 72, 74, 74,
    //                     74, 75, 76, 76, 77, 77, 80, 80, 82, 83, 85, 86, 87, 87, 92, 93, 94, 96, 96, 97, 98, 99};
    //auto r = sol.findClosestElements(nums, 4, 3);
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

}
