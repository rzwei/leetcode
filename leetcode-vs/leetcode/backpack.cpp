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

using namespace std;


struct Point {
    int x;
    int y;

    Point() : x(0), y(0) {}

    Point(int a, int b) : x(a), y(b) {}
};


class AllOne {
public:
    struct Row {
        list<string> strs;
        int val;

        Row(const string &s, int x) : strs({s}), val(x) {}
    };

    unordered_map<string, pair<list<Row>::iterator, list<string>::iterator>> strmap;
    list<Row> matrix;

    /** Initialize your data structure here. */
    AllOne() {

    }

    /** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
    void inc(string key) {
        if (strmap.find(key) == strmap.end()) {
            if (matrix.empty() || matrix.back().val != 1) {
                auto newrow = matrix.emplace(matrix.end(), key, 1);
                strmap[key] = make_pair(newrow, newrow->strs.begin());
            } else {
                auto newrow = --matrix.end();
                newrow->strs.push_front(key);
                strmap[key] = make_pair(newrow, newrow->strs.begin());
            }
        } else {
            auto row = strmap[key].first;
            auto col = strmap[key].second;
            auto lastrow = row;
            --lastrow;
            if (lastrow == matrix.end() || lastrow->val != row->val + 1) {
                auto newrow = matrix.emplace(row, key, row->val + 1);
                strmap[key] = make_pair(newrow, newrow->strs.begin());
            } else {
                auto newrow = lastrow;
                newrow->strs.push_front(key);
                strmap[key] = make_pair(newrow, newrow->strs.begin());
            }
            row->strs.erase(col);
            if (row->strs.empty()) matrix.erase(row);
        }
    }

    /** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
    void dec(string key) {
        if (strmap.find(key) == strmap.end()) {
            return;
        } else {
            auto row = strmap[key].first;
            auto col = strmap[key].second;
            if (row->val == 1) {
                row->strs.erase(col);
                if (row->strs.empty()) matrix.erase(row);
                strmap.erase(key);
                return;
            }
            auto nextrow = row;
            ++nextrow;
            if (nextrow == matrix.end() || nextrow->val != row->val - 1) {
                auto newrow = matrix.emplace(nextrow, key, row->val - 1);
                strmap[key] = make_pair(newrow, newrow->strs.begin());
            } else {
                auto newrow = nextrow;
                newrow->strs.push_front(key);
                strmap[key] = make_pair(newrow, newrow->strs.begin());
            }
            row->strs.erase(col);
            if (row->strs.empty()) matrix.erase(row);
        }
    }

    /** Returns one of the keys with maximal value. */
    string getMaxKey() {
        return matrix.empty() ? "" : matrix.front().strs.front();
    }

    /** Returns one of the keys with Minimal value. */
    string getMinKey() {
        return matrix.empty() ? "" : matrix.back().strs.front();
    }
};

class LFUCache {
    int cap;
    int size;
    int minFreq;
    unordered_map<int, pair<int, int>> m; //key to {value,freq};
    unordered_map<int, list<int>::iterator> mIter; //key to list iterator;
    unordered_map<int, list<int>> fm;  //freq to key list;
public:
    LFUCache(int capacity) {
        cap = capacity;
        size = 0;
    }

    int get(int key) {
        if (m.count(key) == 0) return -1;

        fm[m[key].second].erase(mIter[key]);
        m[key].second++;
        fm[m[key].second].push_back(key);
        mIter[key] = --fm[m[key].second].end();

        if (fm[minFreq].size() == 0)
            minFreq++;

        return m[key].first;
    }

    void put(int key, int value) {
        if (cap <= 0) return;

        int storedValue = get(key);
        if (storedValue != -1) {
            m[key].first = value;
            return;
        }

        if (size >= cap) {
            m.erase(fm[minFreq].front());
            mIter.erase(fm[minFreq].front());
            fm[minFreq].pop_front();
            size--;
        }

        m[key] = {value, 1};
        fm[1].push_back(key);
        mIter[key] = --fm[1].end();
        minFreq = 1;
        size++;
    }
};


class backpack {
public:
    int zeroOnePack(vector<int> worth, vector<int> volume, int Volume) {
        int length = worth.size();
        vector<int> dp(length + 1);
        for (int i = 1; i <= length; i++)
            for (int v = Volume; v >= volume[i]; v--)
                dp[v] = max(dp[v], dp[v - volume[i]] + worth[i]);
        return dp[length];
    }

    int zeroOnePack2(vector<int> worth, vector<int> volume, int Volume) {
        int length = worth.size();
        vector<int> dp(length + 1, 0x80000000);
        dp[0] = 0;
        for (int i = 1; i < length; i++)
            for (int v = Volume; i >= volume[i]; i--)
                dp[v] = max(dp[v], dp[v - volume[i]] + worth[i]);
        return dp[length];
    }

    int zeroOneCompetePack(vector<int> worth, vector<int> volume, int Volume) {
        int length = worth.size();
        vector<int> dp(length + 1);
        for (int i = 1; i < length; i++)
            for (int v = volume[i]; v < length; v++)
                dp[v] = max(dp[v], dp[v - volume[i]] + worth[i]);
        return dp[length];

    }
};

void mergeSort(int s, int e, vector<int> &nums) {
    if (s == e) {
        return;
    }
    if (e - s == 1) {
        if (nums[s] > nums[e]) {
            swap(nums[s], nums[e]);
        }
        return;
    }
    int m = (s + e) / 2;
    mergeSort(s, m, nums);
    mergeSort(m + 1, e, nums);
    vector<int> T(e - s + 1);
    int t = 0, i = s, j = m + 1;
    while (i <= m || j <= e) {
        if (i <= m && j <= e) {
            if (nums[i] < nums[j]) {
                T[t++] = nums[i++];
            } else {
                T[t++] = nums[j++];
            }
        } else if (i <= m) {
            T[t++] = nums[i++];
        } else {
            T[t++] = nums[j++];
        }
    }
    for (i = s; i <= e; i++) {
        nums[i] = T[i - s];

    }
}

void setOneElement(int s, int e, vector<int> &nums) {
    if (s >= e) {
        return;
    }
    int pv = nums[s];
    int i = s, j = e;
    while (i < j) {
        while (i < j && nums[j] >= pv) {
            j--;
        }
        nums[i] = nums[j];
        while (i < j && nums[i] <= pv) {
            i++;
        }
        nums[j] = nums[i];
    }
    nums[i] = pv;
    setOneElement(s, i - 1, nums);
    setOneElement(i + 1, e, nums);
}

void quickSort(vector<int> &nums) {
    setOneElement(0, nums.size() - 1, nums);
}


void heapAdjust(int i, vector<int> &nums, int Len) {
    if (i >= Len / 2) {
        return;
    }
    if (i * 2 + 1 < Len && nums[i] > nums[2 * i + 1]) {
        swap(nums[i], nums[i * 2]);
        heapAdjust(i * 2 + 1);
    }
    if (2 * i + 2 < Len && nums[i] > nums[2 * i + 2]) {
        swap(nums[i], nums[i * 2 + 2]);
        heapAdjust(i * 2 + 2);
    }
}

void heapify(vector<int> &nums, int Len) {
    for (int i = Len / 2; i >= 0; i--) {
        heapAdjust(i, nums);
    }
}

void heapSort(vector<int> &nums) {
    int Len = nums.size();
    for (int i = 0; i < Len; i++) {
        heapify(nums, Len - i);
        swap(nums[0], nums[Len - i - 1]);
    }
}

void heapAdjust(int i, vector<int> &nums, int Len) {
    if (i >= Len / 2) {
        return;
    }
    if (2 * i + 1 < Len && nums[i] < nums[2 * i + 1]) {
        swap(nums[i], nums[i * 2 + 1]);
        heapAdjust(i * 2 + 1, nums, Len);
    }
    if (2 * i + 2 < Len && nums[i] < nums[2 * i + 2]) {
        swap(nums[i], nums[i * 2 + 2]);
        heapAdjust(i * 2 + 2, nums, Len);
    }
}

void heapify(vector<int> &nums, int Len) {
    for (int i = Len / 2 - 1; i >= 0; i--) {
        heapAdjust(i, nums, Len);
    }
}

void heapSort(vector<int> &nums) {
    int Len = nums.size();
    for (int i = 0; i < Len; i++) {
        heapify(nums, Len - i);
        swap(nums[0], nums[Len - i - 1]);
    }
}

int main() {
    return 0;
}