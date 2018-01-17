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

class Solution {
public:
    bool isPrime(int a) {
        static set<int> prinmes{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
        int n = 0;
        while (a) {
            a &= (a - 1);
            n++;
        }
        return prinmes.find(n) != prinmes.end();
    }

    //762. Prime Number of Set Bits in Binary Representation
    int countPrimeSetBits(int L, int R) {
        int ans = 0;
        for (int i = L; i <= R; i++)
            ans += isPrime(i);
        return ans;
    }

    //763. Partition Labels
    vector<int> partitionLabels(string S) {
        map<char, vector<int>> idxes;
        int len = S.size();
        for (int i = 0; i < len; i++) {
            if (idxes.find(S[i]) == idxes.end())
                idxes[S[i]] = {i, i};
            else
                idxes[S[i]][1] = i;
        }

    }

    //764. Largest Plus Sign
    int orderOfLargestPlusSign(int n, vector<vector<int>> &mines) {
        vector<vector<int>> matrix(n, vector<int>(n, 1));

        for (auto &zero : mines)
            matrix[zero[0]][zero[1]] = 0;

        vector<vector<int>> lf(n, vector<int>(n, 0));
        auto rt = lf, up = lf, dn = lf;

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (matrix[i][j] == 1) {
                    lf[i][j] = j == 0 ? 1 : lf[i][j - 1] + 1;
                    up[i][j] = i == 0 ? 1 : up[i - 1][j] + 1;
                }
        int ans = 0;
        for (int i = n - 1; i >= 0; --i)
            for (int j = n - 1; j >= 0; --j)
                if (matrix[i][j] == 1) {
                    rt[i][j] = j == n - 1 ? 1 : rt[i][j + 1] + 1;
                    dn[i][j] = i == n - 1 ? 1 : dn[i + 1][j] + 1;

                    ans = max(ans, min(min(lf[i][j], rt[i][j]), min(up[i][j], dn[i][j])));
                }
        return ans;
    }

    //765. Couples Holding Hands
    int minSwapsCouples(vector<int> &row) {
        for (int &i : row)
            i /= 2;
        unsigned len = 0;
        for (auto it = row.begin(); it != row.end(); it += 2)
            if (*it != *(it + 1)) {
                auto toswap = find(it + 2, row.end(), *it);
                iter_swap(it + 1, toswap);
                ++len;
            }
        return len;
    }

    void visit_findItinerary(vector<string> &path, string cur, map<string, vector<string>> &m) {
        while (!m[cur].empty()) {
            string c = m[cur].back();
            m[cur].pop_back();
            visit_findItinerary(path, c, m);
        }
        path.push_back(cur);
    }

    //332. Reconstruct Itinerary
    vector<string> findItinerary(vector<pair<string, string>> tickets) {
        sort(tickets.begin(), tickets.end());
        map<string, vector<string>> m;
        for (auto &i : tickets)
            m[i.first].push_back(i.second);
        vector<string> path;
        visit_findItinerary(path, "JFK", m);
        reverse(path.begin(), path.end());
        return path;
    }

    bool onOneLine(Point &p0, Point &p1, Point &p2) {
        return (long long) (p2.y - p0.y) * (p1.x - p0.x) == (long long) (p1.y - p0.y) * (p2.x - p0.x);
    }

    //149. Max Points on a Line
    int maxPoints(vector<Point> &points) {
        int N = points.size();
        if (N <= 2)
            return N;
        if (N == 3) {
            int f = 1;
            for (int i = 0; i < N; i++) {
                if (points[i].x == 1 && points[i].y == 1)
                    continue;
                f = 0;
                break;
            }
            if (f == 1)
                return 3;
        }
        int ans = 0;
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++) {
                int t = 2;
                for (int k = 0; k < N; k++) {
                    if (k == i || k == j || (points[i].x == points[j].x && points[i].y == points[j].y))
                        continue;
                    if (onOneLine(points[i], points[j], points[k]))
                        t++;
                }
                ans = max(ans, t);
            }
        return ans;
    }

    //30. Substring with Concatenation of All Words
    vector<int> findSubstring_(string S, vector<string> &words) {
        //set<string> word_set(words.begin(), words.end()), visited;
        map<string, int> word_M;
        for (auto &i : words)
            word_M[i]++;
        int slen = S.length(), len = words[0].length();
        vector<int> ans;
        for (int start = 0; start < len; start++) {
            int s = start, e = start;
            auto word_set = word_M;
            while (e + len <= slen) {
                string t(S.substr(e, len));
                if (word_set.find(t) == word_set.end()) {
                    word_set = word_M;
                    e += len;
                    s = e;
                    continue;
                } else {
                    if (word_set[t] > 0)
                        word_set[t]--;
                    else {
                        while (word_set[t] <= 0) {
                            word_set[S.substr(s, len)]++;
                            s += len;
                        }
                        word_set[t]--;
                    }
                    int f = 1;
                    for (auto &i : word_set)
                        if (i.second > 0) {
                            f = 0;
                            break;
                        }
                    if (f) {
                        ans.push_back(s);
                    }
                }
                e += len;
            }
        }
        return ans;
    }

    vector<int> findSubstring__(string S, vector<string> &words) {
        map<string, int> word_M;
        int slen = S.length(), len = words[0].length(), cnt = words.size();
        for (auto &i : words)
            word_M[i]++;
        vector<int> ans;
        map<string, int> tdict;

        for (int start = 0; start < len; start++) {
            int s = start, e = start, count = 0;
            tdict.clear();
            while (e + len <= slen) {
                string t = S.substr(e, len);
                if (word_M.count(t)) {
                    tdict[t]++;
                    if (tdict[t] <= word_M[t])
                        count++;
                    else {
                        while (tdict[t] > word_M[t]) {
                            string tt(S.substr(s, len));
                            tdict[tt]--;
                            s += len;
                            if (tdict[tt] < word_M[tt])
                                count--;
                        }
                    }
                    if (count == cnt) {
                        ans.push_back(s);
                        tdict[S.substr(s, len)]--;
                        s += len;
                        count--;
                    }
                } else {
                    count = 0;
                    tdict.clear();
                    s = e + len;
                }
                e += len;
            }
        }
        return ans;
    }

    vector<int> findSubstring(string s, vector<string> &words) {
        vector<int> ans;
        int n = s.size(), cnt = words.size();
        if (n <= 0 || cnt <= 0) return ans;

        // init word occurence
        unordered_map<string, int> dict;
        for (int i = 0; i < cnt; ++i) dict[words[i]]++;

        // travel all sub string combinations
        int wl = words[0].size();
        for (int i = 0; i < wl; ++i) {
            int left = i, count = 0;
            unordered_map<string, int> tdict;
            for (int j = i; j <= n - wl; j += wl) {
                string str = s.substr(j, wl);
                // a valid word, accumulate results
                if (dict.count(str)) {
                    tdict[str]++;
                    if (tdict[str] <= dict[str])
                        count++;
                    else {
                        // a more word, advance the window left side possiablly
                        while (tdict[str] > dict[str]) {
                            string str1 = s.substr(left, wl);
                            tdict[str1]--;
                            if (tdict[str1] < dict[str1]) count--;
                            left += wl;
                        }
                    }
                    // come to a result
                    if (count == cnt) {
                        ans.push_back(left);
                        // advance one word
                        tdict[s.substr(left, wl)]--;
                        count--;
                        left += wl;
                    }
                }
                    // not a valid word, reset all vars
                else {
                    tdict.clear();
                    count = 0;
                    left = j + wl;
                }
            }
        }
        return ans;
    }

    //556. Next Greater Element III
    int nextGreaterElement(int n) {
        vector<int> s;
        while (n) {
            s.push_back(n % 10);
            n /= 10;
        }
        int len = s.size(), i = 0, start = 0;
        while (i + 1 < len) {
            if (s[i + 1] >= s[i]) {
                i++;
            } else {
                break;
            }
        }
        if (i == len - 1) {
            return -1;
        }
        int next = s[i + 1], t = i;
        for (int j = 0; j < i + 1; ++j) {
            if (s[j] > next && s[j] < s[t]) {
                t = j;
            }
        }
        swap(s[t], s[i + 1]);
        sort(s.begin(), s.begin() + i + 1, [](int x, int y) { return x > y; });
        long long T = 0;
        for (i = s.size() - 1; i >= 0; i--) {
            T = T * 10 + s[i];
            if (T > INT32_MAX) {
                return -1;
            }
        }
        return int(T);
    }
};

int main() {
    Solution sol;
//    vector<int> nums{1, 2, 4, 5, 2, 3, 4, 5, 2};
//    sort(nums.begin(), nums.end(), [](int x, int y) { return x < y; });
//    for (auto i:nums) {
//        cout << i << " ";
//    }
//    cout << endl;

//    int n = 11;
//    int n = 1999999999;
//    cout << n << endl;
//    while ((n = sol.nextGreaterElement(n)) != -1) {
//        cout << n << endl;
//    }
//    vector<string> words{"foo", "bar", "the"};
//    vector<string> words{"a", "b", "a"};
    //vector<string> words{ "man" };
//    auto r = sol.findSubstring__("barfoofoobarthefoobarman", words);
//    auto r = sol.findSubstring__("abababab", words);
//    for (auto i : r)
//        cout << i << " ";
//    cout << endl;
    //vector<Point> points;
    ////vector<vector<int>> ps{ { 84,250 },{ 0,0 },{ 1,0 },{ 0,-70 },{ 0,-70 },{ 1,-1 },{ 21,10 },{ 42,90 },{ -42,-230 } };
    //vector<vector<int>> ps{ { 0, 0 },{ 1,1 },{ 1,-1 } };
    //for (auto i : ps)
    //	points.push_back(Point(i[0], i[1]));
    //cout << sol.maxPoints(points) << endl;
    //vector<pair<string, string>> ticks{ { "MUC", "LHR" },{ "JFK", "MUC" },{ "SFO", "SJC" },{ "LHR", "SFO" } };
    //auto r = sol.findItinerary(ticks);
    //for (auto &i : r)
    //	cout << i << " ";
    //cout << endl;
    return 0;
}