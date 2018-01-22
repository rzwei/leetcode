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

#define __DEBUG
using namespace std;


struct Point {
	int x;
	int y;

	Point() : x(0), y(0) {}

	Point(int a, int b) : x(a), y(b) {}
};

//715. Range Module
class RangeModule {
public:
	//    vector<pair<int, int>> intervels;
	//    map<int, int> intervels;
	set<pair<int, int>> intervels;

	RangeModule() {

	}

	void addRange(int left, int right) {
		auto it = intervels.lower_bound({ left, right });
		if (it != intervels.end() && it->second < left) {
			it++;
		}
		while (it != intervels.end() && right > it->first) {
			left = min(left, it->first);
			right = max(right, it->second);
			it = intervels.erase(it);
		}
		intervels.insert({ left, right });
	}

	bool queryRange(int left, int right) {
		auto it = intervels.lower_bound({ left, right });
		return it != intervels.end() && left >= it->first && right < it->second;
	}

	void removeRange(int left, int right) {
		auto it = intervels.lower_bound({ left, right });
		if (it != intervels.end() && it->second <= left)
			it++;
		while (it != intervels.end() && it->first < right)
		{
			if (it->first >= left && it->second <= right)
			{
				it = intervels.erase(it);
			}
			else {
				int nl = max(it->first, left), nr = min(it->second, right);
				if (nl < nr)
				{
					auto NxIt = ++it;
					if (nl < it->second)
					{
						intervels.insert({ it->first,nl });
					}
					if (nr < it->second)
					{
						intervels.insert({ nr,it->second });
					}
					it = NxIt;
				}
				else {
					
				}
			}
		}
	}
};

class AllOne {
public:
	struct Row {
		list<string> strs;
		int val;

		Row(const string &s, int x) : strs({ s }), val(x) {}
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
			}
			else {
				auto newrow = --matrix.end();
				newrow->strs.push_front(key);
				strmap[key] = make_pair(newrow, newrow->strs.begin());
			}
		}
		else {
			auto row = strmap[key].first;
			auto col = strmap[key].second;
			auto lastrow = row;
			--lastrow;
			if (lastrow == matrix.end() || lastrow->val != row->val + 1) {
				auto newrow = matrix.emplace(row, key, row->val + 1);
				strmap[key] = make_pair(newrow, newrow->strs.begin());
			}
			else {
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
		}
		else {
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
			}
			else {
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

		m[key] = { value, 1 };
		fm[1].push_back(key);
		mIter[key] = --fm[1].end();
		minFreq = 1;
		size++;
	}
};

class Solution {
public:
	bool isPrime(int a) {
		static set<int> prinmes{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 };
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
				idxes[S[i]] = { i, i };
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
		return (long long)(p2.y - p0.y) * (p1.x - p0.x) == (long long)(p1.y - p0.y) * (p2.x - p0.x);
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
				}
				else {
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
				}
				else {
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
			}
			else {
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


	//446. Arithmetic Slices II - Subsequence
	int numberOfArithmeticSlices(vector<int> &A) {
		if (A.empty()) {
			return 0;
		}
		int res = 0, n = A.size();
		vector<unordered_map<long long, int>> dp(n);
		set<int> s(A.begin(), A.end());
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				long long d = (long long)A[i] - (long long)A[j];
				if (d > INT32_MAX || d < INT32_MIN) {
					continue;
				}
				int t = 0;
				if (dp[j].count(d)) {
					t = dp[j][d];
					res += t;
				}
				dp[i][d] += 1 + t;
			}
		}
		return res;
	}

	int dp_691(string &target, vector<vector<int>> &stickers, map<string, int> &memo) {
		//        cout << target << endl;
		if (memo.find(target) != memo.end()) {
			return memo[target];
		}
		if (target.empty()) {
			return 0;
		}
		vector<int> tar(26);
		for (auto c : target) {
			tar[c - 'a']++;
		}
		int res = INT32_MAX;
		for (int i = 0; i < stickers.size(); i++) {
			vector<int> t = tar;
			int f = 0;
			for (int j = 0; j < 26; j++)
				if (tar[j] > 0 && stickers[i][j] > 0) {
					t[j] -= min(stickers[i][j], t[j]);
					f = 1;
				}
			if (f) {
				string newtarget;
				for (int j = 0; j < 26;) {
					if (t[j] > 0) {
						newtarget.push_back(j + 'a');
						t[j]--;
					}
					else {
						j++;
					}
				}
				//                cout << "use " << i << " ";
				int r = dp_691(newtarget, stickers, memo);
				if (r != -1) {
					res = min(res, r + 1);
				}
			}
		}
		if (res == INT32_MAX) {
			res = -1;
		}
		memo[target] = res;
		return res;
	}

	//691. Stickers to Spell Word
	int minStickers(vector<string> &stickers, string target) {
		int n = stickers.size();
		map<string, int> dp;
		vector<vector<int>> Stickers(n, vector<int>(26));
		for (int i = 0; i < n; i++) {
			for (auto c : stickers[i]) {
				Stickers[i][c - 'a']++;
			}
		}
		for (auto c : target) {
			int f = 1;
			for (int i = 0; i < n && f; ++i) {
				if (Stickers[i][c - 'a'] > 0) {
					f = 0;
					break;
				}
			}
			if (f) {
				return -1;
			}
		}

		return dp_691(target, Stickers, dp);
	}

	//649. Dota2 Senate
	string predictPartyVictory(string senate) {
		int r = 0, d = 0, N = senate.size(), R = 0, D = 0;
		for (auto c : senate) {
			if (c == 'R') {
				R++;
			}
			else {
				D++;
			}
		}
		vector<int> flags(N, 1);
		for (int i = 0; i < N;) {
			char c = senate[i];
			if (flags[i]) {
				if (c == 'R') {
					if (d > 0) {
						d--;
						flags[i] = 0;
						R--;

					}
					else {
						r++;
					}
				}
				else if (c == 'D') {
					if (r > 0) {
						r--;
						flags[i] = 0;
						D--;
					}
					else {
						d++;
					}
				}
			}
			if (i == N - 1 && R && D) {
				i = 0;
			}
			else if (D == 0) {
				return "Radiant";
			}
			else if (R == 0) {
				return "Dire";
			}
			else {
				i++;
			}
		}
	}

	//483. Smallest Good Base
	string smallestGoodBase(string n) {
		unsigned long long tn = (unsigned long long) stoll(n);
		unsigned long long x = 1;
		for (int i = 62; i >= 1; i--) {
			if ((x << i) < tn) {
				unsigned long long cur = mysolve(tn, i);
				if (cur != 0) return to_string(cur);
			}
		}
		return to_string(tn - 1);
	}

	unsigned long long mysolve(unsigned long long n, int d) {
		double tn = (double)n;
		unsigned long long right = (unsigned long long) (pow(tn, 1.0 / d) + 1);
		unsigned long long left = 1;
		while (left <= right) {
			unsigned long long mid = left + (right - left) / 2;
			unsigned long long sum = 1, cur = 1;
			for (int i = 1; i <= d; i++) {
				cur *= mid;
				sum += cur;
			}
			if (sum == n) return mid;
			if (sum > n) right = mid - 1;
			else left = mid + 1;
		}
		return 0;
	}

	//    //730. Count Different Palindromic Subsequences
	//    int countPalindromicSubsequences(string s) {
	//        int md = 1000000007;
	//        int n = s.size();
	//        int dp[3][n][4];
	//        for (int len = 1; len <= n; ++len) {
	//            for (int i = 0; i + len <= n; ++i)
	//                for (int x = 0; x < 4; ++x) {
	//                    int &ans = dp[2][i][x];
	//                    ans = 0;
	//                    int j = i + len - 1;
	//                    char c = 'a' + x;
	//                    if (len == 1) ans = s[i] == c;
	//                    else {
	//                        if (s[i] != c) ans = dp[1][i + 1][x];
	//                        else if (s[j] != c) ans = dp[1][i][x];
	//                        else {
	//                            ans = 2;
	//                            if (len > 2)
	//                                for (int y = 0; y < 4; ++y) {
	//                                    ans += dp[0][i + 1][y];
	//                                    ans %= md;
	//                                }
	//                        }
	//                    }
	//                }
	//            for (int i = 0; i < 2; ++i)
	//                for (int j = 0; j < n; ++j)
	//                    for (int x = 0; x < 4; ++x)
	//                        dp[i][j][x] = dp[i + 1][j][x];
	//        }
	//        int ret = 0;
	//        for (int x = 0; x < 4; ++x) ret = (ret + dp[2][0][x]) % md;
	//        return ret;
	//    }
	//766. Toeplitz Matrix
	bool isToeplitzMatrix(vector<vector<int>> &matrix) {
		int m = matrix.size(), n = matrix[0].size();
		for (int i = 0; i < n; i++) {
			int d = 0, x = 0, y = i, v = matrix[x][y];
			while (x + d < m && y + d < n) {
				if (matrix[x + d][y + d] != v)
					return false;
				d++;

			}
		}
		for (int i = 0; i < m; i++) {
			int d = 0, x = i, y = 0, v = matrix[x][y];
			while (x + d < m && y + d < n) {
				if (matrix[x + d][y + d] != v)
					return false;
				d++;
			}
		}
		return true;
	}

	//769. Max Chunks to Make Sorted (ver. 1)
	int maxChunksToSorted(vector<int> &arr) {
		int n = arr.size();

		int ans = 0, s = 0, e = 0;
		vector<int> vis(n, 0);
		for (int i = 0; i < n; i++)
			vis[arr[i]] = i;

		int vi = 1;
		for (int i = 0; i < n;) {
			if (i == arr[i]) {
				ans++;
				i++;
			}
			else {
				int right = vis[i] + 1;
				for (int t = 0; t < right; t++) {
					right = max(right, vis[t] + 1);
				}
				ans++;
				i = right;
			}
		}
		return ans;
	}

	//768. Max Chunks to Make Sorted (ver. 2)
	int maxChunksToSortedII(vector<int> &arr) {
		int n = arr.size();
		vector<vector<int>> pairs(n, vector<int>(2));
		for (int i = 0; i < n; i++) {
			pairs[i][0] = arr[i];
			pairs[i][1] = i;
		}
		sort(pairs.begin(), pairs.end());
		vector<int> idx(n);
		for (int i = 0; i < n; i++) {
			idx[i] = pairs[i][1];
		}
		int ans = 0;
		for (int i = 0; i < n;) {
			if (i == idx[i]) {
				ans++;
				i++;
			}
			else {
				int right = idx[i];
				for (int j = i; j <= right; j++) {
					right = max(right, idx[j]);
				}
				i = right + 1;
				ans++;
			}
		}
		return ans;
	}

	//767. Reorganize String
	string reorganizeString(string S) {
		priority_queue<pair<int, char>> pq;
		vector<int> counter(26);
		for (auto c : S) {
			counter[c - 'a']++;
		}

		for (int i = 0; i < 26; i++) {
			if (counter[i]) {
				pq.push({ counter[i], i + 'a' });
			}
		}
		string ret;
		while (!pq.empty()) {
			auto t = pq.top();
			pq.pop();
			if (ret.empty() || t.second != ret.back()) {
				ret.push_back(t.second);
				if (--t.first) {
					pq.push(t);
				}
			}
			else {
				if (!pq.empty()) {
					auto m = pq.top();
					pq.pop();
					ret.push_back(m.second);
					if (--m.first) {
						pq.push(m);
					}
					pq.push(t);
				}
				else {
					return "";
				}
			}
		}
		return ret;
	}

	//722. Remove Comments
	vector<string> removeComments(vector<string> &s) {
		vector<string> ans;
		bool inBlock = false;
		string newline;
		for (auto &line : s) {
			int Len = line.size();
			for (int i = 0; i < Len;) {
				if (!inBlock) {
					if (i == Len - 1) {
						newline.push_back(line[i++]);
					}
					else {
						auto m = line.substr(i, 2);
						if (m == "/*") {
							i += 2;
							inBlock = true;
						}
						else if (m == "//") {
							break;
						}
						else {
							newline.push_back(line[i++]);
						}
					}
				}
				else {
					if (i == Len - 1) {
						i++;
					}
					else {
						auto m = line.substr(i, 2);
						if (m == "*/") {
							i += 2;
							inBlock = false;
						}
						else {
							i++;
						}
					}
				}

			}
			if (!newline.empty() && !inBlock) {
				ans.push_back(newline);
				newline.clear();
			}
		}
		return ans;
	}

	int countPairs(vector<int> &nums, int mid) {
		int res = 0, Len = nums.size();
		for (int i = 0, j = 0; j < Len; j++) {
			while (i < j && nums[j] - nums[i] > mid) {
				i++;
			}
			res += j - i;
		}
		return res;
	}

	//719. Find K-th Smallest Pair Distance
	int smallestDistancePair(vector<int> &nums, int k) {
		sort(nums.begin(), nums.end());
		int Len = nums.size(), low = nums[1] - nums[0], high = nums[nums.size() - 1] - nums[0];
		for (int i = 1; i < Len; ++i) {
			low = min(low, nums[i] - nums[i - 1]);
		}

		while (low < high) {
			int m = (low + high) / 2, c = countPairs(nums, m);
			if (c >= k) {
				high = m;
			}
			else {
				low = m + 1;
			}

		}
		return low;
	}

	//207. Course Schedule
	bool canFinish(int numCourses, vector<pair<int, int>> &prerequisites) {
		vector<vector<int>> mat(numCourses);
		vector<int> inv(numCourses);
		for (auto &p : prerequisites) {
			mat[p.first].push_back(p.second);
			inv[p.second]++;
		}
		queue<int> q;
		for (int i = 0; i < numCourses; i++) {
			if (inv[i] == 0) {
				q.push(i);
			}
		}
		int cnt = 0;
		while (!q.empty()) {
			int cur = q.front();
			q.pop();
			cnt++;
			for (int n : mat[cur]) {
				inv[n]--;
				if (inv[n] == 0) {
					q.push(n);
				}
			}
		}
		return cnt == numCourses;
	}

	//210. Course Schedule II
	vector<int> findOrder(int numCourses, vector<pair<int, int>> &prerequisites) {
		vector<vector<int>> mat(numCourses);
		vector<int> inv(numCourses);
		for (auto &p : prerequisites) {
			mat[p.first].push_back(p.second);
			inv[p.second]++;
		}
		queue<int> q;
		for (int i = 0; i < numCourses; i++) {
			if (inv[i] == 0) {
				q.push(i);
			}
		}
		vector<int> ret;
		while (!q.empty()) {
			auto cur = q.front();
			q.pop();
			ret.push_back(cur);
			for (auto n : mat[cur]) {
				inv[n]--;
				if (inv[n] == 0) {
					q.push(n);
				}
			}
		}
		if (ret.size() == numCourses) {
			reverse(ret.begin(), ret.end());
			return ret;
		}
		else {
			return vector<int>();
		}
	}

	//630. Course Schedule III
	int scheduleCourse(vector<vector<int>> &courses) {
		sort(courses.begin(), courses.end(), [](vector<int> &p1, vector<int> &p2) {
			return p1[1] < p2[1];
		});
		priority_queue<int> pq;
		int ans = 0, e = 0;
		for (auto c : courses) {
			e += c[0];
			pq.push(c[0]);
			ans++;
			while (!pq.empty() && e > c[1]) {
				e -= pq.top();
				pq.pop();
				ans--;
			}
		}
		return ans;
	}
};

int main() {
	Solution sol;
	vector<vector<int>> course{ {100,  200},
							   {200,  1300},
							   {1000, 1250},
							   {200,  3200} };
	cout << sol.scheduleCourse(course) << endl;
	//    auto r = sol.findOrder(4, course);
	//    for (int i :r) {
	//        cout << i << " ";
	//    }
	//    cout << endl;
	//    cout << sol.canFinish(2, course) << endl;
	//    vector<int> nums{1, 2, 3, 4, 5, 0};
	//    nums = {1, 3, 1};
	//    nums = {60, 100, 4};
	//    cout << sol.smallestDistancePair(nums, 1) << endl;
	//    cout << sol.smallestDistancePair(nums, 2) << endl;
	//    cout << sol.smallestDistancePair(nums, 3) << endl;
		//nums = { 1,0,2,3,4 };
	//    cout << sol.maxChunksToSorted(nums) << endl;
		//vector<vector<int>> matrix{ { 1,2,3,4 },{ 5,1,2,3 },{ 9,5,1,2 } };
		//cout << sol.isToeplitzMatrix(matrix) << endl;
		//cout << sol.smallestGoodBase("26546") << endl;
		//    cout << sol.predictPartyVictory("RD") << endl;
		//    cout << sol.predictPartyVictory("D") << endl;
		//    cout << sol.predictPartyVictory("RDD") << endl;
		//    vector<string> stickers{"notice", "possible",};
		//    cout << sol.minStickers(stickers, "basicbasic");
		//    vector<int> nums{2, 4, 6, 8, 10};
		//    cout << sol.numberOfArithmeticSlices(nums) << endl;
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