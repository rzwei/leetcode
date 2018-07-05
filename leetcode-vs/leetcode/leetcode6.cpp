//
// Created by rzhon on 18/4/22.
//
#include "headers.h"
#include <bitset>
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
#include <list>

using namespace std;

class Master {
public:
	int getmatch(string &a, string &b) {
		int len = a.size();
		int ans = 0;
		for (int i = 0; i < len; ++i) {
			if (a[i] == b[i]) ans++;
		}
		return ans;
	}

	string ss = "acckzz";

	int guess(string word) {
		if (word == ss) cout << "find" << endl;
		return getmatch(ss, word);
	};
};

class DSU {
	vector<int> parent;
	vector<int> rank;
	vector<int> sz;
public:
	DSU(int N) : parent(N), rank(N), sz(N, 1) {
		for (int i = 0; i < N; ++i)
			parent[i] = i;
	}

	int find(int x) {
		if (parent[x] != x) parent[x] = find(parent[x]);
		return parent[x];
	}

	void Union(int x, int y) {
		int xr = find(x), yr = find(y);
		if (xr == yr) return;

		if (rank[xr] < rank[yr]) {
			int tmp = yr;
			yr = xr;
			xr = tmp;
		}
		if (rank[xr] == rank[yr])
			rank[xr]++;

		parent[yr] = xr;
		sz[xr] += sz[yr];
	}

	int size(int x) {
		return sz[find(x)];
	}

	int top() {
		return size(sz.size() - 1) - 1;
	}
};

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
			}
			else {
				t.insert(fronts[i]);
				t.insert(backs[i]);
			}
		}
		for (auto n : t) {
			if (!f.count(n))
				ans = min(n, ans);
		}
		return ans == INT_MAX ? 0 : ans;
	}

	class Tire {
	public:
		Tire * next[26];

		Tire() : next{ nullptr } {}
	};

	//820. Short Encoding of Words
	int minimumLengthEncoding(vector<string> &words) {
		auto cmp = [](string &a, string &b) {
			return a.length() > b.length();
		};
		sort(words.begin(), words.end(), cmp);
		int ans = 0;
		Tire *root = new Tire();
		for (auto &word : words) {
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
				long long v = (long long)(A[i]) * A[j];
				if (idx.count(v)) {
					int n = (long long)(dp[i]) * dp[j] % mod;
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

	//825. Friends Of Appropriate Ages
	int numFriendRequests(vector<int> &ages) {
		int ans = 0;
		vector<int> nums(120 + 1);
		for (auto n : ages) {
			nums[n] += 1;
		}
		for (int a = 1; a <= 120; ++a) {
			for (int b = 1; b <= 120; ++b) {
				if (2 * b - 14 <= a || b > a || b > 100 && a < 100) {
					continue;
				}
				if (a == b) {
					ans += nums[a] * (nums[a] - 1);
				}
				else {
					ans += nums[a] * nums[b];
				}
			}
		}
		return ans;
	}

	//826. Most Profit Assigning Work
	int maxProfitAssignment(vector<int> &difficulty, vector<int> &profit, vector<int> &worker) {
		int len = difficulty.size();
		if (len == 0) {
			return 0;
		}
		vector<int> dp(len);
		vector<pair<int, int>> nums(len);
		for (int i = 0; i < len; ++i) {
			nums[i] = { difficulty[i], -profit[i] };
		}
		sort(nums.begin(), nums.end());
		dp[0] = -nums[0].second;
		for (int i = 1; i < len; ++i) {
			dp[i] = max(dp[i - 1], -nums[i].second);
		}
		int ans = 0;
		sort(worker.begin(), worker.end());
		int j = worker.size() - 1, i = len - 1;

		while (j >= 0) {
			while (i >= 0 && nums[i].first > worker[j]) {
				--i;
			}
			if (i < 0) break;
			ans += dp[i];
			j -= 1;
		}
		return ans;
	}

	int dfs_island(int x, int y, vector<vector<int>> &grid, int idx) {
		static vector<vector<int>> dirs = { {0,  1},
										   {0,  -1},
										   {1,  0},
										   {-1, 0} };
		int m = grid[0].size(), n = grid[1].size();
		grid[x][y] = idx;
		int cnt = 1;
		for (auto &d : dirs) {
			int nx = x + d[0], ny = y + d[1];
			if (0 <= nx && nx < m && 0 <= ny && ny < n && grid[nx][ny] == 1) {
				cnt += dfs_island(nx, ny, grid, idx);
			}
		}
		return cnt;
	}

	//827. Making A Large Island
	int largestIsland(vector<vector<int>> &grid) {
		int m = grid[0].size(), n = grid[1].size();
		unordered_map<int, int> cnt;
		int idx = -1;
		int ans = 0;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (grid[i][j] == 1) {
					int t = dfs_island(i, j, grid, idx);
					cnt[idx] = t;
					ans = max(ans, t);
					--idx;
				}
			}
		}
		vector<vector<int>> dirs = { {0,  1},
									{0,  -1},
									{1,  0},
									{-1, 0} };
		unordered_set<int> cc;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (grid[i][j] == 0) {
					cc.clear();
					for (auto &d : dirs) {
						int nx = i + d[0], ny = j + d[1];
						if (0 <= nx && nx < m && 0 <= ny && ny < n) {
							if (grid[nx][ny] < 0) {
								cc.insert(grid[nx][ny]);
							}
						}
					}
					int t = 1;
					for (auto cx : cc) {
						t += cnt[cx];
					}
					ans = max(ans, t);
				}
			}
		}
		return ans;
	}

	bool ck(char ch) {
		string v = "aeiouAEIOU";
		return (v.find(ch) != string::npos);
	}

	//824. Goat Latin
	string toGoatLatin(string S) {
		istringstream is(S);
		vector<string> res;
		string s;
		string A = "";
		while (is >> s) {
			A += "a";
			string t = "";
			if (ck(s[0])) {
				t = s + "ma";
			}
			else {
				t = s.substr(1) + s[0] + "ma";
			}
			t += A;
			res.push_back(t);
		}
		string ans = res[0];
		for (int i = 1; i < (int)res.size(); i++) {
			ans += " " + res[i];
		}
		return ans;
	}

	//830. Positions of Large Groups
	vector<vector<int>> largeGroupPositions(string S) {
		int last = -1, len = S.size(), cnt = 0, lasti = -1;
		vector<vector<int>> ans;
		for (int i = 0; i <= len; ++i) {
			if (i == len || S[i] != last) {
				if (cnt >= 3)
					ans.push_back({ lasti, i - 1 });
				last = S[i];
				lasti = i;
				cnt = 1;
			}
			else {
				cnt += 1;
			}
		}
		return ans;
	}

	//829. Consecutive Numbers Sum
	int consecutiveNumbersSum(int N) {
		int x = N, ans = 0;;
		for (int i = 1; 2 * x > (i - 1) * i; ++i) {
			long long a2 = 2 * x / i - i + 1;
			long long a = a2 / 2;
			if ((a * i + (i - 1) * i / 2) == x) {
				ans += 1;
			}
		}
		return ans;
	}

	//828. Unique Letter String
	int uniqueLetterString(string S) {
		unordered_set<char> letters;
		for (auto c : S)
			letters.insert(c);
		int ans = 0;
		int const mod = 1e9 + 7;
		int len = S.size();
		for (auto c : letters) {
			int cnt0 = 0, cnt1 = 0, i = 0;
			while (i < len && S[i] != c) {
				cnt0 += 1;
				++i;
			}
			cnt0 += 1;
			i += 1;
			cnt1 = 1;

			for (; i <= len; ++i) {
				if (i == len || S[i] == c) {
					ans = (ans + cnt0 % mod * cnt1 % mod) % mod;
					cnt0 = cnt1;
					cnt1 = 1;
				}
				else {
					cnt1 += 1;
				}
			}
		}
		return ans;
	}

	//832. Flipping an Image
	vector<vector<int>> flipAndInvertImage(vector<vector<int>> &A) {
		int m = A.size(), n = A[0].size();
		for (int i = 0; i < m; ++i) {
			reverse(A[i].begin(), A[i].end());
			for (int j = 0; j < n; ++j)
				A[i][j] = !A[i][j];
		}
		return A;
	}

	//833. Find And Replace in String
	string findReplaceString(string S, vector<int> &indexes, vector<string> &sources, vector<string> &targets) {
		int len = S.size();
		vector<pair<string, string>> replaces(len);
		int leni = indexes.size();
		for (int i = 0; i < leni; ++i) {
			replaces[indexes[i]] = { sources[i], targets[i] };
		}
		string ans;
		int i = 0;
		while (i < len) {
			if (replaces[i].first != "") {
				int j = 0, t = i;
				while (j < replaces[i].first.size() && t < len) {
					if (replaces[i].first[j] == S[t]) {
						++t;
						++j;
					}
					else break;
				}
				if (j == replaces[i].first.size()) {
					ans += replaces[i].second;
					i = t;
				}
				else {
					ans.push_back(S[i]);
					++i;
				}
			}
			else
				ans.push_back(S[i++]);
		}
		return ans;
	}

	int count(int ox, int oy, int len, vector<vector<int>> &A, vector<vector<int>> &B) {
		int cnt = 0;
		for (int i = 0; i < len; ++i) {
			for (int j = 0; j < len; ++j) {
				int ai = (i + ox + len) % len, aj = (j + oy + len) % len;
				//int ai = (i + ox), aj = (j + oy);
				//if (0 <= ai && ai < len && 0 <= aj && aj < len)
				if (B[i][j] == A[ai][aj] && B[i][j] == 1) cnt += 1;
			}
		}
		return cnt;
	}

	//835. Image Overlap
	int largestOverlap(vector<vector<int>> &A, vector<vector<int>> &B) {
		int ans = 0, len = A.size();
		for (int ox = 0; ox < len; ++ox)
			for (int oy = 0; oy < len; ++oy) {
				//if (count(ox, oy, len, A, B) == 5) cout << ox << " " << oy << endl;
				ans = max(ans, count(ox, oy, len, A, B));
			}
		return ans;
	}

	//834. Sum of Distances in Tree
	vector<int> sumOfDistancesInTree(int N, vector<vector<int>> &edges) {
		vector<unordered_set<int>> tree;
		vector<int> res;
		vector<int> count;

		if (N == 1) return res;
		int n = N;

		tree.resize(N);
		res.assign(N, 0);
		count.assign(N, 0);
		for (auto e : edges) {
			tree[e[0]].insert(e[1]);
			tree[e[1]].insert(e[0]);
		}
		unordered_set<int> seen1, seen2;
		dfs(0, seen1, n, tree, count, res);
		dfs2(0, seen2, n, tree, count, res);
		return res;

	}

	void dfs(int root, unordered_set<int> &seen, int n, vector<unordered_set<int>> &tree, vector<int> &count,
		vector<int> &res) {
		seen.insert(root);
		for (auto i : tree[root])
			if (seen.count(i) == 0) {
				dfs(i, seen, n, tree, count, res);
				count[root] += count[i];
				res[root] += res[i] + count[i];
			}
		count[root] += 1;
	}

	void dfs2(int root, unordered_set<int> &seen, int n, vector<unordered_set<int>> &tree, vector<int> &count,
		vector<int> &res) {
		seen.insert(root);
		for (auto i : tree[root])
			if (seen.count(i) == 0) {
				res[i] = res[root] - count[i] + n - count[i];
				dfs2(i, seen, n, tree, count, res);
			};
	}

	//836. Rectangle Overlap
	bool isRectangleOverlap(vector<int> &rec1, vector<int> &rec2) {
		int xl = max(rec1[0], rec2[0]), yl = max(rec1[1], rec2[1]);
		int xr = min(rec1[2], rec2[2]), yr = min(rec1[3], rec2[3]);
		return xl < xr && yl < yr;
	}

	//838. Push Dominoes
	string pushDominoes(string dominoes) {
		int len = dominoes.size();
		vector<int> lefts(len, -1), rights(len, -1);
		int last = -1;
		for (int i = 0; i < len; ++i) {
			if (dominoes[i] == 'R')
				last = i;
			else if (dominoes[i] == '.')
				rights[i] = last;
			else if (dominoes[i] == 'L')
				last = -1;
		}
		last = -1;
		for (int i = len - 1; i >= 0; --i) {
			if (dominoes[i] == 'L')
				last = i;
			else if (dominoes[i] == '.')
				lefts[i] = last;
			else if (dominoes[i] == 'R')
				last = -1;
		}
		string ans = dominoes;
		for (int i = 0; i < len; ++i) {
			if (dominoes[i] != '.') continue;
			int l = lefts[i], r = rights[i];
			if (l == -1 && r == -1) continue;
			if (l != -1 && r == -1) {
				ans[i] = 'L';
			}
			else if (l == -1 && r != -1) {
				ans[i] = 'R';
			}
			else if (l != -1 && r != -1) {
				if (i - r == l - i) continue;
				if (i - r > l - i) ans[i] = 'L';
				else ans[i] = 'R';
			}
		}
		return ans;
	}

	bool check(string &a, string &b) {
		int f = 0;
		for (int i = 0; i < a.size(); ++i)
			if (a[i] != b[i]) f++;
		return f <= 2;
	}

	//839. Similar String Groups
	int numSimilarGroups(vector<string> &A) {
		int len = A.size();
		DSU dsu(A.size());
		for (int i = 0; i < len; ++i) {
			for (int j = i + 1; j < len; ++j) {
				if (check(A[i], A[j])) {
					dsu.Union(i, j);
				}
			}
		}
		int ans = 0;
		for (int i = 0; i < len; ++i)
			if (dsu.find(i) == i) ans++;
		return ans;
	}

	//837. New 21 Game
	double new21Game(int N, int K, int W) {
		if (K == 0) return 1.0;
		vector<double> dp(N + W + 1);
		for (int i = K; i <= N; ++i)
			dp[i] = 1;
		double S = min(N - K + 1, W);
		for (int k = K - 1; k >= 0; --k) {
			dp[k] = S / W;
			S += dp[k] - dp[k + W];
		}
		return dp[0];
	}

	bool vaild(vector<vector<int>> &grid) {
		vector<int> vis(9);
		int n = 3, s = 0;
		for (int i = 0; i < n; ++i) {
			s += grid[i][0];
		}
		for (int i = 0; i < n; ++i) {
			int t = 0;
			for (int j = 0; j < n; ++j) {
				t += grid[i][j];

				if (grid[i][j] <= 0 || grid[i][j] > 9)
					return false;
				if (vis[grid[i][j] - 1]) return false;
				vis[grid[i][j] - 1] = 1;
			}
			if (t != s) return false;
			t = 0;
			for (int j = 0; j < n; ++j) {
				t += grid[j][i];
			}
			if (t != s) return false;
		}
		int a = 0, b = 0;
		for (int i = 0; i < n; ++i) {
			a += grid[i][i];
			b += grid[i][n - 1 - i];
		}
		if (a != s || b != s) return false;
		return true;
	}

	//840. Magic Squares In Grid
	int numMagicSquaresInside(vector<vector<int>> &grid) {
		if (grid.empty()) return 0;
		int m = grid.size(), n = grid[0].size();
		if (m < 3 || n < 3) return 0;
		int ans = 0;
		for (int i = 0; i <= m - 3; ++i)
			for (int j = 0; j <= n - 3; ++j) {
				vector<vector<int>> tmp(3, vector<int>(3));
				for (int ii = 0; ii < 3; ++ii) {
					for (int jj = 0; jj < 3; ++jj) {
						tmp[ii][jj] = grid[ii + i][jj + j];
					}
				}
				if (vaild(tmp))
					ans++;
			}
		return ans;
	}

	//0 white 1 gray 2 black
	void dfs(int u, vector<int> &color, vector<vector<int>> &rooms) {
		if (color[u] == 2) return;
		if (color[u] == 1) return;
		color[u] = 1;
		for (int v : rooms[u]) {
			dfs(v, color, rooms);
		}
		color[u] = 2;
	}

	//841. Keys and Rooms
	bool canVisitAllRooms(vector<vector<int>> &rooms) {
		int n = rooms.size();
		vector<int> vis(n);
		dfs(0, vis, rooms);
		for (int i = 0; i < n; ++i) {
			if (vis[i] == 0)
				return false;
		}
		return true;
	}

	typedef long long ll;

	bool judge(int s, ll pre, string &num, vector<int> &path) {
		int len = num.size();
		for (int j = s; j <= len; ++j) {
			if (num[s] == '0' && j > s) break;
			ll nv = stoll(num.substr(s, j - s + 1));
			if (nv > INT_MAX) break;
			ll nans = nv + pre;
			string ns = to_string(nans);
			int jj = 0;
			for (int k = j + 1; k < len; ++k) {
				if (ns[jj] != num[k]) break;
				++jj;
				if (jj == ns.size()) break;
			}
			if (jj == ns.size()) {
				path.push_back(nv);
				if (j + jj + 1 == num.size()) {
					ll xxx = stoll(num.substr(j + 1, jj));
					if (xxx > INT_MAX) {
						path.pop_back();
						return false;
					}
					path.push_back(xxx);
					return true;
				}
				if (judge(j + 1, nv, num, path))
					return true;
				path.pop_back();
			}
		}
		return false;
	}

	//842. Split Array into Fibonacci Sequence
	vector<int> splitIntoFibonacci(string num) {
		ll pre = 0, len = num.size();
		vector<int> path;
		for (int i = 0; i < len - 1; ++i) {
			if (num[0] == '0' && i > 0) break;

			pre = pre * 10 + num[i] - '0';
			if (pre > INT_MAX) break;

			path.push_back(pre);
			if (judge(i + 1, pre, num, path))
				return path;
			path.pop_back();
		}
		return {};
	}

	int getmatch(string &a, string &b) {
		int len = a.size();
		int ans = 0;
		for (int i = 0; i < len; ++i) {
			if (a[i] == b[i]) ans++;
		}
		return ans;
	}

	//843. Guess the Word
	void findSecretWord(vector<string> &wordlist, Master &master) {
		int len = wordlist.size();
		vector<int> valid(len);
		for (int i = 0; i < len; ++i)
			valid[i] = i;
		int cnt = 0;
		while (true && !valid.empty()) {
			vector<int> next;
			int idx = valid[rand() % valid.size()];
			int match = master.guess(wordlist[idx]);
			if (match == 6) return;
			for (int nx : valid) {
				if (nx != idx) {
					if (getmatch(wordlist[idx], wordlist[nx]) == match)
						next.push_back(nx);
				}
			}
			valid = next;
			cnt++;
			if (cnt > 10) break;
		}
	}

	//844. Backspace String Compare
	bool backspaceCompare(string S, string T) {
		vector<char> a, b;
		for (char c : S) {
			if (c == '#') {
				if (!a.empty()) a.pop_back();
			}
			else
				a.push_back(c);
		}
		for (char c : T) {
			if (c == '#') {
				if (!b.empty()) b.pop_back();
			}
			else
				b.push_back(c);
		}
		return a == b;
	}

	//845. Longest Mountain in Array
	int longestMountain(vector<int> &A) {
		int len = A.size();
		vector<int> a(len, 1), b(len, 1);
		for (int i = 1; i < len; ++i) {
			if (A[i] > A[i - 1])
				a[i] += a[i - 1];
		}

		for (int i = len - 2; i >= 0; --i) {
			if (A[i] > A[i + 1])
				b[i] += b[i + 1];
		}

		int ans = 0;
		for (int i = 0; i < len; ++i) {
			if (a[i] != 1 && b[i] != 1)
				ans = max(ans, a[i] + b[i] - 1);
		}
		return ans < 3 ? 0 : ans;
	}

	//846. Hand of Straights
	bool isNStraightHand(vector<int> &hand, int W) {
		if (hand.size() % W) return false;
		priority_queue<int, vector<int>, greater<int>> pq(hand.begin(), hand.end());
		for (int i = 0; i < hand.size() / W; ++i) {
			vector<int> tmp, a;
			for (int j = 0; j < W; ++j) {
				if (tmp.empty()) {
					int v = pq.top();
					pq.pop();
					tmp.push_back(v);
				}
				else {
					while (!pq.empty() && pq.top() != tmp.back() + 1) {
						a.push_back(pq.top());
						pq.pop();
					}
					if (!pq.empty() && pq.top() == tmp.back() + 1) {
						tmp.push_back(pq.top());
						pq.pop();
					}
					else return false;
				}
			}
			for (int e : a)pq.push(e);
		}
		return true;
	}

	//847. Shortest Path Visiting All Nodes
	int maxn = INT_MAX / 10;

	int dfs(int u, bitset<13> &s, const vector<vector<int>> &G, vector<vector<int>> &memo) {
		int n = G.size();
		int &ans = memo[s.to_ulong()][u];
		if (ans != -1) return ans;
		ans = maxn;
		int f = 1;
		for (int v = 0; v < n; ++v) {
			if (s[v] == 0) {
				f = 0;
				s[v] = 1;
				ans = min(ans, dfs(v, s, G, memo) + G[u][v]);
				s[v] = 0;
			}
		}
		if (f) ans = 0;
		return ans;
	}

	int shortestPathLength(vector<vector<int>> &graph) {
		int n = graph.size();
		vector<vector<int>> G(n, vector<int>(n, 1e7));
		for (int i = 0; i < n; ++i) {
			G[i][i] = 0;
			for (int j : graph[i])
				G[i][j] = 1;
		}
		for (int k = 0; k < n; ++k)
			for (int i = 0; i < n; ++i)
				for (int j = 0; j < n; ++j)
					G[i][j] = min(G[i][j], G[i][k] + G[k][j]);
		int ans = maxn;
		vector<vector<int>> memo(1 << n, vector<int>(n, -1));
		for (int i = 0; i < n; ++i) {
			bitset<13> vis;
			vis[i] = 1;
			ans = min(ans, dfs(i, vis, G, memo));
		}
		return ans;
	}

	int shortestPathLength_dp(vector<vector<int>> &graph) {
		int const maxn = INT_MAX / 10;
		int n = graph.size();
		vector<vector<int>> dp(1 << n, vector<int>(n, maxn));
		for (int x = 0; x < n; ++x)
			dp[1 << x][x] = 0;
		for (int s = 0; s < (1 << n); ++s) {
			bool repeat = true;
			while (repeat) {
				repeat = false;
				for (int u = 0; u < n; ++u) {
					int d = dp[s][u];
					for (int v : graph[u]) {
						int ns = s | (1 << v);
						if (d + 1 < dp[ns][v]) {
							dp[ns][v] = d + 1;
							if (s == ns) repeat = true;
						}
					}
				}
			}
		}
		int ans = maxn;
		for (int e : dp[(1 << n) - 1])
			ans = min(ans, e);
		return ans;
	}

	int shortestPathLength_bfs(vector<vector<int>> &g) {
		int n = g.size();
		int const maxn = INT_MAX / 8;
		typedef pair<int, int> ii;
		queue<ii> q;
		vector<vector<int>> dp(1 << n, vector<int>(n, maxn));
		for (int i = 0; i < n; ++i) {
			dp[1 << i][i] = 0;
			q.emplace(1 << i, i);
		}
		while (!q.empty()) {
			int mask = q.front().first;
			int u = q.front().second;
			q.pop();
			for (int v : g[u]) {
				int nx_mask = mask | (1 << v);
				if (dp[nx_mask][v] != maxn) continue;
				dp[nx_mask][v] = dp[mask][u] + 1;
				q.emplace(nx_mask, v);
			}
		}
		int res = maxn;
		for (int i = 0; i < n; ++i) {
			res = min(res, dp[(1 << n) - 1][i]);
		}
		return res;
	}

	//848. Shifting Letters
	string shiftingLetters(string s, vector<int> &a) {
		int n = a.size();
		for (int i = n - 2; i >= 0; --i) {
			a[i] = (a[i] + a[i + 1]) % 26;
		}
		for (int i = 0; i < n; ++i) {
			s[i] = (s[i] - 'a' + a[i]) % 26 + 'a';
		}
		return s;
	}

	//849. Maximize Distance to Closest Person
	int maxDistToClosest(vector<int> &a) {
		int n = a.size();
		vector<int> right(n);
		int last = 1000000;
		for (int i = n - 1; i >= 0; --i) {
			if (a[i] == 1) last = i;
			else right[i] = last;
		}
		last = -1000000;
		int ans = 0;
		for (int i = 0; i < n; ++i) {
			if (a[i] == 0) {
				ans = max(ans, min(i - last, right[i] - i));
			}
			else last = i;
		}
		return ans;
	}

	int dfs_851(int u, vector<int> &q, vector<vector<int>> &g, vector<bool> &vis, vector<int> &ans) {
		if (vis[u]) return ans[u];
		vis[u] = 1;
		if (g[u].empty()) {
			ans[u] = u;
			return u;
		}
		int low = u;
		for (int v : g[u]) {
			int nv = dfs_851(v, q, g, vis, ans);
			if (q[low] > q[nv]) {
				low = nv;
			}
		}
		ans[u] = low;
		return low;
	}

	//851. Loud and Rich
	vector<int> loudAndRich(vector<vector<int>> &richer, vector<int> &quiet) {
		int n = quiet.size();
		vector<vector<int>> g(n);
		for (auto &e : richer) {
			int u = e[0], v = e[1];
			g[v].push_back(u);
		}
		vector<bool> vis(n);
		vector<int> ans(n);
		for (int i = 0; i < n; ++i)
			ans[i] = i;
		for (int i = 0; i < n; ++i) {
			dfs_851(i, quiet, g, vis, ans);
		}
		return ans;
	}

	//850. Rectangle Area II
	int rectangleArea(vector<vector<int>> &a) {
		typedef long long int64;
		const ll mod = 1e9 + 7;
		int n = a.size();
		vector<int> X, Y;
		for (int i = 0; i < n; ++i) {
			X.push_back(a[i][0]);
			X.push_back(a[i][2]);
			Y.push_back(a[i][1]);
			Y.push_back(a[i][3]);
		}
		sort(X.begin(), X.end());
		X.erase(unique(X.begin(), X.end()), X.end());
		sort(Y.begin(), Y.end());
		Y.erase(unique(Y.begin(), Y.end()), Y.end());
		vector<vector<bool>> visit(X.size(), vector<bool>(Y.size()));
		for (int i = 0; i < n; ++i) {
			int x1 = lower_bound(X.begin(), X.end(), a[i][0]) - X.begin();
			int x2 = lower_bound(X.begin(), X.end(), a[i][2]) - X.begin();
			int y1 = lower_bound(Y.begin(), Y.end(), a[i][1]) - Y.begin();
			int y2 = lower_bound(Y.begin(), Y.end(), a[i][3]) - Y.begin();
			for (int u = x1; u < x2; ++u) {
				for (int v = y1; v < y2; ++v) {
					visit[u][v] = true;
				}
			}
		}
		int ret = 0;
		for (int i = 0; i < X.size(); ++i) {
			for (int j = 0; j < Y.size(); ++j) {
				if (visit[i][j]) {
					ret = (ret + (int64)(X[i + 1] - X[i]) * (Y[j + 1] - Y[j])) % mod;
				}
			}
		}
		return ret;
	}

	//852. Peak Index in a Mountain Array
	int peakIndexInMountainArray(vector<int> &A) {
		for (int i = 1; i < A.size() - 1; ++i) {
			if (A[i] > A[i - 1] && A[i] > A[i + 1]) return i;
		}
		return -1;
	}

	//853. Car Fleet
	int carFleet(int target, vector<int> &position, vector<int> &speed) {
		if (position.empty()) return 0;
		int n = position.size();
		vector<pair<int, int>> a(n);
		for (int i = 0; i < n; ++i) {
			a[i].first = position[i];
			a[i].second = speed[i];
		}
		sort(a.begin(), a.end());
		double ft = (target - a.back().first) / double(a.back().second);
		int ans = n;
		for (int i = n - 2; i >= 0; --i) {
			double tt = (target - a[i].first) / double(a[i].second);
			if (tt <= ft) {
				ans--;
			}
			else ft = tt;
		}
		return ans;
	}

	//855. Exam Room
	class ExamRoom {
	public:
		set<int> s;
		int n;

		ExamRoom(int N) {
			n = N;
		}

		int seat() {
			if (s.empty()) {
				s.insert(0);
				return 0;
			}
			else if (s.size() == 1) {
				int v = *s.begin();
				if (v >= n - 1 - v) {
					s.insert(0);
					return 0;
				}
				else {
					s.insert(n - 1);
					return n - 1;
				}
			}
			else {
				int d = *s.begin();
				int v = 0;
				auto bound = --s.end();
				for (auto it = s.begin(); it != bound; ++it) {
					auto nx = ++it;
					--it;
					if ((*nx - *it) / 2 > d) {
						d = (*nx - *it) / 2;
						v = *it + (*nx - *it) / 2;
					}
				}
				auto it = --s.end();
				if (n - 1 - *it > d) {
					v = n - 1;
				}
				s.insert(v);
				return v;
			}
		}


		void leave(int p) {
			s.erase(p);
		}
	};

	int dfs_854(string &a, string &b, int idx, unordered_map<string, int> &memo) {
		if (idx == a.size()) {
			return 0;
		}
		if (a[idx] == b[idx]) return dfs_854(a, b, idx + 1, memo);
		string key = a + to_string(idx);
		if (memo.count(key)) return memo[key];
		int ans = 100000;
		for (int i = idx + 1; i < a.size(); ++i) {
			if (a[i] == b[idx]) {
				swap(a[idx], a[i]);
				ans = min(ans, dfs_854(a, b, idx + 1, memo) + 1);
				swap(a[idx], a[i]);
			}
		}
		memo[key] = ans;
		return ans;
	}

	//854. K-Similar Strings
	int kSimilarity(string A, string B) {
		if (A == B) return 0;
		unordered_map<string, int> memo;
		return dfs_854(A, B, 0, memo);
	}

	//859. Buddy Strings
	bool buddyStrings(string A, string B) {
		if (A.size() != B.size()) return false;
		int n = A.size();
		int cnt = 0;
		vector<int> idx;
		for (int i = 0; i < n; ++i)
		{
			if (A[i] != B[i])
			{
				idx.push_back(i);
				if (idx.size() == 2) break;;
			}
		}
		if (idx.empty())
		{
			map<char, int> cnt;
			int f = 0;
			for (char c : A)
			{
				cnt[c]++;
				if (cnt[c] > 1) { f = 1; break; }
			}
			return f;
		}
		swap(A[idx[0]], A[idx[1]]);
		return A == B;
	}


	int parse(int &i, string &s)
	{
		int ans = 0;
		while (i < s.size())
		{

			if (s[i] == '(')
			{
				if (i + 1 < s.size() && s[i + 1] == ')')
				{
					ans++;
					i += 2;
				}
				else {
					i++;
					ans += 2 * (parse(i, s));
				}
			}
			else {
				i++;
				return ans;
			}
		}
		return ans;
	}
	//856. Score of Parentheses
	int scoreOfParentheses(string S) {
		int i = 0;
		return parse(i, S);
	}


	int gcd(int a, int b)
	{
		int t;
		while (b)
		{
			t = a % b;
			a = b;
			b = t;
		}
		return a;
	}
	//858. Mirror Reflection
	int mirrorReflection(int p, int q) {

		int d = p * q / gcd(p, q);
		int i = d / q;

		if (i & 1)
		{
			if ((d / p) & 1) return 1;
			else return 0;
		}
		else {
			return 2;
		}
		return -1;
	}
	//857. Minimum Cost to Hire K Workers
	double mincostToHireWorkers(vector<int>& q, vector<int>& w, int K) {
		vector<vector<double>> workers;
		for (int i = 0; i < q.size(); ++i)
			workers.push_back({ (double)(w[i]) / q[i], (double)q[i] });
		sort(workers.begin(), workers.end());
		double res = numeric_limits<double>::max(), qsum = 0;
		priority_queue<int> pq;
		for (auto worker : workers) {
			qsum += worker[1], pq.push(worker[1]);
			if (pq.size() > K) qsum -= pq.top(), pq.pop();
			if (pq.size() == K) res = min(res, qsum * worker[0]);
		}
		return res;
	}

	//860. Lemonade Change
	bool lemonadeChange(vector<int>& bills) {
		int a = 0, b = 0, c = 0;
		for (int e : bills)
		{
			if (e == 5) a++;
			else if (e == 10) {
				if (a == 0) return false;
				a--;
				b++;
			}
			else if (e == 20)
			{
				if (b && a) {
					b--;
					a--;
				}
				else {
					if (a >= 3)
					{
						a -= 3;
					}
					else
						return false;
				}
			}
		}
		return true;
	}

	void dfs_862(TreeNode *cur, vector<vector<int>> &g)
	{
		if (!cur) return;
		if (cur->left)
		{
			g[cur->val].push_back(cur->left->val);
			g[cur->left->val].push_back(cur->val);
			dfs_862(cur->left, g);
		}
		if (cur->right)
		{
			g[cur->val].push_back(cur->right->val);
			g[cur->right->val].push_back(cur->val);
			dfs_862(cur->right, g);
		}
	}
	//862. Shortest Subarray with Sum at Least K
	vector<int> distanceK(TreeNode* root, TreeNode* target, int K) {
		vector<int> ans;
		if (K == 0) return { target->val };
		vector<vector<int>> g(101);
		dfs_862(root, g);
		queue<int> q;
		q.push(target->val);
		vector<bool> vis(101);
		vis[target->val] = 1;
		while (K)
		{
			K--;
			int size = q.size();
			while (size--)
			{
				auto u = q.front();
				q.pop();
				for (auto v : g[u])
				{
					if (!vis[v])
					{
						q.push(v);
						vis[v] = 1;
					}
				}
			}
		}
		while (!q.empty())
		{
			ans.push_back(q.front());
			q.pop();
		}
		return ans;
	}
	//861. Score After Flipping Matrix
	int matrixScore(vector<vector<int>>& a) {
		int n = a.size(), m = a[0].size();
		for (int i = 0; i < n; ++i)
		{
			if (a[i][0] == 0)
			{
				for (int j = 0; j < m; ++j)
				{
					a[i][j] ^= 1;
				}
			}
		}
		for (int j = 1; j < m; ++j)
		{
			int cnt = 0;
			for (int i = 0; i < n; ++i)
			{
				cnt += a[i][j];
			}
			if (cnt <= n / 2)
			{
				for (int i = 0; i < n; ++i)
				{
					a[i][j] ^= 1;
				}
			}
		}
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			int t = 0;
			for (int j = 0; j < m; ++j)
			{
				t = t * 2 + a[i][j];
			}
			ans += t;
		}
		return ans;
	}

	//862. Shortest Subarray with Sum at Least K
	int shortestSubarray(vector<int>& A, int K) {
		int n = A.size();
		vector<pair<int, long long> > all;
		all.push_back(make_pair(0, 0LL));
		long long sum = 0LL;
		int answer = -1;
		for (int i = 1; i <= A.size(); ++i) {
			sum += A[i - 1];
			while (!all.empty() && all.back().second >= sum) {
				all.pop_back();
			}
			int left = 0, right = all.size() - 1;
			while (left <= right) {
				int mid = (left + right >> 1);
				if (sum - all[mid].second >= K) {
					left = mid + 1;
					int may = i - all[mid].first;
					if (answer < 0 || answer > may) {
						answer = may;
					}
				}
				else {
					right = mid - 1;
				}
			}
			all.push_back(make_pair(i, sum));
		}
		return answer;
	}
	int shortestSubarray_sliding_window(vector<int>& A, int K) {
		int n = A.size();
		vector<long long> sums(n + 1);
		for (int i = 1; i <= n; ++i)
		{
			sums[i] = sums[i - 1] + A[i - 1];
		}
		int ans = n + 1;
		deque<int> q;
		for (int i = 0; i < n + 1; ++i)
		{
			while (!q.empty() && sums[i] <= sums[q.back()])
			{
				q.pop_back();
			}
			while (!q.empty() && sums[i] >= sums[q.front()] + K)
			{
				ans = min(ans, i - q.front());
				q.pop_front();
			}
			q.push_back(i);
		}
		return ans == n + 1 ? -1 : ans;
	}


	bool check(vector<int> &nums, int k, double avg)
	{
		double sum = 0;
		for (int i = 0; i < k; ++i)
		{
			sum += nums[i] - avg;
		}
		if (sum >= 0) return true;
		double minv = 0;
		double pre = 0;
		for (int i = k; i < nums.size(); ++i)
		{
			sum += nums[i] - avg;
			pre += nums[i - k] - avg;
			minv = min(minv, pre);
			if (sum >= minv) return true;
		}
		return false;
	}

	//644. Maximum Average Subarray II
	double findMaxAverage(vector<int> nums, int k) {
		int mi = INT_MAX, mx = INT_MIN;
		for (int e : nums)
		{
			mi = min(mi, e);
			mx = max(mx, e);
		}
		double l = mi, r = mx;
		while (fabs(r - l) > 1e-7)
		{
			double m = (l + r) / 2;
			if (check(nums, k, m))
			{
				l = m;
			}
			else {
				r = m;
			}
		}
		return l;
	}

	//864. Random Pick with Blacklist
	Solution() {}
	unordered_map<int, int> a;
	int n, m;
	Solution(int N, vector<int> blacklist) :n(N) {
		int s = 0, i = 0, len = blacklist.size();
		m = n - blacklist.size();
		for (int e : blacklist) a[e] = -1;
		for (int e : blacklist)
		{
			if (e < m)
			{
				while (a.count(N - 1)) N--;
				a[e] = N - 1;
				N--;
			}
		}
	}

	int pick() {
		int idx = rand() % m;
		if (a.count(idx)) return a[idx];
		return idx;
	}


};

int main() {
	Solution sol;
	//cout << sol.mirrorReflection(3, 2) << endl;
	//string a = "aabc", b = "abca";
	//cout << sol.kSimilarity(a, b) << endl;
	//    Solution::ExamRoom room(10);
	//    cout << room.seat() << endl;
	//    cout << room.seat() << endl;
	//    cout << room.seat() << endl;
	//    cout << room.seat() << endl;
	//    room.leave(4);
	//    cout << room.seat() << endl;
	//    vector<vector<int>> G;
	//    G = {{1},
	//         {0, 2, 4},
	//         {1, 3, 4},
	//         {2},
	//         {1, 2}};
	//    cout << sol.shortestPathLength(G) << endl;
		//cout << sol.new21Game(21, 17, 10) << endl;
	return 0;
}