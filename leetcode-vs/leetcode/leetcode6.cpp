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
	vector<vector<int>> flipAndInvertImage(vector<vector<int>>& A) {
		int m = A.size(), n = A[0].size();
		for (int i = 0; i < m; ++i)
		{
			reverse(A[i].begin(), A[i].end());
			for (int j = 0; j < n; ++j)
				A[i][j] = !A[i][j];
		}
		return A;
	}
	//833. Find And Replace in String
	string findReplaceString(string S, vector<int>& indexes, vector<string>& sources, vector<string>& targets) {
		int len = S.size();
		vector<pair<string, string>> replaces(len);
		int leni = indexes.size();
		for (int i = 0; i < leni; ++i)
		{
			replaces[indexes[i]] = { sources[i],targets[i] };
		}
		string ans;
		int i = 0;
		while (i < len)
		{
			if (replaces[i].first != "")
			{
				int j = 0, t = i;
				while (j < replaces[i].first.size() && t < len)
				{
					if (replaces[i].first[j] == S[t])
					{
						++t;
						++j;
					}
					else break;
				}
				if (j == replaces[i].first.size())
				{
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
	int count(int ox, int oy, int len, vector<vector<int>>& A, vector<vector<int>>& B)
	{
		int cnt = 0;
		for (int i = 0; i < len; ++i)
		{
			for (int j = 0; j < len; ++j)
			{
				int ai = (i + ox + len) % len, aj = (j + oy + len) % len;
				//int ai = (i + ox), aj = (j + oy);
				//if (0 <= ai && ai < len && 0 <= aj && aj < len)
				if (B[i][j] == A[ai][aj] && B[i][j] == 1) cnt += 1;
			}
		}
		return cnt;
	}
	//835. Image Overlap
	int largestOverlap(vector<vector<int>>& A, vector<vector<int>>& B) {
		int ans = 0, len = A.size();
		for (int ox = 0; ox < len; ++ox)
			for (int oy = 0; oy < len; ++oy)
			{
				//if (count(ox, oy, len, A, B) == 5) cout << ox << " " << oy << endl;
				ans = max(ans, count(ox, oy, len, A, B));
			}
		return ans;
	}

	//834. Sum of Distances in Tree
	vector<int> sumOfDistancesInTree(int N, vector<vector<int>>& edges) {
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

	void dfs(int root, unordered_set<int>& seen, int n, vector<unordered_set<int>> &tree, vector<int> &count, vector<int> &res) {
		seen.insert(root);
		for (auto i : tree[root])
			if (seen.count(i) == 0) {
				dfs(i, seen, n, tree, count, res);
				count[root] += count[i];
				res[root] += res[i] + count[i];
			}
		count[root] += 1;
	}

	void dfs2(int root, unordered_set<int>& seen, int n, vector<unordered_set<int>> &tree, vector<int> &count, vector<int> &res) {
		seen.insert(root);
		for (auto i : tree[root])
			if (seen.count(i) == 0) {
				res[i] = res[root] - count[i] + n - count[i];
				dfs2(i, seen, n, tree, count, res);
			};
	}
};

int main() {
	return 0;
}