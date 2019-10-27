#include <numeric>
#include <future>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <set>
#include <queue>
#include <assert.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <thread>
#include <functional>
#include <mutex>
#include <string>
using namespace std;
typedef long long ll;
/*
typedef pair<int, int> ii;
const int N = 1e5 + 10;
vector<ii> a[N];
int dfn[N], low[N];
int bcc[N];

int stamp;
void DFS(int u, int up_edge, vector<vector<int>>& ret) {
	static stack<int> S;
	low[u] = dfn[u] = stamp++;
	S.push(u);
	for (auto& it : a[u]) {
		if (it.second == up_edge) continue;
		int v = it.first;
		if (dfn[v] == 0) {
			DFS(v, it.second, ret);
			low[u] = min(low[u], low[v]);
			if (low[v] > dfn[u]) {
				ret.push_back({ u, v });
				while (S.top() != v) {
					bcc[S.top()] = v;
					S.pop();
				}
				bcc[v] = v;
				S.pop();
			}
		}
		else {
			low[u] = min(low[u], dfn[v]);
		}
	}
}

class Solution {
public:
	//1192. Critical Connections in a Network
	vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
		for (int i = 0; i < n; ++i) {
			a[i].clear();
			dfn[i] = 0;
		}
		for (int k = 0; k < connections.size(); ++k) {
			int x = connections[k][0];
			int y = connections[k][1];
			a[x].push_back({ y, k });
			a[y].push_back({ x, k });
		}
		stamp = 1;
		vector<vector<int>> ret;
		DFS(0, -1, ret);
		return ret;
	}
};
*/

class BIT
{
	int lowbit(int x)
	{
		return x & (-x);
	}
	int n;
public:
	vector<int> a;
	BIT(int n) : a(n + 1, 0), n(n)
	{

	}
	int add(int i, int v)
	{
		while (i <= n)
		{
			a[i] = max(a[i], v);
			i += lowbit(i);
		}
		return 0;
	}
	int query(int i)
	{
		int ans = 0;
		while (i > 0)
		{
			ans = max(ans, a[i]);
			i -= lowbit(i);
		}
		return ans;
	}
};


struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

//1195. Fizz Buzz Multithreaded
class FizzBuzz {
private:
	int n;
	condition_variable cv;
	mutex mtx;
	int i;
	bool finish;
public:
	FizzBuzz(int n) : i(1), finish(false) {
		this->n = n;
	}

	// printFizz() outputs "fizz".
	void fizz(function<void()> printFizz) {
		int last = -1;
		while (true)
		{
			unique_lock<mutex> lck(mtx);
			cv.wait(lck, [&]() {
				return (i % 3 == 0 && i % 5 && i != last) || finish;
				});
			if (finish) break;
			printFizz();
			last = i;
		}
	}

	// printBuzz() outputs "buzz".
	void buzz(function<void()> printBuzz) {
		int last = -1;
		while (true)
		{
			unique_lock<mutex> lck(mtx);
			cv.wait(lck, [&]() {
				return (i % 5 == 0 && i % 3 && last != i) || finish;
				});
			if (finish) break;
			printBuzz();
			last = i;
		}
	}

	// printFizzBuzz() outputs "fizzbuzz".
	void fizzbuzz(function<void()> printFizzBuzz) {
		int last = -1;
		while (true)
		{
			unique_lock<mutex> lck(mtx);
			cv.wait(lck, [&]() {
				return (last != i && i % 3 == 0 && i % 5 == 0) || finish;
				});
			if (finish) break;
			printFizzBuzz();
			last = i;
		}
	}

	// printNumber(x) outputs "x", where x is an integer.
	void number(function<void(int)> printNumber) {
		for (; i <= n; ++i)
		{
			if (i % 3 == 0 || i % 5 == 0)
			{
				cv.notify_all();
			}
			else
			{
				printNumber(i);
			}
		}
		finish = true;
		cv.notify_all();
	}
};



//1226. The Dining Philosophers
class DiningPhilosophers {
public:
	DiningPhilosophers() {
	}

	void wantsToEat(int philosopher,
		function<void()> pickLeftFork,
		function<void()> pickRightFork,
		function<void()> eat,
		function<void()> putLeftFork,
		function<void()> putRightFork) {
		int l = philosopher;
		int r = (philosopher + 1) % 5;
		if (philosopher % 2 == 0) {
			lock[r].lock();
			lock[l].lock();
			pickLeftFork();
			pickRightFork();
		}
		else {
			lock[l].lock();
			lock[r].lock();
			pickLeftFork();
			pickRightFork();
		}

		eat();
		putRightFork();
		putLeftFork();
		lock[l].unlock();
		lock[r].unlock();
	}
private:
	std::mutex lock[5];
};

class Solution
{
public:
	//1196. How Many Apples Can You Put into the Basket
	int maxNumberOfApples(vector<int>& a) {
		int n = a.size();
		sort(a.begin(), a.end());
		int u = 0;
		int const maxn = 5000;
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			u += a[i];
			if (u > maxn) break;
			ans++;
		}
		return ans;
	}

	//1197. Minimum Knight Moves
	int minKnightMoves(int X, int Y) {
		vector <int> dr = { -2, -2, -1, 1, 2, 2, 1, -1 };
		vector <int> dc = { 1, -1, -2, -2, -1, 1, 2, 2 };
		X = abs(X);
		Y = abs(Y);
		int x = min(X, Y);
		int y = max(X, Y);

		if (x == 0 && y == 0) {
			return 0;
		}
		if (x == 0 && y == 2) {
			return 2;
		}
		if (x == 1 && y == 1) {
			return 2;
		}
		if (x == 1 && y == 3) {
			return 2;
		}
		if (x == 3 && y == 4) {
			return 3;
		}
		return 1 + minKnightMoves(x - 1, y - 2);
	}

	//1198. Find Smallest Common Element in All Rows
	int smallestCommonElement(vector<vector<int>>& a) {
		int n = a.size();
		int m = a[0].size();
		for (int i = 0; i < m; ++i)
		{
			bool f = true;
			for (int j = 1; f && j < n; ++j)
			{
				auto it = lower_bound(a[j].begin(), a[j].end(), a[0][i]);
				if (it == a[j].end() || *it != a[0][i])
				{
					f = false;
				}
			}
			if (f) return a[0][i];
		}
		return -1;
	}

	//1200. Minimum Absolute Difference
	vector<vector<int>> minimumAbsDifference(vector<int>& a) {
		int n = a.size();
		sort(a.begin(), a.end());
		int diff = INT_MAX;
		for (int i = 1; i < n; ++i)
		{
			diff = min(diff, a[i] - a[i - 1]);
		}
		vector<vector<int>> ans;
		for (int i = 1; i < n; ++i)
		{
			if (a[i] - a[i - 1] == diff)
			{
				ans.push_back({ a[i - 1], a[i] });
			}
		}
		return ans;
	}

	ll check_1201(ll n, ll a, ll b, ll c)
	{
		ll ans = 0;
		ans += n / a;
		ans += n / b;
		ans += n / c;
		ans -= n / lcm(a, b);
		ans -= n / lcm(a, c);
		ans -= n / lcm(b, c);
		ans += n / lcm(lcm(a, b), c);
		return ans;
	}

	//1201. Ugly Number III
	int nthUglyNumber(int n, int a, int b, int c) {
		ll l = 1, r = 2 * 10e9;
		while (l < r)
		{
			ll m = (l + r) / 2;
			ll tot = check_1201(m, a, b, c);
			if (tot >= n)
			{
				r = m;
			}
			else
			{
				l = m + 1;
			}
		}
		return l;
	}

	void dfs_1202(int u, vector<vector<int>>& g, vector<int>& vis, int color)
	{
		if (vis[u]) return;
		vis[u] = color;
		for (auto& v : g[u])
		{
			dfs_1202(v, g, vis, color);
		}
	}
	//1202. Smallest String With Swaps
	string smallestStringWithSwaps(string s, vector<vector<int>>& a) {
		int n = s.size();
		vector<vector<int>> g(n);
		for (auto& e : a)
		{
			int u = e[0], v = e[1];
			g[u].push_back(v);
			g[v].push_back(u);
		}
		int color = 1;
		vector<int> vis(n);
		for (int i = 0; i < n; ++i)
		{
			if (vis[i] == 0)
			{
				dfs_1202(i, g, vis, color++);
			}
		}
		vector<vector<int>> index(color);
		vector<vector<int>> value(color);
		for (int i = 0; i < n; ++i)
		{
			int c = vis[i];
			index[c].push_back(i);
			value[c].push_back(s[i]);
		}
		for (auto& e : value)
		{
			sort(e.begin(), e.end());
		}
		for (int i = 0; i < index.size(); ++i)
		{
			for (int j = 0; j < index[i].size(); ++j)
			{
				int idx = index[i][j];
				char val = value[i][j];
				s[idx] = val;
			}
		}
		return s;
	}

	//1199. Minimum Time to Build Blocks
	int minBuildTime(vector<int>& a, int split) {
		priority_queue<int, vector<int>, greater<int>> pq(a.begin(), a.end());
		int ans = 0;
		while (pq.size() >= 2)
		{
			int x = pq.top(); pq.pop();
			int y = pq.top(); pq.pop();
			pq.push(split + y);
		}
		return pq.top();
	}

	bool topoSort(vector<vector<int>>& a, vector<int>& level)
	{
		int n = a.size();
		vector<int> order(n);
		vector<vector<int>> g(n);
		for (int i = 0; i < n; ++i)
		{
			order[i] = a[i].size();
			for (auto& v : a[i])
			{
				g[v].push_back(i);
			}
		}
		queue<int> q;
		vector<bool> vis(n);
		level.resize(n);
		fill(level.begin(), level.end(), 0);
		for (int i = 0; i < n; ++i)
		{
			if (order[i] == 0)
			{
				vis[i] = 1;
				q.push(i);
			}
		}
		int cur = 0;
		while (!q.empty())
		{
			int size = q.size();
			cur++;
			while (size--)
			{
				auto u = q.front(); q.pop();
				for (auto& v : g[u])
				{
					if (vis[v]) continue;
					if (--order[v] == 0)
					{
						q.push(v);
						level[v] = cur;
						vis[v] = 1;
					}
				}
			}
		}
		for (auto& e : order)
		{
			if (e)
			{
				return false;
			}
		}
		return true;
	}

	//1203. Sort Items by Groups Respecting Dependencies
	vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems) 
	{
		for (int& e : group)
		{
			if (e == -1)
			{
				e = m++;
			}
		}
		vector<int> item_level;
		if (!topoSort(beforeItems, item_level)) return {};
		vector<vector<int>> gps(m);
		for (int i = 0; i < n; ++i)
		{
			gps[group[i]].push_back(i);
		}
		vector<vector<int>> g_before(m);
		for (int i = 0; i < m; ++i)
		{
			for (auto& j : gps[i])
			{
				for (auto& v : beforeItems[j])
				{
					if (group[v] == i) continue;
					g_before[i].push_back(group[v]);
				}
			}
		}
		for (auto& e : g_before)
		{
			sort(e.begin(), e.end());
			e.erase(unique(e.begin(), e.end()), e.end());
		}
		vector<int> g_level;
		topoSort(g_before, g_level);
		for (auto& e : gps)
		{
			sort(e.begin(), e.end(), [&](int a, int b) {
				return item_level[a] < item_level[b];
				});
		}
		vector<int> gps_idx(m);
		iota(gps_idx.begin(), gps_idx.end(), 0);
		sort(gps_idx.begin(), gps_idx.end(), [&](int a, int b) {
			return g_level[a] < g_level[b];
			});
		vector<int> ans;
		ans.reserve(n);
		for (auto& i : gps_idx)
		{
			for (auto& e : gps[i])
			{
				ans.push_back(e);
			}
		}
		return ans;
	}
	//1207. Unique Number of Occurrences
	bool uniqueOccurrences(vector<int>& a) {
		map<int, int> cnt;
		for (auto& e : a)
		{
			cnt[e] ++;
		}
		set<int> pre;
		for (auto& e : cnt)
		{
			if (pre.count(e.second)) return false;
			pre.insert(e.second);
		}
		return true;
	}
	//1208. Get Equal Substrings Within Budget
	int equalSubstring(string s, string t, int V) {
		int n = s.size();
		vector<int> cost(n);
		for (int i = 0; i < n; ++i) cost[i] = abs(s[i] - t[i]);
		int j = 0;
		int cur = 0, ans = 0;
		for (int i = 0; i < n; ++i)
		{
			cur += cost[i];
			while (cur > V)
			{
				cur -= cost[j++];
			}
			ans = max(ans, i - j + 1);
		}
		return ans;
	}

	//1209. Remove All Adjacent Duplicates in String II
	string removeDuplicates(string s, int k) {
		vector<pair<char, int>> stk;
		for (auto& e : s)
		{
			if (!stk.empty() && stk.back().first == e)
			{
				if (++stk.back().second == k)
				{
					stk.pop_back();
				}
			}
			else
			{
				stk.push_back({ e, 1 });
			}
		}
		string ans;
		for (auto& e : stk)
		{
			ans += string(e.second, e.first);
		}
		return ans;
	}
	//1210. Minimum Moves to Reach Target with Rotations
	int minimumMoves(vector<vector<int>>& grid) {
		int f[105][105][2];
		int n = grid.size(), i, j, ans;
		memset(f, 127, sizeof(f));
		f[0][0][0] = 0;
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				if (!grid[i][j])
				{
					if (i + 1 < n && j + 1 < n && !grid[i + 1][j] && !grid[i][j + 1] && !grid[i + 1][j + 1])
					{
						f[i][j][0] = min(f[i][j][0], f[i][j][1] + 1);
						f[i][j][1] = min(f[i][j][1], f[i][j][0] + 1);
						f[i + 1][j][0] = min(f[i + 1][j][0], f[i][j][0] + 1);
						f[i][j + 1][1] = min(f[i][j + 1][1], f[i][j][1] + 1);
					}
					if (j + 2 < n && !grid[i][j + 1] && !grid[i][j + 2])
						f[i][j + 1][0] = min(f[i][j + 1][0], f[i][j][0] + 1);
					if (i + 2 < n && !grid[i + 1][j] && !grid[i + 2][j])
						f[i + 1][j][1] = min(f[i + 1][j][1], f[i][j][1] + 1);
				}
		ans = f[n - 1][n - 2][0];
		if (ans == 2139062143) ans = -1;
		return ans;
	}

	//1213. Intersection of Three Sorted Arrays
	vector<int> arraysIntersection(vector<int>& a, vector<int>& b, vector<int>& c) {
		set<int> ins;
		for (auto& e : a)
		{
			auto it = lower_bound(b.begin(), b.end(), e);
			if (it != b.end() && *it == e)
			{
				ins.insert(e);
			}
		}
		vector<int> ans;
		for (auto& e : c)
		{
			if (ins.count(e))
			{
				ans.push_back(e);
			}
		}
		ans.erase(unique(ans.begin(), ans.end()), ans.end());
		return ans;
	}

	void dfs_1214(TreeNode* u, vector<int>& out)
	{
		if (!u) return;
		dfs_1214(u->left, out);
		out.push_back(u->val);
		dfs_1214(u->right, out);
	}

	//1214. Two Sum BSTs
	bool twoSumBSTs(TreeNode* root1, TreeNode* root2, int target) {
		vector<int> a, b;
		dfs_1214(root1, a);
		dfs_1214(root2, b);
		int n = a.size(), m = b.size();
		vector<int> c(n + m);
		vector<int> pos(n + m);
		int i = 0, j = 0, k = 0;
		while (i < n || j < m)
		{
			if (i < n && j < m)
			{
				if (a[i] < b[j])
				{
					pos[k] = 1;
					c[k++] = a[i++];
				}
				else
				{
					pos[k] = 2;
					c[k++] = b[j++];
				}
			}
			else if (i < n)
			{
				pos[k] = 1;
				c[k++] = a[i++];
			}
			else
			{
				pos[k] = 2;
				c[k++] = b[j++];
			}
		}
		for (i = 0, j = 1; j < n + m; ++j)
		{
			if (c[i] != c[j])
			{
				c[++i] = c[j];
				pos[i] = pos[j];
			}
			else
			{
				pos[i] |= pos[j];
			}
		}
		j = 0;
		while (j < i)
		{
			int v = c[j] + c[i];
			if (v == target) return (pos[j] | pos[i]) == 3;
			else if (v > target) --i;
			else ++j;
		}
		return false;
	}

	void dfs_1215(ll num, int pre, int low, int high, set<int>& ans)
	{
		if (num > high) return;
		if (low <= num && num <= high)
		{
			ans.insert(num);
		}
		if (pre + 1 < 10) dfs_1215(num * 10 + pre + 1, pre + 1, low, high, ans);
		if (pre - 1 >= 0) dfs_1215(num * 10 + pre - 1, pre - 1, low, high, ans);
	}

	//1215. Stepping Numbers
	vector<int> countSteppingNumbers(int low, int high) {
		set<int> nums;
		for (int i = 0; i < 10; ++i)
		{
			dfs_1215(i, i, low, high, nums);
		}
		vector<int> ans(nums.begin(), nums.end());
		return ans;
	}

	//1216. Valid Palindrome III
	bool isValidPalindrome(string s, int k) {
		int n = s.size();
		vector<vector<int>> dp(n, vector<int>(n));
		for (int i = 0; i < n; ++i)
		{
			dp[i][i] = 1;
			if (i + 1 < n)
			{
				dp[i][i + 1] = 1 + (s[i] == s[i + 1]);
			}
		}
		for (int l = 3; l <= n; ++l)
		{
			for (int i = 0, j = i + l - 1; j < n; ++i, ++j)
			{
				if (s[i] == s[j])
				{
					dp[i][j] = dp[i + 1][j - 1] + 2;
				}
				else
				{
					dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
				}
			}
		}
		return dp[0][n - 1] + k >= n;
	}

	//1217. Play with Chips
	int minCostToMoveChips(vector<int>& a) {
		int ans = 0, n = a.size();
		for (auto& e : a) ans += e % 2;
		return min(ans, n - ans);
	}

	//1218. Longest Arithmetic Subsequence of Given Difference
	int longestSubsequence(vector<int>& a, int diff) {
		int const maxn = 1e4;
		int const offset = maxn;
		vector<int> dp(maxn * 2 + 1);
		for (auto e : a)
		{
			e += offset;
			if (e - diff >= 0 && e - diff < maxn * 2)
			{
				dp[e] = max(dp[e], dp[e - diff] + 1);
			}
			dp[e] = max(dp[e], 1);
		}
		return *max_element(dp.begin(), dp.end());
	}

	//1220. Count Vowels Permutation
	int countVowelPermutation(int n) {
		int const mod = 1e9 + 7;
		vector<long long> dp(5, 1);
		for (int i = 1; i < n; ++i)
		{
			vector<long long> nx(5);
			nx[0] = dp[1];
			nx[1] = (dp[0] + dp[2]) % mod;
			nx[2] = (dp[0] + dp[1] + dp[3] + dp[4]) % mod;
			nx[3] = (dp[2] + dp[4]) % mod;
			nx[4] = dp[0];
			dp = nx;
		}
		return accumulate(dp.begin(), dp.end(), 0ll) % mod;
	}

	int dfs_1219(int x, int y, vector<vector<int>>& g, vector<vector<bool>>& vis)
	{
		static int dr[] = { 0, 1, 0, -1 };
		static int dc[] = { 1, 0, -1, 0 };
		int ans = g[x][y];
		vis[x][y] = 1;
		int n = g.size(), m = g.size();
		for (int d = 0; d < 4; ++d)
		{
			int nx = x + dr[d], ny = y + dc[d];
			if (0 <= nx && nx < n && 0 <= ny && ny < m && vis[nx][ny] == 0 && g[nx][ny] > 0)
			{
				ans = max(ans, g[x][y] + dfs_1219(nx, ny, g, vis));
			}
		}
		vis[x][y] = 0;
		return ans;
	}
	//1219. Path with Maximum Gold
	int getMaximumGold(vector<vector<int>>& g) {
		int ans = 0;
		int n = g.size(), m = g.size();
		vector<vector<bool>> vis(n, vector<bool>(m));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				if (g[i][j])
				{
					ans = max(ans, dfs_1219(i, j, g, vis));
				}
			}
		}
		return ans;
	}

	//1221. Split a String in Balanced Strings
	int balancedStringSplit(string s) {
		int ans = 0;
		int cur = 0;
		for (auto& c : s)
		{
			if (c == 'L') cur++;
			else cur--;
			ans += cur == 0;
		}
		return ans;
	}

	//1222. Queens That Can Attack the King
	vector<vector<int>> queensAttacktheKing(vector<vector<int>>& qs, vector<int>& ks) {
		int x = ks[0], y = ks[1];
		const int n = 8;
		set<pair<int, int>> pos;
		for (auto& e : qs) pos.insert({ e[0], e[1] });
		vector<vector<int>> ans;
		for (int dx = -1; dx <= 1; dx += 1)
		{
			for (int dy = -1; dy <= 1; dy += 1)
			{
				if (dx == 0 && dy == 0) continue;
				int nx = x + dx, ny = y + dy;
				while (0 <= nx && nx < n && 0 <= ny && ny < n)
				{
					if (pos.count({ nx, ny }))
					{
						ans.push_back({ nx, ny });
						break;
					}
					nx += dx;
					ny += dy;
				}
			}
		}
		return ans;
	}

	//1223. Dice Roll Simulation
	int dieSimulator(int n, vector<int> rollMax) {
		long divisor = (long)pow(10, 9) + 7;
		vector<vector<long long>> dp(n, vector<long long>(7));
		for (int i = 0; i < 6; i++) {
			dp[0][i] = 1;
		}
		dp[0][6] = 6;
		for (int i = 1; i < n; i++) {
			long sum = 0;
			for (int j = 0; j < 6; j++) {
				dp[i][j] = dp[i - 1][6];
				if (i - rollMax[j] < 0) {
					sum = (sum + dp[i][j]) % divisor;
				}
				else {
					if (i - rollMax[j] - 1 >= 0) dp[i][j] = (dp[i][j] - (dp[i - rollMax[j] - 1][6] - dp[i - rollMax[j] - 1][j])) % divisor + divisor;
					else dp[i][j] = (dp[i][j] - 1) % divisor;
					sum = (sum + dp[i][j]) % divisor;
				}

			}
			dp[i][6] = sum;
		}
		return (int)(dp[n - 1][6]);
	}

	//1224. Maximum Equal Frequency
	int maxEqualFreq(vector<int>& nums) {
		vector<int> cnt(100001, 0), fre(100001, 0);
		int maxcnt = 0, ans = 0;
		for (int i = 0; i < nums.size(); ++i) {
			int num = nums[i];
			++cnt[num];
			++fre[cnt[num]];
			maxcnt = max(maxcnt, cnt[num]);
			if ((fre[maxcnt] == 1 &&
				maxcnt + (fre[maxcnt - 1] - 1) * (maxcnt - 1) == i + 1)
				|| (fre[maxcnt] * maxcnt + 1 == i + 1)
				)
				ans = i + 1;
		}
		if (maxcnt == 1)
			return nums.size();
		return ans;
	}

	//1227. Airplane Seat Assignment Probability
	double nthPersonGetsNthSeat(int n) {
		if (n == 1) return 1.0;
		//return 1.0 / n + (n - 2.0) / n * nthPersonGetsNthSeat(n - 1);
		return 0.5;
	}

	//1228. Missing Number In Arithmetic Progression
	int missingNumber(vector<int>& a) {
		//int n = a.size();
		//int first = a[0], last = a[0], sum = 0;
		//for (auto& e : a)
		//{
		//	first = min(first, e);
		//	last = max(last, e);
		//	sum += e;
		//}
		//return (first + last) * (n + 1) / 2 - sum;
		int n = a.size(), d = (a[n - 1] - a[0]) / n, left = 0, right = n;
		while (left < right) {
			int mid = (left + right) / 2;
			if (a[mid] == a[0] + d * mid)
				left = mid + 1;
			else
				right = mid;
		}
		return a[0] + d * left;
	}
	//1229. Meeting Scheduler
	vector<int> minAvailableDuration(vector<vector<int>>& a, vector<vector<int>>& b, int k) {
		//int n = a.size(), m = b.size();
		//sort(a.begin(), a.end());
		//sort(b.begin(), b.end());
		//int j = 0;
		//int nj = 0;
		//for (int i = 0; i < n; ++i)
		//{
		//	if (a[i][1] - a[i][0] < k) continue;
		//	while (j < m && b[j][1] < a[i][0]) j++;
		//	while (j < m && b[j][0] <= a[i][1])
		//	{
		//		if (i + 1 < n && b[j][1] < a[i + 1][0])
		//		{
		//			nj = j;
		//		}
		//		int t = min(a[i][1], b[j][1]);
		//		int s = max(a[i][0], b[j][0]);
		//		int len = t - s;
		//		if (len >= k) return { s, s + k };
		//		j++;
		//	}
		//	j = nj;
		//}
		//return {};
		sort(a.begin(), b.end()); // sort increasing by start time (default sorting by first value)
		sort(b.begin(), b.end()); // sort increasing by start time (default sorting by first value)

		int i = 0, j = 0;
		int n1 = a.size(), n2 = b.size();
		while (i < n1 && j < n2) {
			// Find intersect between slots1[i] and slots2[j]
			int intersectStart = max(a[i][0], b[j][0]);
			int intersectEnd = min(a[i][1], b[j][1]);

			if (intersectStart + k <= intersectEnd) // Found the result
				return { intersectStart, intersectStart + k };
			else if (a[i][1] < b[j][1])
				i++;
			else
				j++;
		}
		return {};
	}

	bool check_1231(vector<int>& a, int val, int k)
	{
		int cur = 0, cnt = 0;
		for (auto& e : a)
		{
			cur += e;
			if (cur >= val)
			{
				cnt++;
				cur = 0;
			}
		}
		return cnt >= k;
	}
	//1231. Divide Chocolate
	int maximizeSweetness(vector<int>& a, int k) {
		int l = INT_MAX, r = 0;
		for (auto& e : a)
		{
			l = min(l, e);
			r += e;
		}
		if (k == 0)
		{
			return r;
		}
		k++;
		while (l < r)
		{
			int m = l + (r - l) / 2;
			if (!check_1231(a, m, k))
			{
				r = m;
			}
			else
			{
				l = m + 1;
			}
		}
		return l - 1;
	}

	//1230. Toss Strange Coins
	double probabilityOfHeads(vector<double>& prob, int m) {
		int n = prob.size();
		vector<double> dp(m + 1);
		dp[0] = 1.0;
		for (int i = 0; i < n; ++i)
		{
			for (int j = min(m, i + 1); j >= 0; --j)
			{
				dp[j] = dp[j] * (1 - prob[i]);
				if (j > 0)
				{
					dp[j] += dp[j - 1] * prob[i];
				}
			}
		}
		return dp[m];
	}

	//1232. Check If It Is a Straight Line
	bool checkStraightLine(vector<vector<int>>& a) {
		int n = a.size();
		if (n <= 2) return true;
		sort(a.begin(), a.end());
		for (int i = 2; i < n; ++i)
		{
			if ((a[1][1] - a[0][1]) * (a[i][0] - a[0][0]) != (a[i][1] - a[0][1]) * (a[1][0] - a[0][0]))
				return false;
		}
		return true;
	}

	//1233. Remove Sub-Folders from the Filesystem
	vector<string> removeSubfolders(vector<string>& a) {
		sort(a.begin(), a.end());
		vector<string> ans;
		for (auto& e : a)
		{
			if (ans.empty()) ans.push_back(e);
			else
			{
				if (e.size() >= ans.back().size() && e.substr(0, ans.back().size()) == ans.back() && (e.size() == ans.size() || e[ans.back().size()] == '/'))
				{

				}
				else
				{
					ans.push_back(e);
				}
			}
		}
		return ans;
	}

	bool valid_1234(vector<int>& cnt, int n)
	{
		for (auto& e : cnt)
		{
			if (e > n / 4) return false;
		}
		return true;
	}
	//1234. Replace the Substring for Balanced String
	int balancedString(string s) {
		int n = s.size();
		vector<int> cnt(4);
		map<char, int> idx;
		idx['Q'] = 0;
		idx['W'] = 1;
		idx['E'] = 2;
		idx['R'] = 3;
		int i = 0, j = 0;
		for (auto& c : s)
		{
			cnt[idx[c]] ++;
		}
		if (valid_1234(cnt, n)) return 0;
		int ans = n;
		for (; i < n; ++i)
		{
			cnt[idx[s[i]]] --;
			while (j <= i && valid_1234(cnt, n))
			{
				ans = min(ans, i - j + 1);
				cnt[idx[s[j++]]] ++;
			}
		}
		return ans;
	}
	int jobScheduling(vector<int>& st, vector<int>& ed, vector<int>& pf) {
		int n = st.size();
		vector<int> pts(st.begin(), st.end());
		for (auto& e : ed) pts.push_back(e);
		sort(pts.begin(), pts.end());
		pts.erase(unique(pts.begin(), pts.end()), pts.end());
		int tot = pts.size();
		map<int, int> idx;
		for (int i = 0; i < tot; ++i)
		{
			idx[pts[i]] = i + 1;
		}
		vector<vector<int>> a(n, vector<int>(3));
		for (int i = 0; i < n; ++i)
		{
			a[i] = { idx[ed[i]], idx[st[i]], pf[i] };
		}
		sort(a.begin(), a.end());

		BIT bt(tot + 1);
		int ans = 0;
		vector<int> dp(tot);
		for (int i = 0; i < n; ++i)
		{
			int s = a[i][1], t = a[i][0], p = a[i][2];
			int last = bt.query(s);
			if ((i > 0 && dp[i - 1] < p + last) || i == 0)
			{
				dp[i] = p + last;
			}
			else if (i > 0)
			{
				dp[i] = dp[i - 1];
			}
			bt.add(t, dp[i]);
		}
		return dp[n - 1];
	}

	//bool check(string& s, string& p)
	//{
	//	return s == p || (s.size() > p.size() && s.substr(0, p.size()) == p && s[p.size()] == '/');
	//}
	//1236. Web Crawler
	//vector<string> crawl(string startUrl, HtmlParser htmlParser) {
	//	string domain = "http://";
	//	for (int i = 7; i < startUrl.size(); ++i)
	//	{
	//		if (startUrl[i] == '/') break;
	//		domain.push_back(startUrl[i]);
	//	}
	//	queue<string> urls;
	//	urls.push(startUrl);
	//	set<string> vis;
	//	vector<string> ans;
	//	ans.push_back(startUrl);
	//	vis.insert(startUrl);
	//	while (!urls.empty())
	//	{
	//		auto u = urls.front();
	//		urls.pop();
	//		for (auto& sub : htmlParser.getUrls(u))
	//		{
	//			if (!vis.count(sub) && check(sub, domain))
	//			{
	//				vis.insert(sub);
	//				ans.push_back(sub);
	//				urls.push(sub);
	//			}
	//		}
	//	}
	//	return ans;
	//}

	//1237. Find Positive Integer Solution for a Given Equation
	//vector<vector<int>> findSolution(CustomFunction& fun, int z) {
	//	vector<vector<int>> ans;
	//	int const N = 1000;
	//	for (int i = 1; i <= N; ++i)
	//	{
	//		int l = 1, r = 1000;
	//		while (l <= r)
	//		{
	//			int m = (l + r) / 2;
	//			int val = fun.f(i, m);
	//			if (val == z)
	//			{
	//				ans.push_back({ i, m });
	//				break;
	//			}
	//			else if (val > z)
	//			{
	//				r = m - 1;
	//			}
	//			else
	//			{
	//				l = m + 1;
	//			}
	//		}
	//	}
	//	return ans;
	//}

	//1238. Circular Permutation in Binary Representation
	vector<int> circularPermutation(int n, int start) {
		int cnt = (1 << n);
		vector<int> ans(cnt);
		vector<int> tmp;
		int offset = 0;
		bool f = true;
		for (int i = 0; i < cnt; ++i)
		{
			ans[i] = (i ^ (i / 2));
			if (ans[i] == start)
			{
				offset = i;
				f = false;
			}
			if (f)
			{
				tmp.push_back(ans[i]);
			}
		}
		int j = 0;
		for (int i = offset; i < cnt; ++i)
		{
			ans[j++] = ans[i];
		}
		for (int i = 0; i < tmp.size(); ++i)
		{
			ans[j++] = tmp[i];
		}
		return ans;
	}

	int convert_1239(string& s)
	{
		int ret = 0;
		for (auto& c : s)
		{
			int mask = (1 << (c - 'a'));
			if (ret & mask) return -1;
			ret |= mask;
		}
		return ret;
	}
	int dfs_1239(int u, int s, int len, vector<pair<int, int>>& a)
	{
		if (u == len) return 0;
		int ans = 0;
		for (int i = u; i < len; ++i)
		{
			if (s & a[i].first) continue;
			ans = max(ans, a[i].second + dfs_1239(i + 1, s | a[i].first, len, a));
		}
		return ans;
	}
	//1239. Maximum Length of a Concatenated String with Unique Characters
	int maxLength(vector<string>& arr) {
		int n = arr.size();
		vector<pair<int, int>> a;
		for (int i = 0; i < n; ++i)
		{
			int ret = convert_1239(arr[i]);
			if (ret != -1)
			{
				a.push_back({ ret, static_cast<int>(arr[i].length()) });
			}
		}
		return dfs_1239(0, 0, a.size(), a);
	}

	//1240. Tiling a Rectangle with the Fewest Squares
	int tilingRectangle(int n, int m) {
		static vector<vector<int>> M = {
			{1 },
			{2, 1 },
			{3, 3, 1 },
			{4, 2, 4, 1 },
			{5, 4, 4, 5, 1 },
			{6, 3, 2, 3, 5, 1 },
			{7, 5, 5, 5, 5, 5, 1 },
			{8, 4, 5, 2, 5, 4, 7, 1 },
			{9, 6, 3, 6, 6, 3, 6, 7, 1 },
			{10, 5, 6, 4, 2, 4, 6, 5, 6, 1 },
			{11, 7, 6, 6, 6, 6, 6, 6, 7, 6, 1 },
			{12, 6, 4, 3, 6, 2, 6, 3, 4, 5, 7, 1 },
			{13, 8, 7, 7, 6, 6, 6, 6, 7, 7, 6, 7, 1 },
			{14, 7, 7, 5, 7, 5, 2, 5, 7, 5, 7, 5, 7, 1 },
			{15, 9, 5, 7, 3, 4, 8, 8, 4, 3, 7, 5, 8, 7, 1},
		};
		if (n < m) swap(n, m);
		return M[n - 1][m - 1];
	}
};

int main()
{
	
	return 0;
}