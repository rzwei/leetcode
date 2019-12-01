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

//1244. Design A Leaderboard
class Leaderboard {
public:
	map<int, int> sc;
	map<int, set<int>> mem_top;
	Leaderboard() {

	}

	void addScore(int id, int c) {
		if (sc.count(id))
		{
			mem_top[sc[id]].erase(id);
			if (mem_top[sc[id]].empty())
			{
				mem_top.erase(sc[id]);
			}
		}
		sc[id] += c;
		mem_top[sc[id]].insert(id);
	}

	int top(int K) {
		auto it = mem_top.rbegin();
		int ans = 0;
		while (K > 0)
		{
			ans += it->first * min(K, (int)it->second.size());
			K -= it->second.size();
			if (it == mem_top.rend())
				break;
			it++;
		}
		return ans;
	}

	void reset(int id) {
		int old = sc[id];
		mem_top[old].erase(id);
		if (mem_top[old].empty())
		{
			mem_top.erase(old);
		}
		sc[id] = 0;
		mem_top[sc[id]].insert(id);
	}
};

//1261. Find Elements in a Contaminated Binary Tree
class FindElements {
public:
	set<int> nums;
	void dfs(TreeNode* u, int val)
	{
		if (!u) return;
		nums.insert(val);
		u->val = val;
		dfs(u->left, val * 2 + 1);
		dfs(u->right, val * 2 + 2);
	}

	FindElements(TreeNode* root) {
		nums.clear();
		dfs(root, 0);
	}

	bool find(int target) {
		return nums.count(target);
	}
};

/*
//1263. Minimum Moves to Move a Box to Their Target Location
int dr[] = { 0, 1, 0, -1 };
int dc[] = { 1, 0, -1, 0 };
int idr[] = { 0, -1, 0, 1 };
int idc[] = { -1, 0, 1, 0 };
class Solution {
public:
	int n, m;
	bool ok(int x, int y)
	{
		return 0 <= x && x < n && 0 <= y && y < m;
	}
	int check(vector<vector<char>>& g, int sx, int sy, int ex, int ey)
	{
		if (g[sx][sy] == '#') return -1;
		if (sx == ex && sy == ey) return 0;
		queue<pair<int, int>> q;
		q.push({ sx, sy });
		vector<vector<bool>> vis(n, vector<bool>(m));
		vis[sx][sy] = true;
		int level = 0;
		while (!q.empty())
		{
			int size = q.size();
			level++;
			while (size--)
			{
				auto [x, y] = q.front(); q.pop();
				for (int d = 0; d < 4; ++d)
				{
					int nx = x + dr[d], ny = y + dc[d];
					if (ok(nx, ny) && g[nx][ny] == '.' && !vis[nx][ny])
					{
						if (nx == ex && ny == ey)
						{
							return level;
						}
						vis[nx][ny] = 1;
						q.push({ nx, ny });
					}
				}
			}
		}
		return -1;
	}

	struct State
	{
		int x, y;
		int px, py;
		bool operator < (const State& rhs) const
		{
			if (x != rhs.x) return  x < rhs.x;
			if (y != rhs.y) return  y < rhs.y;
			if (px != rhs.px) return  px < rhs.px;
			//if (py != rhs.py) return  py < rhs.py;
			return  py < rhs.py;
		}
	};

	int minPushBox(vector<vector<char>>& g) {
		n = g.size(), m = g[0].size();
		State st;
		int tx, ty;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				if (g[i][j] == 'B')
				{
					st.x = i;
					st.y = j;
					g[i][j] = '.';
				}
				if (g[i][j] == 'S')
				{
					st.px = i;
					st.py = j;
					g[i][j] = '.';
				}
				if (g[i][j] == 'T')
				{
					tx = i;
					ty = j;
					g[i][j] = '.';
				}
			}
		}
		queue<State> q;
		q.push(st);
		map<State, int> dp;
		dp[st] = 0;
		int ans = 0;
		while (!q.empty())
		{
			int size = q.size();
			ans++;
			while (size--)
			{
				auto curr = q.front();
				auto [x, y, px, py] = q.front(); q.pop();
				int cost = dp[curr];
				for (int d = 0; d < 4; ++d)
				{
					int nx = x + dr[d], ny = y + dc[d];
					int pnx = x + idr[d], pny = y + idc[d];
					if (ok(nx, ny) && ok(pnx, pny) && g[nx][ny] != '#' && g[pnx][pny] != '#')
					{
						g[px][py] = 'S';
						g[x][y] = 'B';
						int move = check(g, px, py, pnx, pny);
						g[px][py] = '.';
						g[x][y] = '.';
						if (move != -1)
						{
							if (nx == tx && ny == ty)
							{
								return ans;
							}
							State nxt = { nx, ny, x, y };
							if (!dp.count(nxt))
							{
								dp[nxt] = cost + 1;
								q.push(nxt);
							}
							else if (dp[nxt] > cost + 1)
							{
								dp[nxt] = cost + 1;
								q.push(nxt);
							}
						}
					}
				}
			}
		}
		return -1;
	}
};
*/

/*
//1258. Synonymous Sentences
class Solution {
public:
	map<string, string> fa;
	map<string, set<string>> order;
	vector<string> split(string& s)
	{
		int i = 0, n = s.size();
		string u;
		vector<string> ans;
		while (i < n)
		{
			while (i < n && s[i] == ' ') ++i;
			while (i < n && s[i] != ' ')
			{
				u.push_back(s[i++]);
			}
			if (!u.empty())
			{
				ans.push_back(u);
				u.clear();
			}
		}
		if (!u.empty())
		{
			ans.push_back(u);
			u.clear();
		}
		return ans;
	}
	string find(string& s, map<string, string>& fa)
	{
		if (fa[s] != s) return fa[s] = find(fa[s], fa);
		return s;
	}
	void merge(string& u, string& v, map<string, string>& fa)
	{
		auto fu = find(u, fa), fv = find(v, fa);
		fa[fu] = fv;
	}
	void dfs(string u, int i, vector<string>& tokens, vector<string>& ans)
	{
		if (i == tokens.size())
		{
			if (!u.empty())
				u.pop_back();
			ans.push_back(u);
			return;
		}
		auto& val = tokens[i];
		if (fa.count(val))
		{
			auto fval = find(val, fa);
			for (auto& sub_value : order[fval])
			{
				dfs(u + sub_value + " ", i + 1, tokens, ans);
			}
		}
		else
		{
			dfs(u + val + " ", i + 1, tokens, ans);
		}
	}

	vector<string> generateSentences(vector<vector<string>>& sy, string text) {
		vector<string> tokens = split(text);
		fa.clear();
		order.clear();
		for (auto& e : sy)
		{
			auto& u = e[0], & v = e[1];
			if (!fa.count(u)) fa[u] = u;
			if (!fa.count(v)) fa[v] = v;
			merge(u, v, fa);
		}

		for (auto& e : sy)
		{
			auto& u = e[0], & v = e[1];
			auto fu = find(u, fa);
			order[fu].insert(u);
			order[fu].insert(v);
		}
		vector<string> ans;
		dfs("", 0, tokens, ans);
		return ans;
	}
};
*/

/*
//1268. Search Suggestions System
class TrieNode {
public:
	char v;
	int isword;
	TrieNode() :isword(0), next{ nullptr } {}
	TrieNode(int c) :v(c), isword(0), next{ nullptr } {}
	TrieNode* next[26];
};


class Trie {
public:
	TrieNode* root;
	Trie() {
		root = new TrieNode();
	}

	void add(string word) {
		TrieNode* cur = root;
		for (char i : word) {
			if (cur->next[i - 'a'] == nullptr)
				cur->next[i - 'a'] = new TrieNode(i);
			cur = cur->next[i - 'a'];
		}
		cur->isword++;
	}

	void search(string& s, vector<string>& res, TrieNode* u)
	{
		if (!u) return;

		if (u->isword > 0)
		{
			int num = min(int(3 - res.size()), u->isword);
			for (int i = 0; i < num; ++i)
			{
				res.push_back(s);
			}
			if (res.size() == 3) return;
		}
		for (int i = 0; i < 26; ++i)
		{
			if (u->next[i])
			{
				s.push_back(i + 'a');
				search(s, res, u->next[i]);
				s.pop_back();
				if (res.size() == 3) return;
			}
		}
	}
	TrieNode* topThree(char c, TrieNode* u, string& prefix, vector<string>& res)
	{
		if (!u) return u;
		prefix.push_back(c);
		res.clear();
		search(prefix, res, u->next[c - 'a']);
		return u->next[c - 'a'];
	}
};



class Solution {
public:
	vector<vector<string>> suggestedProducts(vector<string>& products, string searchWord) {
		Trie tr;
		for (auto& e : products)
		{
			tr.add(e);
		}
		vector<vector<string>> res;
		string prefix;
		TrieNode* u = tr.root;
		for (auto& c : searchWord)
		{
			vector<string> cur;
			u = tr.topThree(c, u, prefix, cur);
			res.push_back(cur);
		}
		return res;
	}
};
*/

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

	//1246. Palindrome Removal
	int minimumMoves(vector<int>& a) {
		int n = a.size();
		vector<vector<int>> dp(n, vector<int>(n));
		for (int i = 0; i < n; ++i) dp[i][i] = 1;
		for (int l = 2; l <= n; ++l)
		{
			for (int i = 0, j = i + l - 1; j < n; ++i, ++j)
			{
				dp[i][j] = INT_MAX;
				if (a[i] == a[j])
				{
					dp[i][j] = l > 2 ? dp[i + 1][j - 1] : 1;
				}
				for (int k = i; k + 1 <= j; ++k)
				{
					dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j]);
				}
			}
		}
		return dp[0][n - 1];
	}

	//1243. Array Transformation
	vector<int> transformArray(vector<int>& a) {
		int n = a.size();
		bool change = true;
		while (change)
		{
			change = false;
			auto nx = a;
			for (int i = 1; i < n - 1; ++i)
			{
				if (a[i] > a[i - 1] && a[i] > a[i + 1])
				{
					nx[i] --;
					change = true;
				}
				else if (a[i] < a[i - 1] && a[i] < a[i + 1])
				{
					nx[i] ++;
					change = true;
				}
			}
			a = nx;
		}
		return a;
	}

	//1245. Tree Diameter
	int treeDiameter(vector<vector<int>>& es) {
		int n = es.size() + 1;
		vector<vector<int>> g(n);
		for (auto& e : es)
		{
			int u = e[0], v = e[1];
			g[u].push_back(v);
			g[v].push_back(u);
		}
		int st = 0;
		queue<int> q;
		q.push(st);
		vector<bool> vis(n);
		vis[st] = 1;
		int last = -1;
		while (!q.empty())
		{
			int size = q.size();
			while (size--)
			{
				auto u = q.front();
				q.pop();
				last = u;
				for (auto& v : g[u])
				{
					if (vis[v] == 0)
					{
						vis[v] = 1;
						q.push(v);
					}
				}
			}
		}
		int level = 0;
		st = last;
		q.push(st);
		for (int i = 0; i < n; ++i) vis[i] = 0;
		vis[st] = 1;
		while (!q.empty())
		{
			int size = q.size();
			level++;
			while (size--)
			{
				auto u = q.front(); q.pop();
				for (auto& v : g[u])
				{
					if (!vis[v])
					{
						vis[v] = 1;
						q.push(v);
					}
				}
			}
		}
		return level - 1;
	}

	//1247. Minimum Swaps to Make Strings Equal
	int minimumSwap(string s1, string s2) {
		int n = s1.size();
		int c0 = 0, c1 = 0;
		for (int i = 0; i < n; ++i)
		{
			if (s1[i] == s2[i]) continue;
			else if (s1[i] == 'x') c0++;
			else c1++;
		}
		int ans = 0;
		ans += c0 / 2 + c1 / 2;
		c0 %= 2;
		c1 %= 2;
		if (c0 && c1 || c0 == 0 && c1 == 0)
		{
			ans += c0 * 2;
			return ans;
		}
		return -1;
	}
	//1248. Count Number of Nice Subarrays
	int numberOfSubarrays(vector<int>& nums, int k) {
		int n = nums.size();
		vector<int> sums(n + 1);
		for (int i = 0; i < n; ++i)
		{
			sums[i + 1] = sums[i] + nums[i] % 2;
		}
		vector<int> cnt(n + 1);
		int ans = 0;
		cnt[0] = 1;
		for (int i = 0; i < n; ++i)
		{
			int val = sums[i + 1] - k;
			if (val >= 0) ans += cnt[val];
			cnt[sums[i + 1]] ++;
		}
		return ans;
	}
	//1250. Check If It Is a Good Array
	bool isGoodArray(vector<int>& nums) {
		int n = nums.size();
		for (int i = 0; i < n; ++i)
		{
			if (nums[i] == 1) return true;
		}
		if (n < 2) return false;
		int g = nums[0];
		for (int i = 1; i < n; ++i)
		{
			g = gcd(g, nums[i]);
			if (g == 1) return true;
		}
		return false;
	}

	//1249. Minimum Remove to Make Valid Parentheses
	string minRemoveToMakeValid(string s) {
		stack<int> st;
		for (auto i = 0; i < s.size(); ++i) {
			if (s[i] == '(') st.push(i);
			if (s[i] == ')') {
				if (!st.empty()) st.pop();
				else s[i] = '*';
			}
		}
		while (!st.empty()) {
			s[st.top()] = '*';
			st.pop();
		}
		s.erase(remove(s.begin(), s.end(), '*'), s.end());
		return s;
	}

	//1252. Cells with Odd Values in a Matrix
	int oddCells(int n, int m, vector<vector<int>>& a) {
		vector<int> row(n), col(m);
		for (auto& e : a)
		{
			row[e[0]] ^= 1;
			col[e[1]] ^= 1;
		}
		int cntrow = 0, cntcol = 0;
		for (int i = 0; i < n; ++i)
		{
			cntrow += row[i];
		}
		for (int j = 0; j < m; ++j)
		{
			cntcol += col[j];
		}
		return m * cntrow + n * cntcol - 2 * cntrow * cntcol;
	}

	//1253. Reconstruct a 2-Row Binary Matrix
	vector<vector<int>> reconstructMatrix(int upper, int lower, vector<int>& colsum) {
		int n = colsum.size();
		vector<vector<int>> ans(2, vector<int>(n));
		for (int i = 0; i < n; ++i)
		{
			if (colsum[i] == 2)
			{
				upper--;
				lower--;
				ans[0][i] = 1;
				ans[1][i] = 1;
			}
			else if (colsum[i] == 1)
			{
				if (upper > lower)
				{
					upper--;
					ans[0][i] = 1;
				}
				else
				{
					lower--;
					ans[1][i] = 1;
				}
			}
			else return {};
		}
		if (upper == 0 && lower == 0)
		{
			return ans;
		}
		else return {};
	}


	void dfs_1254(vector<vector<int>>& g, int x, int y, int color)
	{
		static int dr[] = { 0, 1, 0, -1 };
		static int dc[] = { 1, 0, -1, 0 };
		g[x][y] = color;
		int n = g.size(), m = g[0].size();
		for (int d = 0; d < 4; ++d)
		{
			int nx = x + dr[d], ny = y + dc[d];
			if (0 <= nx && nx < n && 0 <= ny && ny < m && g[nx][ny] == 0)
			{
				dfs_1254(g, nx, ny, color);
			}
		}
	}

	//1254. Number of Closed Islands
	int closedIsland(vector<vector<int>>& grid) {
		int n = grid.size(), m = grid[0].size();
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			if (grid[i][0] == 0)
			{
				dfs_1254(grid, i, 0, -1);
			}
			if (grid[i][m - 1] == 0)
			{
				dfs_1254(grid, i, m - 1, -1);
			}
		}
		for (int j = 1; j < m - 1; ++j)
		{
			if (grid[0][j] == 0)
			{
				dfs_1254(grid, 0, j, -1);
			}
			if (grid[n - 1][j] == 0)
			{
				dfs_1254(grid, n - 1, j, -1);
			}
		}
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				if (grid[i][j] == 0)
				{
					ans++;
					dfs_1254(grid, i, j, 3);
				}
			}
		}
		return ans;
	}
	int dfs_1255(int u, vector<string>& ws, vector<int>& cnt, vector<int>& sc)
	{
		if (u > ws.size()) return 0;
		int ans = 0;
		for (int i = u; i < ws.size(); ++i)
		{
			vector<int> tmp(26);
			int value = 0;
			for (auto& c : ws[i])
			{
				tmp[c - 'a'] ++;
				value += sc[c - 'a'];
			}
			bool valid = true;
			for (int i = 0; i < 26; ++i)
			{
				if (tmp[i] > cnt[i])
				{
					valid = false;
					break;
				}
			}
			if (valid)
			{
				for (int i = 0; i < 26; ++i)
				{
					cnt[i] -= tmp[i];
				}
				ans = max(ans, value + dfs_1255(i + 1, ws, cnt, sc));
				for (int i = 0; i < 26; ++i)
				{
					cnt[i] += tmp[i];
				}
			}
		}
		return ans;
	}
	//1255. Maximum Score Words Formed by Letters
	int maxScoreWords(vector<string>& words, vector<char>& letters, vector<int>& score) {
		vector<int> cnt(26);
		for (auto& c : letters)
		{
			cnt[c - 'a'] ++;
		}
		return dfs_1255(0, words, cnt, score);
	}

	//1256. Encode Number
	string encode_1256(int n) {
		return n > 0 ? encode_1256((n - 1) / 2) + "10"[n % 2] : "";
	}
	//string encode_1256(int num) {
	//	if (num == 0) return "";
	//	long len = 0;
	//	long cur = 0;
	//	while (cur + (1ll << len) <= num)
	//	{
	//		cur += (1 << len);
	//		len++;
	//	}
	//	int tmp = num - cur;
	//	string n;
	//	while (tmp)
	//	{
	//		n.push_back(tmp % 2 + '0');
	//		tmp /= 2;
	//	}
	//	reverse(n.begin(), n.end());
	//	return (len > n.size() ? string(len - n.size(), '0') : "") + n;
	//}

	string lca_1257(string& u, string& a, string& b, map<string, vector<string>>& g)
	{
		if (u == a || u == b) return u;
		vector<string> rets;
		for (auto& v : g[u])
		{
			auto ret = lca_1257(v, a, b, g);
			if (!ret.empty())
			{
				rets.push_back(ret);
			}
		}
		if (rets.empty()) return "";
		else if (rets.size() == 1) return rets[0];
		else if (rets.size() == 2) return u;
		return "";
	}

	//1257. Smallest Common Region
	string findSmallestRegion(vector<vector<string>>& regions, string region1, string region2) {
		map<string, vector<string>> g;
		map<string, int> indegree;
		for (auto& e : regions)
		{
			auto name = e[0];
			for (int i = 1; i < e.size(); ++i)
			{
				g[name].push_back(e[i]);
				indegree[e[i]] ++;
			}
		}
		for (auto& e : regions)
		{
			auto name = e[0];
			if (indegree.count(name)) continue;
			auto ret = lca_1257(name, region1, region2, g);
			if (!ret.empty())
			{
				return ret;
			}
		}
		return "";
	}

	//1259. Handshakes That Don't Cross
	int numberOfWays(int n) {
		vector<int> dp(n + 1);
		dp[0] = 1;
		int const mod = 1e9 + 7;
		for (int i = 2; i <= n; i += 2)
		{
			for (int l = 0; l <= i - 2; l += 2)
			{
				int r = i - 2 - l;
				dp[i] = (dp[i] + static_cast<long long>(dp[l]) * dp[r]) % mod;
			}
		}
		return dp[n];
	}

	//1260. Shift 2D Grid
	vector<vector<int>> shiftGrid(vector<vector<int>>& g, int k) {
		int n = g.size(), m = g[0].size();
		vector<int> tmp(n * m);
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				tmp[i * m + j] = g[i][j];
			}
		}
		k %= (n * m);
		for (int i = 0; i < n * m - k; ++i)
		{
			int offset = i + k;
			int x = offset / m, y = offset % m;
			g[x][y] = tmp[i];
		}
		for (int i = n * m - k; i < n * m; ++i)
		{
			int offset = i - (n * m - k);
			int x = offset / m, y = offset % m;
			g[x][y] = tmp[i];
		}
		return g;
	}

	//1262. Greatest Sum Divisible by Three
	int maxSumDivThree(vector<int>& nums) {
		int n = nums.size();
		vector<vector<int>> dp(n, vector<int>(3, -1));
		dp[0][nums[0] % 3] = nums[0];
		for (int i = 1; i < n; ++i)
		{
			dp[i] = dp[i - 1];
			dp[i][nums[i] % 3] = max(dp[i][nums[i] % 3], nums[i]);
			for (int j = 0; j < 3; ++j)
			{
				if (dp[i - 1][j] == -1) continue;
				int k = (j + nums[i] % 3) % 3;
				dp[i][k] = max(dp[i - 1][k], dp[i - 1][j] + nums[i]);
			}
		}
		return dp[n - 1][0] == -1 ? 0 : dp[n - 1][0];
	}

	//1271. Hexspeak
	string toHexspeak(string num) {
		vector<char> valid = { 'A', 'B', 'C', 'D', 'E', 'F', 'I', 'O' };
		set<char> sv(valid.begin(), valid.end());
		long long v = 0;
		for (auto& c : num) v = v * 10 + c - '0';
		string s;
		while (v)
		{
			int x = v % 16;
			if (x < 10)
				s.push_back(x + '0');
			else
				s.push_back(x - 10 + 'A');

			v /= 16;
		}
		reverse(s.begin(), s.end());
		for (auto& c : s)
		{
			if (c == '0') c = 'O';
			else if (c == '1') c = 'I';
		}
		for (auto& c : s) if (!sv.count(c)) return "ERROR";
		return s;
	}

	vector<vector<int>> minus_1272(int s, int t, int st, int ed)
	{
		if (t < st || s >= ed) return { {s, t} };
		int it = min(ed, t);
		int is = max(s, st);
		//if (is >= it) return { {s,t} };
		vector<vector<int>> ans;
		if (s < is) ans.push_back({ s, is });
		if (it < t) ans.push_back({ it, t });
		return ans;
	}

	//1272. Remove Interval
	vector<vector<int>> removeInterval(vector<vector<int>>& ins, vector<int>& rem) {
		int st = rem[0], ed = rem[1];
		int n = ins.size();
		vector<vector<int>> ans;
		for (int i = 0; i < n; ++i)
		{
			for (auto& e : minus_1272(ins[i][0], ins[i][1], st, ed))
			{
				ans.push_back(e);
			}
		}
		return ans;
	}

	class Node_1273
	{
	public:
		int val;
		vector<Node_1273*> nx;
		Node_1273() : val(-1) {}
		Node_1273(int v) : val(v) {}
	};
	pair<int, int> remove_1273(Node_1273* u)
	{
		if (!u) return { 0, 0 };
		int sum = u->val, cnt = 1;
		for (auto& e : u->nx)
		{
			auto [sum_e, cnt_e] = remove_1273(e);
			sum += sum_e;
			cnt += cnt_e;
		}
		if (sum == 0)
		{
			return { 0, 0 };
		}
		else
		{
			return { sum, cnt };
		}
	}

	//1273. Delete Tree Nodes
	int deleteTreeNodes(int nodes_, vector<int>& parent, vector<int>& value) {
		int n = nodes_;
		Node_1273* root;
		vector<Node_1273*> nodes(n), idx(n);

		for (int i = 0; i < n; ++i)
		{
			nodes[i] = new Node_1273(value[i]);
		}
		for (int i = 1; i < n; ++i)
		{
			int par = parent[i];
			nodes[par]->nx.push_back(nodes[i]);
		}
		root = nodes[0];
		return remove_1273(root).second;
	}


	class Sea {
	public:
		bool hasShips(vector<int> topRight, vector<int> bottomLeft)
		{
			return 0;
		}
	};
	int check_1274(Sea sea, int sx, int sy, int tx, int ty)
	{
		if (sx > tx || sy > ty) return 0;
		if (!sea.hasShips({ tx, ty }, { sx, sy })) return 0;

		if ((abs(tx - sx) + 1) <= 1 || (abs(ty - sy) + 1) <= 1)
		{
			int ans = 0;
			int fx = tx - sx >= 0 ? 1 : -1;
			int fy = ty - sy >= 0 ? 1 : -1;
			for (int i = sx; i <= fx * tx; i += fx)
			{
				for (int j = sy; j <= fy * ty; j += fy)
				{
					ans += sea.hasShips({ i, j }, { i, j });
				}
			}
			return ans;
		}
		int dx = tx - sx, dy = ty - sy;
		int ox = dx / 2, oy = dy / 2;
		/*
		(sx, ty)		(sx + ox, ty)			(tx, ty)

		(sx,  sy + oy)    (sx + ox, sy + oy)   (tx, sy + oy)

		(sx, sy)		(sx + ox, sy)   		(tx, sy)
		*/
		int ans = 0;
		ans += check_1274(sea, sx, sy, sx + ox - 1, sy + oy);
		ans += check_1274(sea, sx, sy + oy + 1, sx + ox, ty);

		ans += check_1274(sea, sx + ox + 1, sy + oy, tx, ty);
		ans += check_1274(sea, sx + ox, sy, tx, sy + oy - 1);
		ans += sea.hasShips({ sx + ox, sy + oy }, { sx + ox, sy + oy });
		return ans;
	}

	//1274. Number of Ships in a Rectangle
	int countShips(Sea sea, vector<int> tr, vector<int> bl) {
		int sx = bl[0], sy = bl[1];
		int tx = tr[0], ty = tr[1];
		return check_1274(sea, sx, sy, tx, ty);
	}


	bool check(vector<vector<char>>& b)
	{
		int n = 3;
		for (int i = 0; i < n; ++i)
		{
			bool win = true;
			if (b[i][0] == ' ') continue;
			for (int j = 1; j < n; ++j)
			{
				if (b[i][j] != b[i][0])
				{
					win = false;
					break;
				}
			}
			if (win) return true;
		}
		for (int j = 0; j < n; ++j)
		{
			bool win = true;
			if (b[0][j] == ' ') continue;
			for (int i = 1; i < n; ++i)
			{
				if (b[i][j] != b[0][j])
				{
					win = false;
					break;
				}
			}
			if (win) return true;

		}
		{
			if (b[0][0] != ' ')
			{
				bool win = true;

				for (int i = 1; i < n; ++i)
				{
					if (b[i][i] != b[0][0])
					{
						win = false;
					}
				}
				if (win) return true;
			}
		}
		{
			if (b[0][n - 1] != ' ')
			{
				bool win = true;
				for (int i = 1; i < n; ++i)
				{
					int j = n - 1 - i;
					if (b[i][j] != b[0][n - 1])
					{
						win = false;
						break;
					}
				}
				if (win) return true;
			}
		}
		return false;
	}
	//1275. Find Winner on a Tic Tac Toe Game
	string tictactoe(vector<vector<int>>& moves) {
		vector<vector<char>> b(3, vector<char>(3, ' '));
		int n = moves.size();
		for (int i = 0; i < n; ++i)
		{
			b[moves[i][0]][moves[i][1]] = 'A' + i % 2;
			if (check(b))
			{
				string ret;
				ret.push_back('A' + i % 2);
				return ret;
			}
			if (i >= 8)
				return "Draw";
		}
		return "Pending";
	}

	//1276. Number of Burgers with No Waste of Ingredients
	vector<int> numOfBurgers(int ts, int cs) {
		int tmp = ts - cs * 2;
		if (tmp % 2) return {};
		int x = tmp / 2;
		int y = cs - x;
		if (y < 0 || x < 0) return {};
		return { x, y };
	}

	class NumMatrix_1277 {
	public:
		vector<vector<int>> sums;
		int n, m;
		NumMatrix_1277(vector<vector<int>> a) {
			n = a.size();
			if (n == 0)
			{
				a.push_back({ 0 });
				return;
			}
			m = a[0].size();
			sums.assign(n + 1, vector<int>(m + 1));
			for (int i = 1; i <= n; ++i)
			{
				int cur = 0;
				for (int j = 1; j <= m; ++j)
				{
					cur += a[i - 1][j - 1];
					sums[i][j] = sums[i - 1][j] + cur;
				}
			}
		}

		int sumRegion(int row1, int col1, int row2, int col2) {
			return sums[row2 + 1][col2 + 1] - sums[row2 + 1][col1] - sums[row1][col2 + 1] + sums[row1][col1];
		}
	};

	//1277. Count Square Submatrices with All Ones
	int countSquares(vector<vector<int>>& a) {
		int n = a.size(), m = a[0].size();
		NumMatrix_1277 nm(a);
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				ans += a[i][j];
			}
		}
		for (int l = 2; l <= min(n, m); ++l)
		{
			for (int i = 0; i + l - 1 < n; ++i)
			{
				for (int j = 0; j + l - 1 < m; ++j)
				{
					if (nm.sumRegion(i, j, i + l - 1, j + l - 1) == l * l) ans++;
				}
			}
		}
		return ans;
	}

	//1278. Palindrome Partitioning III
	int palindromePartition(string s, int k) {
		int n = s.size();
		vector<vector<int>> change(n, vector<int>(n));
		for (int l = 1; l <= n; ++l)
		{
			for (int i = 0, j = i + l - 1; j < n; ++j, ++i)
			{
				if (l == 1) change[i][j] = 0;
				else if (l == 2)
				{
					if (s[i] == s[j]) change[i][j] = 0;
					else change[i][j] = 1;
				}
				else
				{
					change[i][j] = change[i + 1][j - 1] + (s[i] != s[j]);
				}
			}
		}
		vector<vector<int>> dp(k + 1, vector<int>(n + 1, n));
		for (int i = 0; i < n; ++i)
		{
			dp[0][i] = change[0][i];
		}
		for (int j = 0; j < k; ++j)
		{
			for (int i = 0; i < n; ++i)
			{
				for (int m = i + 1; m < n; ++m)
				{
					dp[j + 1][m] = min(dp[j + 1][m], dp[j][i] + change[i + 1][m]);
				}
			}
		}
		return dp[k - 1][n - 1];
	}
};

int main()
{
	return 0;
}