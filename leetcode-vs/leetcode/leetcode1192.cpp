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

	ll check(ll n, ll a, ll b, ll c)
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
			ll tot = check(m, a, b, c);
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
	vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems) {
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
};

int main()
{
	return 0;
}