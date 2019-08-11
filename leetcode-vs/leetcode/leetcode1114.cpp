
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
using namespace std;

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
class Sem_N
{
	mutex mtx;
	condition_variable cv;
	int n;
public:
	Sem_N(int n) : n(n)
	{

	}
	void wait()
	{
		unique_lock<mutex> lck(mtx);
		cv.wait(lck, [&]() { return n > 0; });
		n--;
	}

	void notify()
	{
		unique_lock<mutex> lck(mtx);
		n++;
		cv.notify_one();
	}
};


class Sem_Q
{
	mutex mtx;
	condition_variable cv;
	queue<int> q;
	int n;
public:
	Sem_Q(int n) : n(n)
	{

	}
	int wait()
	{
		unique_lock<mutex> lck(mtx);
		//cv.wait(lck, [&]() { return n > 0; });
		cv.wait(lck, [&]() { return !q.empty(); });
		int ret = q.front();
		q.pop();
		return ret;
	}

	void notify(int v)
	{
		unique_lock<mutex> lck(mtx);
		q.push(v);
		cv.notify_one();
	}
};

//1114. Print in Order
class Foo {
public:

	Sem_N two, three;

	Foo() : two(0), three(0) {

	}


	void first(function<void()> printFirst) {

		// printFirst() outputs "first". Do not change or remove this line.
		printFirst();
		two.notify();
	}

	void second(function<void()> printSecond) {
		two.wait();

		// printSecond() outputs "second". Do not change or remove this line.
		printSecond();

		three.notify();
	}

	void third(function<void()> printThird) {
		three.wait();
		// printThird() outputs "third". Do not change or remove this line.
		printThird();
	}
};


//1115. Print FooBar Alternately
class FooBar {
private:
	int n;
	mutex mtx;
public:
	FooBar(int n) {
		this->n = n;
	}

	void foo(function<void()> printFoo) {

		for (int i = 0; i < n; i++) {
			mtx.lock();
			// printFoo() outputs "foo". Do not change or remove this line.
			printFoo();
		}
	}

	void bar(function<void()> printBar) {

		for (int i = 0; i < n; i++) {

			// printBar() outputs "bar". Do not change or remove this line.
			printBar();
			mtx.unlock();
		}
	}
};



//1116. Print Zero Even Odd
class ZeroEvenOdd {
private:
	int n, cur;
	Sem_Q modd, meven;
	mutex mtx;
	condition_variable cv;
	promise<int> pOdd, pEven, pZero;
public:
	ZeroEvenOdd(int n) : modd(1), meven(0), cur(1) {
		this->n = n;
		mtx.unlock();
	}

	// printNumber(x) outputs "x", where x is an integer.
	void zero(function<void(int)> printNumber) {
		for (int i = 0; i < n; ++i)
		{
			mtx.lock();
			printNumber(0);
			if (cur & 1)
			{
				modd.notify(cur++);
			}
			else
			{
				meven.notify(cur++);
			}
		}
	}

	void even(function<void(int)> printNumber) {
		for (int i = 0; i < n / 2; ++i)
		{
			printNumber(meven.wait());
			mtx.unlock();
		}
	}

	void odd(function<void(int)> printNumber) {
		for (int i = 0; i < (n + 1) / 2; ++i)
		{
			printNumber(modd.wait());
			mtx.unlock();
		}
	}
};

//1117. Building H2O
class H2O {
	Sem_N h, o;
	int cnt;
public:

	H2O() : h(2), o(1), cnt(0) {

	}

	void hydrogen(function<void()> releaseHydrogen) {
		h.wait();
		// releaseHydrogen() outputs "H". Do not change or remove this line.
		releaseHydrogen();
		if (++cnt == 2)
		{
			cnt = 0;
			o.notify();
		}
	}

	void oxygen(function<void()> releaseOxygen) {
		o.wait();
		// releaseOxygen() outputs "O". Do not change or remove this line.
		releaseOxygen();
		h.notify();
		h.notify();
	}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

/*
//1125. Smallest Sufficient Team
const int maxn = INT_MAX / 2;
int memo[1 << 16][60 + 1];
long long path[1 << 16][60 + 1];

class Solution {
public:
	int sub_bit(int s, int v)
	{
		return s ^ (s & v);
	}
	int dfs(int s, int u, vector<int>& a)
	{
		if (u == a.size())
		{
			return s == 0 ? 0 : maxn;
		}
		if (memo[s][u] != -1) return memo[s][u];

		int v_noSelect = dfs(s, u + 1, a);
		int v_Select = (sub_bit(s, a[u]) < s ? dfs(sub_bit(s, a[u]), u + 1, a) + 1 : maxn);

		if (v_Select < v_noSelect)
		{
			path[s][u] = path[sub_bit(s, a[u])][u + 1] | (1LL << u);
		}
		else
		{
			path[s][u] = path[s][u + 1];
		}

		// int ans = min(dfs(sub_bit(s, a[u]), u + 1, a) + 1, dfs(s, u + 1, a));
		return memo[s][u] = min(v_Select, v_noSelect);
	}

	vector<int> smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people) {
		map<string, int> idx;
		vector<int> req;
		int n = people.size();
		vector<int> a(n);
		for (int i = 0; i < n; ++i)
		{
			int u = 0;
			for (int j = 0; j < people[i].size(); ++j)
			{
				auto& s = people[i][j];
				int val = idx.size();
				if (!idx.count(s)) idx[s] = val;
				u |= (1 << idx[s]);
			}
			a[i] = u;
		}
		int tot = idx.size();
		for (int s = 0; s < (1 << idx.size()); ++s)
		{
			for (int i = 0; i < people.size(); ++i)
			{
				memo[s][i] = -1;
				path[s][i] = 0;
			}
		}
		int s = 0;
		for (auto& e : req_skills)
		{
			if (!idx.count(e))
			{
				int val = idx.size();
				idx[e] = val;
			}
			s |= (1 << idx[e]);
		}
		int ret = dfs(s, 0, a);
		long long vis = path[s][0];
		vector<int> ans;
		for (long long i = 0, mask = 1; i < people.size(); ++i, mask <<= 1)
		{
			if (mask & vis)
			{
				ans.push_back(i);
			}
		}
		return ans;
	}
};
*/

//1146. Snapshot Array
class SnapshotArray {
public:

	vector<map<int, int>> a;
	int m_snap;
	SnapshotArray(int length) : a(length), m_snap(0) {
	}

	void set(int index, int val) {
		a[index][m_snap] = val;
	}

	int snap() {
		return m_snap++;
	}

	int get(int index, int snap_id) {
		auto& val = a[index];
		auto it = val.upper_bound(snap_id);
		if (it == val.begin()) return 0;
		--it;
		return it->second;
	}
};

//1157. Online Majority Element In Subarray
class MajorityChecker {
public:
	vector<map<int, int>> cnt;
	int n;
	map<int, vector<int>> idx;
	vector<vector<int>> order;
	MajorityChecker(vector<int>& a) : n(a.size()) {
		for (int i = 0; i < n; ++i)
		{
			idx[a[i]].push_back(i);
		}
		for (auto& e : idx)
		{
			e.second.push_back(e.first);
			order.push_back(e.second);
		}
		sort(order.begin(), order.end(), [](const vector<int>& a, const vector<int>& b) {
			return a.size() < b.size(); });
	}
	int count(int v, vector<int>& a)
	{
		auto it = lower_bound(a.begin(), a.end() - 1, v);
		return it - a.begin();
	}
	int count2(int v, vector<int>& a)
	{
		auto it = upper_bound(a.begin(), a.end() - 1, v);
		return it - a.begin();
	}
	int query(int left, int right, int t) {
		int l = 0, r = order.size();
		while (l < r)
		{
			int m = (l + r) / 2;
			if (order[m].size() >= t)
			{
				r = m;
			}
			else
			{
				l = m + 1;
			}
		}
		if (l == order.size()) return -1;
		for (int i = l; i < order.size(); ++i)
		{
			int tot = count2(right, order[i]) - count(left, order[i]);
			if (tot >= t) return order[i].back();
		}
		return -1;
	}
};



class Solution
{
public:
	//1118. Number of Days in a Month
	int numberOfDays(int Y, int M) {
		//                      1   2   3   4   5   6   7   8   9  10   
		vector<int> nums = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
		if (M == 2 && (Y % 400 == 0 || Y % 4 == 0 && Y % 100)) return nums[2] + 1;
		else return nums[M];
	}
	//1119. Remove Vowels from a String
	string removeVowels(string s) {
		vector<char> ss;
		ss = { 'a', 'e', 'i', 'o', 'u' };
		string ans;
		for (char c : s)
		{
			if (find(ss.begin(), ss.end(), c) == ss.end())
				ans.push_back(c);
		}
		return ans;
	}
	pair<int, int> dfs_1120(TreeNode* u, double& ans)
	{
		if (!u) return { 0, 0 };
		auto r = dfs_1120(u->right, ans);
		auto l = dfs_1120(u->left, ans);
		if (u->left) ans = max(ans, ((double)(l.first) / l.second));
		if (u->right) ans = max(ans, ((double)(r.first) / r.second));
		int tot = l.first + r.first + u->val;
		int num = l.second + r.second + 1;
		ans = max(ans, (double)tot / num);
		return { tot, num };
	}
	//1120. Maximum Average Subtree
	double maximumAverageSubtree(TreeNode* root) {
		double ans = 0;
		dfs_1120(root, ans);
		return ans;
	}

	//1121. Divide Array Into Increasing Sequences
	bool canDivideIntoSubsequences(vector<int>& a, int k) {
		const int maxn = 1e5 + 1;
		vector<int> cnt(maxn);
		int need = 0;
		for (auto& e : a)
		{
			need = max(need, ++cnt[e]);
		}
		return static_cast<long long>(k) * need <= a.size();
	}

	//1122. Relative Sort Array
	vector<int> relativeSortArray(vector<int>& a, vector<int>& b) {
		const int maxn = 1000;
		vector<int> order(maxn + 1);
		for (auto& e : a)
		{
			if (order[e] == maxn + maxn)
			{
				order[e] = maxn + e;
			}
		}
		for (int i = 0; i < b.size(); ++i)
		{
			order[b[i]] = i;
		}
		sort(a.begin(), a.end(), [&](int i, int j) {
			return order[i] < order[j];
			});
		return a;
	}


	pair<int, TreeNode*> dfs_1123(TreeNode* u)
	{
		if (!u) return { 0, u };
		// return max(dfs(u->left), dfs(u->right)) + 1;
		auto left = dfs_1123(u->left), right = dfs_1123(u->right);
		if (left.first > right.first)
		{
			return { left.first + 1, left.second };
		}
		else if (left.first == right.first)
		{
			return { left.first + 1, u };
		}
		else
		{
			return { right.first + 1, right.second };
		}
	}
	//1123. Lowest Common Ancestor of Deepest Leaves
	TreeNode* lcaDeepestLeaves(TreeNode* root) {
		auto ret = dfs_1123(root);
		return ret.second;
	}

	//1124. Longest Well-Performing Interval
	int longestWPI(vector<int>& hours) {
		unordered_map<int, int> m;
		int n = hours.size();
		int rolling = 0, res = 0;
		for (int i = 0; i < n; i++) {
			rolling += hours[i] > 8 ? 1 : -1;
			if (rolling > 0) {
				res = i + 1;
			}
			else {
				if (m.count(rolling - 1))
					res = max(res, i - m[rolling - 1]);
			}

			if (!m.count(rolling))
				m[rolling] = i;
		}
		return res;
	}
	int dfs_1130(int i, int j, vector<int>& a, vector<vector<int>>& g, vector<vector<int>>& memo)
	{
		if (i >= j) return 0;
		if (memo[i][j] != -1) return memo[i][j];
		int ans = INT_MAX;
		for (int k = i; k + 1 <= j; ++k)
		{
			ans = min(ans, dfs_1130(i, k, a, g, memo) + dfs_1130(k + 1, j, a, g, memo) + g[i][k] * g[k + 1][j]);
		}
		memo[i][j] = ans;
		return ans;
	}
	//1130. Minimum Cost Tree From Leaf Values
	int mctFromLeafValues(vector<int>& a) {
		int n = a.size();
		vector<vector<int>> g(n, vector<int>(n));
		for (int i = 0; i < n; ++i)
		{
			g[i][i] = a[i];
			for (int j = i + 1; j < n; ++j)
			{
				g[i][j] = max(a[j], g[i][j - 1]);
			}
		}
		vector<vector<int>> memo(n, vector<int>(n, -1));
		return dfs_1130(0, n - 1, a, g, memo);
	}

	//1131. Maximum of Absolute Value Expression
	int maxAbsValExpr(vector<int>& a, vector<int>& b) {
		int n = a.size();

		vector<function<int(int)>> funs;
		funs.push_back([&](int i) { return a[i] + b[i] + i; });
		funs.push_back([&](int i) { return -a[i] - b[i] + i; });
		funs.push_back([&](int i) { return -a[i] + b[i] + i; });
		funs.push_back([&](int i) { return a[i] - b[i] + i; });

		auto get_value = [&](int i, int j)
		{
			return abs(a[i] - a[j]) + abs(b[i] - b[j]) + abs(i - j);
		};
		vector<int> pre(8, 0);
		int ans = 0;

		for (int i = 1; i < n; ++i)
		{
			for (int j = 0; j < 8; ++j)
			{
				ans = max(ans, get_value(i, pre[j]));
			}
			int f = 1;
			for (int j = 0; j < 8; ++j)
			{
				if (j % 2) f = -1;
				else f = 1;

				if (f * funs[j / 2](i) > f * funs[j / 2](pre[j]))
				{
					pre[j] = i;
				}
			}
		}
		return ans;
	}
	
	//1128. Number of Equivalent Domino Pairs
	int numEquivDominoPairs(vector<vector<int>>& a) {
		int n = a.size();
		map<pair<int, int>, int> cnt;
		for (auto& e : a)
		{
			if (e[0] > e[1])
			{
				swap(e[0], e[1]);
			}
			cnt[{e[0], e[1]}] ++;
		}
		int ans = 0;
		for (auto& e : cnt)
		{
			int u = e.second;
			if (u > 1)
			{
				ans += u * (u - 1) / 2;
			}
		}
		return ans;
	}


	struct Node_1129
	{
		int u, prev;
		bool operator < (const Node_1129& v) const
		{
			return u == v.u ? prev < v.prev : u < v.u;
		}
	};
	//1129. Shortest Path with Alternating Colors
	vector<int> shortestAlternatingPaths(int n, vector<vector<int>>& red, vector<vector<int>>& blue) {
		vector<vector<pair<int, int>>> g(n);
		for (auto& e : red)
		{
			int u = e[0], v = e[1];
			g[u].emplace_back(v, 0);
		}
		for (auto& e : blue)
		{
			int u = e[0], v = e[1];
			g[u].emplace_back(v, 1);
		}
		int const maxn = INT_MAX / 2;
		vector<int> dist(n, maxn);
		dist[0] = 0;
		queue<Node_1129> q;
		const int N = 100 + 1;
		vector<vector<int>> memo(N, vector<int>(2, maxn));
		memo[0][0] = 0;
		memo[0][1] = 0;
		q.push({ 0, 0 });
		q.push({ 0, 1 });
		while (!q.empty())
		{
			auto x = q.front();
			q.pop();
			int u = x.u, prev = x.prev;
			for (auto& e : g[u])
			{
				int v = e.first, color = e.second;
				if (prev != color)
				{
					if (memo[v][color] > memo[u][prev] + 1)
					{
						memo[v][color] = memo[u][prev] + 1;
						dist[v] = memo[v][color];
						q.push({ v, color });
					}
				}
			}
		}
		for (int i = 0; i < n; ++i)
		{
			dist[i] = min(dist[i], memo[i][0]);
			dist[i] = min(dist[i], memo[i][1]);
		}
		for (auto& e : dist)
		{
			if (e == maxn) e = -1;
		}
		return dist;
	}

	//1133. Largest Unique Number
	int largestUniqueNumber(vector<int>& a) {
		vector<int> cnt(1001);
		for (auto& e : a) cnt[e] ++;
		for (int i = 1000; i >= 0; --i)
		{
			if (cnt[i] == 1) return i;
		}
		return -1;
	}

	//1134. Armstrong Number
	bool isArmstrong(int n) {
		int val = n;
		long long ans = 0;
		int len = log(n) / log(10) + 1;
		while (n)
		{
			ans += pow(n % 10, len);
			n /= 10;
		}
		return ans == val;
	}

	//1135.	Connecting Cities With Minimum Cost
	int minimumCost(int N, vector<vector<int>>& a) {
		sort(a.begin(), a.end(), [](vector<int>& a, vector<int>& b) {
			return a[2] < b[2];
			});

		DSU d(N + 1);

		vector<int> vis(N + 1);
		int ans = 0;
		for (auto& e : a)
		{
			int u = e[0], v = e[1], cost = e[2];
			int fu = d.find(u), fv = d.find(v);
			if (fu != fv)
			{
				ans += cost;
				d.Union(u, v);
			}
		}
		int tot = 0;
		for (int i = 1; i <= N; ++i)
		{
			tot += d.find(i) == i;
		}
		if (tot > 1) return -1;
		return ans;
	}


	//1136. Parallel Courses
	int minimumSemesters(int n, vector<vector<int>>& a) {
		int ans = 0;
		vector<int> in(n + 1);
		vector<vector<int>> g(n + 1);
		for (auto& e : a)
		{
			int u = e[0], v = e[1];
			in[v]++;
			g[u].push_back(v);
		}
		queue<int> q;
		vector<int> vis(n + 1);
		for (int i = 1; i <= n; ++i)
		{
			if (in[i] == 0)
			{
				q.push(i);
				vis[i] = 1;
			}
		}
		while (!q.empty())
		{
			int size = q.size();
			ans++;
			while (size--)
			{
				auto u = q.front();
				q.pop();
				for (auto v : g[u])
				{
					if (vis[v]) continue;
					if (--in[v] == 0)
						q.push(v);
				}
			}
		}
		for (int i = 1; i <= n; ++i) if (in[i]) return -1;
		return ans;
	}


	//1137. N-th Tribonacci Number
	int tribonacci(int n) {
		vector<int> a(40);
		a[0] = 0;
		a[1] = 1;
		a[2] = 1;
		for (int i = 3; i <= n; ++i)
			a[i] = a[i - 1] + a[i - 2] + a[i - 3];
		return a[n];
	}

	//1138. Alphabet Board Path
	string move_h(int y, int v)
	{
		string ans;
		while (y < v)
		{
			ans += "R";
			y++;
		}
		while (y > v)
		{
			ans += "L";
			y--;
		}
		return ans;
	}

	string move_v(int x, int u)
	{
		string ans;
		while (x < u)
		{
			ans += "D";
			x++;
		}
		while (x > u)
		{
			ans += "U";
			x--;
		}
		return ans;
	}
	string bfs_1138(char from, char to, vector<pair<int, int>>& g)
	{

		if (from == to) return "!";
		int x = g[from - 'a'].first, y = g[from - 'a'].second;
		int u = g[to - 'a'].first, v = g[to - 'a'].second;
		int dy = v - y;
		int dx = u - x;

		string ans;
		if (to == 'z')
		{
			ans += move_h(y, v);
			ans += move_v(x, u);
		}
		else if (from == 'z')
		{
			ans += move_v(x, u);
			ans += move_h(y, v);
		}
		else
		{
			ans += move_v(x, u);
			ans += move_h(y, v);
		}
		ans += "!";
		return ans;
	}

	//1138. Alphabet Board Path
	string alphabetBoardPath(string s) {
		vector<string> b = { "abcde", "fghij", "klmno", "pqrst", "uvwxy", "z" };
		vector<pair<int, int>> g(26);

		for (int i = 0; i < b.size(); ++i)
		{
			for (int j = 0; j < b[i].size(); ++j)
			{
				g[b[i][j] - 'a'] = { i, j };
			}
		}
		string ans;
		char pre = 'a';
		for (int i = 0; i < s.size(); ++i)
		{
			ans += bfs_1138(pre, s[i], g);
			pre = s[i];
		}
		return ans;
	}

	//1139. Largest 1-Bordered Square
	int largest1BorderedSquare(vector<vector<int>>& a) {
		int n = a.size(), m = a[0].size();
		int ans = 0;

		vector<vector<int>> sums_h(n, vector<int>(m)), sums_v(n, vector<int>(m));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				sums_v[i][j] = (i - 1 >= 0 ? sums_v[i - 1][j] : 0) + a[i][j];
			}
		}

		for (int j = 0; j < m; ++j)
		{
			for (int i = 0; i < n; ++i)
			{
				sums_h[i][j] = (j - 1 >= 0 ? sums_h[i][j - 1] : 0) + a[i][j];
			}
		}

		for (int l = min(n, m); l >= 1; --l)
		{
			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < m; ++j)
				{
					if (i + l - 1 >= n || j + l - 1 >= m) continue;
					if (sums_h[i][j + l - 1] - (j - 1 >= 0 ? sums_h[i][j - 1] : 0) == l &&
						sums_h[i + l - 1][j + l - 1] - (j - 1 >= 0 ? sums_h[i + l - 1][j - 1] : 0) == l &&
						
						sums_v[i + l - 1][j] - (i - 1 >= 0 ? sums_v[i - 1][j] : 0) == l &&
						sums_v[i + l - 1][j + l - 1] - (i - 1 >= 0 ? sums_v[i - 1][j + l - 1] : 0) == l
						)
					{
						return l * l;
					}
				}
			}
		}
		return 0;
	}

	int dfs_1140(int u, int m, vector<int>& a, vector<vector<int>>& memo)
	{
		if (u >= a.size()) return 0;
		if (memo[u][m] != -1) return memo[u][m];
		int ans = INT_MIN;
		int sum = 0;
		for (int i = 1; i <= m + m && u + i - 1 < a.size(); ++i)
		{
			sum += a[u + i - 1];
			ans = max(ans, sum - dfs_1140(u + i, max(i, m), a, memo));
		}
		return memo[u][m] = ans;
	}
	//1140. Stone Game II
	int stoneGameII(vector<int>& piles) {
		int n = piles.size();
		vector<vector<int>> memo(n, vector<int>(n + n, -1));
		int tot = 0;
		for (int e : piles) tot += e;
		int diff = dfs_1140(0, 1, piles, memo);
		return (tot + diff) / 2;
	}

	int offset_1144(int s, vector<int>& a)
	{
		int ans = 0;
		int n = a.size();
		for (int i = s; i < n; i += 2)
		{
			int v = INT_MAX;
			if (i + 1 < n) v = a[i + 1];
			if (i - 1 >= 0) v = min(v, a[i - 1]);
			if (a[i] < v) continue;
			ans += a[i] - v + 1;
		}
		return ans;
	}

	//1144. Decrease Elements To Make Array Zigzag
	int movesToMakeZigzag(vector<int>& a) {
		return min(offset_1144(0, a), offset_1144(1, a));
	}

	int dfs_1145(TreeNode* u, int x, int &fa,int &left, int &right, int n)
	{
		if (!u) return 0;
		if (u->val == x)
		{
			int l = dfs_1145(u->left, x, fa, left, right, n);
			int r = dfs_1145(u->right, x, fa, left, right, n);
			left = l;
			right = r;
			fa = n - 1 - l - r;
			return l + r + 1;
		}
		else
		{
			return dfs_1145(u->left, x, fa, left, right, n) + dfs_1145(u->right, x, fa, left, right, n) + 1;
		}
	}
	//1145. Binary Tree Coloring Game
	bool btreeGameWinningMove(TreeNode* root, int n, int x) {
		vector<vector<int>> g(n + 1);
		int fa = 0, left = 0, right = 0;
		dfs_1145(root, x, fa, left, right, n);
		vector<int> a = { fa, left, right };
		for (int i = 0; i < 3; ++i)
		{
			if (a[i] > n - a[i])
			{
				return true;
			}
		}
		return false;
	}


	int dfs_1147(int i, int j, string& s, vector<vector<int>> &memo)
	{
		if (i == j) return 1;
		if (i > j) return 0;
		if (memo[i][j] != -1) return memo[i][j];
		int ans = 1;
		for (int l = 1; i + l - 1 < j - l + 1; ++l)
		{
			bool same = true;
			for (int k = 0; same && k < l; ++k)
			{
				if (s[i + k] != s[j - l + 1 + k])
					same = false;
			}
			if (same)
			{
				ans = max(ans, dfs_1147(i + l, j - l, s, memo) + 2);
			}
		}
		return memo[i][j] = ans;
	}

	//1147. Longest Chunked Palindrome Decomposition
	int longestDecomposition(string s) {
		vector<vector<int>> memo(s.size(), vector<int>(s.size(), -1));
		return dfs_1147(0, s.size() - 1, s, memo);
	}

	//1150. Check If a Number Is Majority Element in a Sorted Array
	bool isMajorityElement(vector<int>& nums, int target) {
		int cnt = 0;
		for (auto& e : nums)
		{
			if (e == target)
			{
				if (++cnt > nums.size() / 2) return true;
			}
		}
		return false;
	}

	//1151. Minimum Swaps to Group All 1's Together
	int minSwaps(vector<int>& a) {
		int n = a.size();
		int cnt = 0;
		for (int& e : a) cnt += e;
		int ans = INT_MAX;
		if (cnt == 0) return 0;
		int u = 0;
		for (int i = 0; i < n; ++i)
		{
			u += a[i];
			if (i >= cnt - 1)
			{
				ans = min(ans, cnt - u);
				u -= a[i - cnt + 1];
			}
		}
		return ans;
	}

	//1153. String Transforms Into Another String
	bool canConvert(string s, string p) {
		if (s == p) return true;
		vector<set<int>> cnt(26);
		int n = s.size();
		for (int i = 0; i < n; ++i)
		{
			cnt[s[i] - 'a'].insert(p[i] - 'a');
		}
		for (int i = 0; i < 26; ++i)
		{
			if (cnt[i].size() > 1) return false;
		}

		int use = 0;
		vector<bool> visp(26);
		for (auto& e : p) visp[e - 'a'] = 1;
		for (int i = 0; i < 26; ++i)
		{
			use += visp[i];
		}

		if (use == 26) return false;
		return true;
	}

	struct Node_1152
	{
	public:
		string name, page;
		int time;
		bool operator < (const Node_1152& v) const
		{
			return name == v.name ? time < v.time : name < v.name;
		}
	};

	int count_1152(vector<string>& p, vector<Node_1152>& a)
	{
		int n = a.size();
		int ans = 0;
		for (int i = 0; i < n; )
		{
			int j = i;
			string name = a[i].name;
			int u = 0;
			for (; j < n && a[j].name == name; ++j)
			{
				if (u < p.size() && p[u] == a[j].page)
				{
					u++;
				}
			}
			if (u == p.size())
				ans++;
			i = j;
		}
		return ans;
	}
	//1152. Analyze User Website Visit Pattern
	vector<string> mostVisitedPattern(vector<string>& username, vector<int>& timestamp, vector<string>& website) {
		int n = username.size();
		vector<Node_1152> a(n);
		for (int i = 0; i < n; ++i)
		{
			a[i] = { username[i], website[i], timestamp[i] };
		}
		sort(a.begin(), a.end());
		int cnt = 0;
		vector<string> ans;
		vector<string> pages;
		for (auto& e : website)
			pages.push_back(e);
		sort(pages.begin(), pages.end());
		pages.erase(unique(pages.begin(), pages.end()), pages.end());
		n = pages.size();
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				for (int k = 0; k < n; ++k)
				{
					vector<string> tmp = { pages[i], pages[j], pages[k] };
					int cur = count_1152(tmp, a);
					if (cnt < cur)
					{
						cnt = cur;
						ans = tmp;
					}
				}
			}
		}
		return ans;
	}


	//1154. Day of the Year
	int dayOfYear(string s) {
		vector<int> nums = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
		for (int i = 1; i < nums.size(); ++i)
		{
			nums[i] += nums[i - 1];
		}
		vector<int> date;
		int u = 0;
		for (int i = 0; i < s.size(); ++i)
		{
			if (s[i] == '-')
			{
				date.push_back(u);
				u = 0;
				continue;
			}
			u = u * 10 + (s[i] - '0');
		}
		date.push_back(u);
		int y = date[0], m = date[1], d = date[2];
		return nums[m - 1] + (2 < m && (y % 400 == 0 || y % 4 == 0 && y % 100) ? 1 : 0) + d;
	}

	//1155. Number of Dice Rolls With Target Sum
	int numRollsToTarget(int d, int f, int t) {
		int const mod = 1e9 + 7;
		vector<int> dp(t + 1), pre(t + 1);
		pre[0] = 1;
		for (int i = 1; i <= d; ++i)
		{
			fill(dp.begin(), dp.end(), 0);
			for (int s = 0; s <= t; ++s)
			{
				for (int v = 1; v <= f && v + s <= t; ++v)
				{
					dp[v + s] = (dp[v + s] + pre[s]) % mod;
				}
			}
			swap(pre, dp);
		}
		return pre[t];
	}

	//1156. Swap For Longest Repeated Character Substring
	int maxRepOpt1(string s) {
		int n = s.size(), ret = 0;
		for (int k = 0; k < 26; ++k) {
			vector<pair<int, int>> a;
			for (int i = 0, j; i < n; i = j) {
				for (; i < n && s[i] != 'a' + k; ++i);
				if (i == n) break;
				for (j = i + 1; j < n && s[j] == s[i]; ++j);
				a.push_back({ i, j });
			}
			if (a.size() > 1) {
				for (int i = 1; i < a.size(); ++i) {
					if (a[i - 1].second + 1 != a[i].first) continue;
					ret = max(ret, a[i - 1].second - a[i - 1].first + a[i].second - a[i].first + (a.size() >= 3));
				}
			}
			for (int i = 0; i < a.size(); ++i) {
				ret = max(ret, a[i].second - a[i].first + (a.size() >= 2));
			}
		}
		return ret;
	}
};

int main()
{
	Solution sol;
	vector<vector<int>> a;
	a = { {1,1,1},{1,0,1},{1,1,1} };
	cout << sol.largest1BorderedSquare(a) << endl;
	return 0;
}