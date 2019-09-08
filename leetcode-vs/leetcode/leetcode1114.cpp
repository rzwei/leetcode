
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
	bool merge(int x, int y) {
		int xr = find(x), yr = find(y);
		if (xr == yr) return false;

		if (rank[xr] < rank[yr]) {
			int tmp = yr;
			yr = xr;
			xr = tmp;
		}
		if (rank[xr] == rank[yr])
			rank[xr]++;

		parent[yr] = xr;
		sz[xr] += sz[yr];
		return true;
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

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
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



struct Node_1166
{
	Node_1166() {
		value = 0;
	};
	Node_1166(int v)
	{
		value = v;
	}
	int value;
	map<string, Node_1166*> next;
};

//1166. Design File System
class FileSystem {
public:
	vector<string> split(string& s)
	{
		vector<string> ans;
		string u;
		for (int i = 1; i < s.size(); ++i)
		{
			char c = s[i];
			if (c == '/')
			{
				ans.push_back(u);
				u.clear();
			}
			else
			{
				u.push_back(c);
			}
		}
		if (!u.empty()) ans.push_back(u);
		return ans;
	}

	Node_1166* root;
	FileSystem() {
		root = new Node_1166();
	}

	bool create(string path, int value) {
		if (path.empty() || path == "/") return false;
		auto tokens = split(path);
		auto u = root;
		for (int i = 0; i < tokens.size() - 1; ++i)
		{
			if (!u->next.count(tokens[i]))
			{
				return false;
			}
			else
			{
				u = u->next[tokens[i]];
			}
		}
		if (u->next.count(tokens.back()))
			return false;
		u->next[tokens.back()] = new Node_1166(value);
		return true;
	}

	int get(string path) {
		if (path.empty() || path == "/") return -1;
		auto ts = split(path);
		auto u = root;
		for (auto& e : ts)
		{
			if (!u->next.count(e)) return -1;
			u = u->next[e];
		}
		if (u == nullptr) return -1;
		return u->value;
	}
};

//1172. Dinner Plate Stacks
class DinnerPlates {
public:
	set<int> p_push;
	map<int, stack<int>> stk;
	int size;
	int pos;
	DinnerPlates(int capacity) : size(capacity), pos(0) {
		p_push.insert(pos);
	}

	int getPush()
	{
		if (p_push.empty()) return -1;
		return *p_push.begin();
	}
	void push(int val) {
		int idx = getPush();
		stk[idx].push(val);
		if (stk[idx].size() == size)
		{
			p_push.erase(idx);
			if (p_push.empty() && idx == pos)
			{
				p_push.insert(++pos);
			}
		}
	}

	int pop() {
		if (stk.empty()) return -1;
		for (auto it = prev(stk.end()); ; --it)
		{
			auto& s = it->second;
			auto idx = it->first;
			if (!s.empty())
			{
				auto ret = s.top();
				s.pop();
				if (s.empty())
				{
					stk.erase(idx);
				}
				p_push.insert(idx);
				return ret;
			}
		}
		return -1;
	}

	int popAtStack(int idx) {
		if (!stk.count(idx)) return -1;
		auto& s = stk[idx];
		auto ret = s.top();
		s.pop();
		p_push.insert(idx);
		if (s.empty())
			stk.erase(idx);
		return ret;
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

	bool check_1160(string& s, vector<int>& cnt)
	{
		vector<int> cnt2(26);
		for (auto& e : s) cnt2[e - 'a'] ++;
		for (int i = 0; i < 26; ++i) if (cnt2[i] > cnt[i]) return false;
		return true;
	}

	//1160. Find Words That Can Be Formed by Characters
	int countCharacters(vector<string>& words, string chars) {
		vector<int> cnt(26);
		for (auto& e : chars) cnt[e - 'a'] ++;
		int ans = 0;
		for (auto& e : words)
		{
			if (check_1160(e, cnt))
				ans += e.size();
		}
		return ans;
	}


	//1161. Maximum Level Sum of a Binary Tree
	int maxLevelSum(TreeNode* root) {
		int ans = 0;
		int ansi = 1;
		queue<TreeNode*> q;
		q.push(root);
		int curi = 1;
		while (!q.empty())
		{
			int size = q.size();
			int cur = 0;
			while (size--)
			{
				auto u = q.front();
				q.pop();
				cur += u->val;
				if (u->left) q.push(u->left);
				if (u->right) q.push(u->right);
			}
			if (cur > ans)
			{
				ans = cur;
				ansi = curi;
			}
			curi++;
		}
		return ansi;
	}


	//1163. Last Substring in Lexicographical Order
	string lastSubstring(string s) {
		if (s.length() < 2) return s;

		int maxindex = 0;
		int i = 1;
		int size = s.length();

		while (i < s.length()) {
			if (s[i] > s[maxindex]) {
				maxindex = i;
			}
			else if (s[i] == s[maxindex]) {
				int curri = i;
				int j = maxindex;
				while (i < size && s[j] == s[i] && j < curri) {
					i++;
					j++;
				}
				if (i >= size || j >= curri || s[j] > s[i])
					continue;
				maxindex = curri;
				continue;
			}
			i++;
		}
		return s.substr(maxindex);
	}

	//1162. As Far from Land as Possible
	int maxDistance(vector<vector<int>>& g) {
		int n = g.size();
		queue<pair<int, int>> q;
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (g[i][j] == 1)
				{
					q.push({ i, j });
				}
			}
		}
		int dr[] = { 0, 1, 0, -1 };
		int dc[] = { 1, 0, -1, 0 };
		int cnt = n * n - q.size();
		if (cnt == 0 || cnt == n * n) return -1;
		while (!q.empty())
		{
			int size = q.size();
			while (size--)
			{
				auto x = q.front().first, y = q.front().second;
				q.pop();
				for (int d = 0; d < 4; ++d)
				{
					int nx = x + dr[d], ny = y + dc[d];
					if (0 <= nx && nx < n && 0 <= ny && ny < n && g[nx][ny] == 0)
					{
						g[nx][ny] = 1;
						q.push({ nx, ny });
					}
				}
			}
			ans++;
		}
		return ans - 1;
	}

	//1165. Single-Row Keyboard
	int calculateTime(string k, string w) {
		vector<int> m(26);
		for (int i = 0; i < 26; ++i)
		{
			m[k[i] - 'a'] = i;
		}
		int ans = 0;
		int last = 0;
		for (auto& c : w)
		{
			ans += abs(m[c - 'a'] - last);
			last = m[c - 'a'];
		}
		return ans;
	}

	//1167. Minimum Cost to Connect Sticks
	int connectSticks(vector<int>& a) {
		if (a.size() == 1) return 0;
		if (a.size() == 2) return a[0] + a[1];

		priority_queue<int, vector<int>, greater<int>> pq(a.begin(), a.end());
		int ans = 0;
		while (pq.size() > 1)
		{
			auto u = pq.top();
			pq.pop();
			auto v = pq.top();
			pq.pop();
			ans += u + v;
			pq.push(u + v);
		}
		return ans;
	}

	//1168. Optimize Water Distribution in a Village
	int minCostToSupplyWater(int n, vector<int>& cost, vector<vector<int>>& a) {
		int m = a.size();
		a.resize(n + m);
		for (int i = 0; i < n; ++i)
		{
			a[m + i] = { n + 1, i + 1, cost[i] };
		}
		sort(a.begin(), a.end(), [](const vector<int>& a, const vector<int>& b) {
			return a[2] < b[2];
			});
		DSU dsu(n + 2);
		int ans = 0;
		for (int i = 0; i < n + m; ++i)
		{
			int u = a[i][0], v = a[i][1], c = a[i][2];
			if (dsu.merge(u, v))
			{
				ans += c;
			}
		}
		return ans;
	}

	//1171. Remove Zero Sum Consecutive Nodes from Linked List
	ListNode* removeZeroSumSublists(ListNode* head) {
		ListNode* dummy = new ListNode(0);
		dummy->next = head;
		auto p = dummy;
		auto q = head;
		map<int, ListNode*> prev;
		int sum = 0;
		while (p)
		{
			sum += p->val;
			if (prev.count(sum))
			{
				auto u = prev[sum]->next;
				while (u && u != p)
				{
					sum += u->val;
					prev.erase(sum);
					u = u->next;
				}
				sum += p->val;
				prev[sum]->next = p->next;
			}
			else
			{
				prev[sum] = p;
			}
			p = p->next;
		}
		return dummy->next;
	}
	int solve_1170(string& s)
	{
		vector<int> cnt(26);
		for (auto& c : s) cnt[c - 'a'] ++;
		for (int i = 0; i < 26; ++i) if (cnt[i]) return cnt[i];
		return 0;
	}

	//1170. Compare Strings by Frequency of the Smallest Character
	vector<int> numSmallerByFrequency(vector<string>& a, vector<string>& b) {
		int n = a.size();
		int m = b.size();
		vector<int> word(m);
		for (int i = 0; i < m; ++i)
		{
			word[i] = solve_1170(b[i]);
		}
		vector<int> q(n);
		for (int i = 0; i < n; ++i)
		{
			q[i] = solve_1170(a[i]);
		}
		sort(word.begin(), word.end());
		vector<int> ans(n);
		for (int i = 0; i < n; ++i)
		{
			ans[i] = m - (upper_bound(word.begin(), word.end(), q[i]) - word.begin());
		}
		return ans;
	}

	vector<string> split_1169(string& s)
	{
		vector<string> ans;
		string u;
		for (auto& c : s)
		{
			if (c == ',')
			{
				ans.push_back(u);
				u.clear();
			}
			else
			{
				u.push_back(c);
			}
		}
		if (!u.empty())
			ans.push_back(u);
		return ans;
	}

	//1169. Invalid Transactions
	vector<string> invalidTransactions(vector<string>& a) {
		map<string, vector<tuple<int, string, int>>> t;
		for (auto& e : a)
		{
			auto ts = split_1169(e);
			string name = ts[0];
			int time = stoi(ts[1]);
			int money = stoi(ts[2]);
			string city = ts[3];
			t[name].push_back({ time, city, money });
		}
		set<string> ans_set;
		vector<string> ans;
		for (auto& e : t)
		{
			string name = e.first;
			sort(e.second.begin(), e.second.end());
			auto& b = e.second;
			for (int i = 0; i < b.size(); ++i)
			{
				if (get<2>(b[i]) > 1000)
				{
					ans_set.insert(
						name + "," + to_string(get<0>(b[i])) + "," + to_string(get<2>(b[i])) + "," + get<1>(b[i])
					);
				}
				for (int j = i - 1; j >= 0; --j)
				{
					if (i == j) continue;
					if (get<1>(b[i]) == get<1>(b[j])) continue;
					if (abs(get<0>(b[i]) - get<0>(b[j])) <= 60)
					{
						ans_set.insert(
							name + "," + to_string(get<0>(b[i])) + "," + to_string(get<2>(b[i])) + "," + get<1>(b[i])
						);
						ans_set.insert(
							name + "," + to_string(get<0>(b[j])) + "," + to_string(get<2>(b[j])) + "," + get<1>(b[j])
						);
					}
					else
					{
						break;
					}
				}
			}
		}
		for (auto& e : ans_set)
			ans.push_back(e);
		return ans;
	}

	//1175. Prime Arrangements
	int numPrimeArrangements(int n) {
		if (n == 1) return 1;
		vector<bool> isprime(n + 1, 1);
		isprime[0] = 0;
		isprime[1] = 0;
		int tot = 0;
		for (int i = 2; i <= n; ++i)
		{
			if (isprime[i])
			{
				for (int j = i + i; j <= n; j += i)
				{
					isprime[j] = 0;
				}
				tot++;
			}
		}

		int const mod = 1e9 + 7;
		long long ans = 1;
		for (int i = 1; i <= tot; ++i)
		{
			ans = (ans * i) % mod;
		}
		for (int i = 1; i <= n - tot; ++i)
		{
			ans = (ans * i) % mod;
		}
		return ans;
	}

	//1176. Diet Plan Performance
	int dietPlanPerformance(vector<int>& a, int k, int lower, int upper) {
		int n = a.size();
		int ans = 0;
		int u = 0;
		for (int i = 0; i < n; ++i)
		{
			u += a[i];
			if (i >= k - 1)
			{
				if (u < lower) ans--;
				else if (u > upper) ans++;
				u -= a[i - k + 1];
			}
		}
		return ans;
	}

	//1177. Can Make Palindrome from Substring
	vector<bool> canMakePaliQueries(string s, vector<vector<int>>& q) {
		int n = s.size();
		vector<vector<int>> sum(n + 1, vector<int>(26));
		for (int i = 1; i <= n; ++i)
		{
			sum[i] = sum[i - 1];
			sum[i][s[i - 1] - 'a'] ++;
		}
		vector<bool> ans(q.size());
		for (int i = 0; i < q.size(); ++i)
		{
			int l = q[i][0], r = q[i][1], x = q[i][2];
			vector<int> cc(2);
			for (int j = 0; j < 26; ++j)
			{
				int v = sum[r + 1][j] - sum[l][j];
				cc[v % 2] ++;
			}
			int odd = cc[1];
			if (l == r)
			{
				ans[i] = true;
			}
			else
			{
				if ((r - l + 1) % 2)
				{
					ans[i] = odd - 1 <= 2 * x;
				}
				else
				{
					ans[i] = odd <= 2 * x;
				}
			}
		}
		return ans;
	}

	//1178. Number of Valid Words for Each Puzzle
	vector<int> findNumOfValidWords(vector<string>& ws, vector<string>& ps)
	{
		map<int, int> wmap;
		for (auto& e : ws)
		{
			int u = 0;
			for (auto& c : e)
			{
				u |= (1 << (c - 'a'));
			}
			wmap[u] ++;
		}
		int n = ps.size();
		vector<int> ans(n);
		for (int i = 0; i < n; ++i)
		{
			string& s = ps[i];
			char st = s[0];
			int u = 0;
			for (auto& c : s)
			{
				u |= (1 << (c - 'a'));
			}
			for (int x = u; x; x = (x - 1) & u)
			{
				if (x & (1 << (st - 'a')))
				{
					ans[i] += wmap[x];
				}
			}
		}
		return ans;
	}
	//1180. Count Substrings with Only One Distinct Letter
	int countLetters(string s) {
		int n = s.size();
		int ans = 0;
		for (int i = 0; i < n; )
		{
			int cnt = 0;
			int val = s[i];
			while (i < n && s[i] == val)
			{
				cnt++;
				ans += cnt;
				i++;
			}
		}
		return ans;
	}

	string get_first(string& s)
	{
		string ans;
		for (int i = 0; i < s.size() && s[i] != ' '; ++i)
		{
			ans.push_back(s[i]);
		}
		return ans;
	}
	string get_last(string& s)
	{
		string ans;
		int n = s.size();
		for (int i = n - 1; i >= 0 && s[i] != ' '; --i)
		{
			ans = s[i] + ans;
		}
		return ans;
	}

	bool judge(string& a, string& b)
	{
		auto left = get_last(a);
		auto right = get_first(b);
		if (left == right)
		{
			return true;
		}
		return false;
	}

	string connect(string& a, string& b)
	{
		auto x = get_last(a);
		return a + b.substr(x.size());
	}

	//1181. Before and After Puzzle
	vector<string> beforeAndAfterPuzzles(vector<string>& a) {
		sort(a.begin(), a.end());
		vector<string> ans;
		int n = a.size();
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (i != j && judge(a[i], a[j]))
				{
					ans.push_back(connect(a[i], a[j]));
				}
			}
		}
		sort(ans.begin(), ans.end());
		ans.erase(unique(ans.begin(), ans.end()), ans.end());
		return ans;
	}
	//1182. Shortest Distance to Target Color
	vector<int> shortestDistanceColor(vector<int>& a, vector<vector<int>>& q) {
		int n = a.size();
		vector<vector<int>> left(n, vector<int>(4, INT_MAX));
		vector<vector<int>> right(n, vector<int>(4, INT_MAX));
		int m = q.size();
		vector<int> ans(m);
		left[0][a[0]] = 0;
		for (int i = 1; i < n; ++i)
		{
			left[i] = left[i - 1];
			left[i][a[i]] = i;
		}
		right[n - 1][a[n - 1]] = n - 1;
		for (int i = n - 2; i >= 0; --i)
		{
			right[i] = right[i + 1];
			right[i][a[i]] = i;
		}
		for (int i = 0; i < m; ++i)
		{
			int idx = q[i][0], c = q[i][1];
			ans[i] = INT_MAX;
			if (left[idx][c] != INT_MAX) ans[i] = min(ans[i], idx - left[idx][c]);
			if (right[idx][c] != INT_MAX) ans[i] = min(ans[i], right[idx][c] - idx);
			if (ans[i] > n)
			{
				ans[i] = -1;
			}
		}
		return ans;
	}

	//1184. Distance Between Bus Stops
	int distanceBetweenBusStops(vector<int>& a, int st, int ed) {
		int ans = 0;
		int n = a.size();
		for (int i = st; i != ed; i = (i + 1) % n)
		{
			ans += a[i];
		}
		int sum = 0;
		for (auto& e : a) sum += e;
		return min(ans, sum - ans);
	}

	//1186. Maximum Subarray Sum with One Deletion
	int maximumSum(vector<int>& a) {
		int n = a.size();
		vector<vector<int>> dp(n, vector<int>(2));
		dp[0][0] = a[0];
		dp[0][1] = max(0, a[0]);
		for (int i = 1; i < n; ++i)
		{
			dp[i][0] = max(a[i], dp[i - 1][0] + a[i]);
			dp[i][1] = max(dp[i - 1][0], dp[i - 1][1] + a[i]);
		}
		int ans = INT_MIN;
		for (int& e : a)
			ans = max(ans, e);
		if (ans <= 0)
		{
			return ans;
		}
		for (int i = 0; i < n; ++i)
		{
			ans = max(ans, dp[i][0]);
			ans = max(ans, dp[i][1]);
		}
		return ans;
	}

	//1185. Day of the Week
	string dayOfTheWeek(int day, int month, int y) {
		if (month < 3) {
			y--;
			month += 12;
		}
		static string S[7] = { "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" };
		int w = (y + y / 4 - y / 100 + y / 400 + (13 * month + 8) / 5 + day) % 7;
		return S[w];
	}

	//1183. Maximum Number of Ones
	int maximumNumberOfOnes(int m, int n, int k, int x) {
		// a[i][j] == a[i + k][j]
		vector<vector<int>> cnt(k, vector<int>(k, 0));
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				cnt[i % k][j % k]++;
			}
		}
		vector<int> result;
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < k; j++) 
				result.push_back(cnt[i][j]);
		}
		sort(result.begin(), result.end());
		reverse(result.begin(), result.end());
		int ans = 0;
		for (int i = 0; i < x; i++) ans += result[i];
		return ans;
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