
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
};

int main()
{
	return 0;
}