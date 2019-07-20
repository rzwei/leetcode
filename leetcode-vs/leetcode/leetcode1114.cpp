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


	pair<int, TreeNode*> dfs(TreeNode* u)
	{
		if (!u) return { 0, u };
		// return max(dfs(u->left), dfs(u->right)) + 1;
		auto left = dfs(u->left), right = dfs(u->right);
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
		auto ret = dfs(root);
		return ret.second;
	}
};

int main()
{
	return 0;
}