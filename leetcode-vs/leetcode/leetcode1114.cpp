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
		map<int, priority_queue<int>> len;
		for (int& e : a)
		{
			if (len.empty() || len.begin()->first >= e)
			{
				len[e].push(-1);
			}
			else
			{
				auto v = -len.begin()->second.top() + 1;
				len[e].push(-v);
				len.begin()->second.pop();
				if (len.begin()->second.empty())
				{
					len.erase(len.begin());
				}
			}
		}
		for (auto& e : len)
		{
			auto& pq = e.second;
			if (!pq.empty() && -pq.top() < k) return false;
		}
		return true;
	}
};

int main()
{
	return 0;
}