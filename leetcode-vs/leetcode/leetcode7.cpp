//#include "stdafx.h"
#include <bitset>
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
typedef long long ll;
//703. Kth Largest Element in a Stream
class KthLargest {
public:
	priority_queue<int, vector<int>, greater<int>> pq;
	int size;
	KthLargest(int k, vector<int> nums) {
		size = k;
		for (int e : nums)
		{
			pq.push(e);
			if (pq.size() > size)
				pq.pop();
		}
	}

	int add(int val) {
		pq.push(val);
		if (pq.size() > size) pq.pop();
		return pq.top();
	}
};
//705. Design HashSet
class MyHashSet {
public:
	/** Initialize your data structure here. */
	const static int MAXN = 1000;
	vector<list<int>> s;
	MyHashSet() {
		s.assign(MAXN, list<int>());
	}

	void add(int key) {
		int i = key % MAXN;
		auto it = find(s[i].begin(), s[i].end(), key);
		if (it == s[i].end()) s[i].push_back(key);
	}

	void remove(int key) {
		int idx = key % MAXN;
		auto it = find(s[idx].begin(), s[idx].end(), key);
		if (it != s[idx].end()) s[idx].erase(it);
	}

	/** Returns true if this set did not already contain the specified element */
	bool contains(int key) {
		int i = key % MAXN;
		auto it = find(s[i].begin(), s[i].end(), key);
		return it != s[i].end();
	}
};
//706. Design HashMap
class MyHashMap {
public:
	const static int MAXN = 1000;
	vector<list<pair<int, int>>> s;
	MyHashMap() {
		s.assign(MAXN, list<pair<int, int>>());
	}

	/** value will always be positive. */
	void put(int key, int value) {
		int i = key % MAXN;
		auto it = find(i, key);
		if (it == s[i].end())
			s[i].emplace_back(key, value);
		else
			it->second = value;
	}

	list<pair<int, int>>::iterator find(int i, int v)
	{
		for (auto it = s[i].begin(); it != s[i].end(); ++it)
			if (it->first == v) return it;
		return s[i].end();
	}

	/** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
	int get(int key) {
		int i = key % MAXN;
		auto it = find(i, key);
		if (it == s[i].end()) return -1;
		return it->second;
	}

	/** Removes the mapping of the specified value key if this map contains a mapping for the key */
	void remove(int key) {
		int i = key % MAXN;
		auto it = find(i, key);
		if (it != s[i].end())
			s[i].erase(it);
	}
};
//707. Design Linked List
class MyLinkedList {
public:
	list<int> nums;
	/** Initialize your data structure here. */
	MyLinkedList() {

	}

	/** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
	int get(int index) {
		if (index >= nums.size()) return -1;
		auto it = getidx(index);
		return *it;
	}


	/** Add a node of value val before the first element of the linked list.
	After the insertion, the new node will be the first node of the linked list. */
	void addAtHead(int val) {
		nums.push_front(val);
	}

	/** Append a node of value val to the last element of the linked list. */
	void addAtTail(int val) {
		nums.push_back(val);
	}

	/** Add a node of value val before the index-th node in the linked list.
	If index equals to the length of linked list, the node will be appended to the end of linked list.
	If index is greater than the length, the node will not be inserted. */
	void addAtIndex(int index, int val) {
		if (index > nums.size()) return;
		if (index == nums.size())
		{
			nums.push_back(val);
			return;
		}
		auto it = getidx(index);
		nums.insert(it, val);
	}
	list<int>::iterator getidx(int idx)
	{
		auto it = nums.begin();
		while (idx)
		{
			it++;
			idx--;
		}
		return it;
	}
	/** Delete the index-th node in the linked list, if the index is valid. */
	void deleteAtIndex(int index) {
		if (index >= nums.size()) return;
		auto it = getidx(index);
		nums.erase(it);
	}
};
//622. Design Circular Queue
class MyCircularQueue {
public:
	/** Initialize your data structure here. Set the size of the queue to be k. */
	int l, r, k, size;
	vector<int> a;
	MyCircularQueue(int k) :a(k), l(0), r(0), k(k), size(0) {

	}

	/** Insert an element into the circular queue. Return true if the operation is successful. */
	bool enQueue(int value) {
		if (isFull()) return false;
		a[(r++) % k] = value;
		size++;
		return true;
	}

	/** Delete an element from the circular queue. Return true if the operation is successful. */
	bool deQueue() {
		if (isEmpty()) return false;
		size--;
		l++;
		return true;
	}

	/** Get the front item from the queue. */
	int Front() {
		if (isEmpty()) return -1;
		return a[l%k];
	}

	/** Get the last item from the queue. */
	int Rear() {
		if (isEmpty()) return -1;
		return a[(r - 1 + k) % k];
	}

	/** Checks whether the circular queue is empty or not. */
	bool isEmpty() {
		return size == 0;
	}

	/** Checks whether the circular queue is full or not. */
	bool isFull() {
		return size == k;
	}
};
//641. Design Circular Deque
class MyCircularDeque {
public:
	vector<int> a;
	int l, r, k, size;
	/** Initialize your data structure here. Set the size of the deque to be k. */
	MyCircularDeque(int k) :l(0), r(0), k(k), size(0), a(k) {

	}

	/** Adds an item at the front of Deque. Return true if the operation is successful. */
	bool insertFront(int value) {
		if (isFull()) return false;
		a[((--l) % k + k) % k] = value;
		size++;
		return true;
	}

	/** Adds an item at the rear of Deque. Return true if the operation is successful. */
	bool insertLast(int value) {
		if (isFull()) return false;
		a[(r%k + k) % k] = value;
		size++;
		r++;
		return true;
	}

	/** Deletes an item from the front of Deque. Return true if the operation is successful. */
	bool deleteFront() {
		if (isEmpty()) return false;
		l++;
		size--;
		return true;
	}

	/** Deletes an item from the rear of Deque. Return true if the operation is successful. */
	bool deleteLast() {
		if (isEmpty()) return false;
		r--;
		size--;
		return true;
	}

	/** Get the front item from the deque. */
	int getFront() {
		if (isEmpty()) return -1;
		return a[(l%k + k) % k];
	}

	/** Get the last item from the deque. */
	int getRear() {
		if (isEmpty()) return -1;
		return a[((r - 1) % k + k) % k];
	}

	/** Checks whether the circular deque is empty or not. */
	bool isEmpty() {
		return size == 0;
	}

	/** Checks whether the circular deque is full or not. */
	bool isFull() {
		return size == k;
	}
};


//880. Random Pick with Weight
//class Solution {
//public:
//	vector<int> sums;
//	int sum = 0;
//	Solution(vector<int> w) {
//		int n = w.size();
//		sums.assign(n, 0);
//		sums[0] = w[0];
//		sum = w[0];
//		for (int i = 1; i < n; ++i)
//		{
//			sums[i] = sums[i - 1] + w[i];
//		}
//		sum = sums.back();
//	}
//
//	int pickIndex() {
//		int idx = rand() % sum;
//		auto it = upper_bound(sums.begin(), sums.end(), idx) - sums.begin();
//		return it;
//	}
//};
//883. Generate Random Point in a Circle
//class Solution {
//public:
//	double r, x, y;
//	const int maxn = 1e3;
//	const double pi = 3.141592653589;
//	Solution(double r, double x, double y) :r(r), x(x), y(y) {
//
//	}
//
//	vector<double> randPoint() {
//		int v1 = rand() % maxn, v2 = rand() % maxn;
//		int v = v1 * maxn + v2;
//		double delta = pi * 2 / 1e6 * v;
//		return { x + r * cos(delta),y + r * sin(delta) };
//	}
//};
//882. Random Point in Non - overlapping Rectangles
//class Solution {
//public:
//	vector<vector<int>> pos;
//	vector<int> w;
//	int tot = 0;
//	int const maxn = 1000;
//	Solution(vector<vector<int>> rects) {
//		int n = rects.size();
//		w.assign(n, 0);
//		pos.assign(n, vector<int>());
//		for (int i = 0; i < n; ++i)
//		{
//			int x0 = rects[i][0], y0 = rects[i][1], x1 = rects[i][2], y1 = rects[i][3];
//			int row = y1 - y0 + 1;
//			int col = x1 - x0 + 1;
//			w[i] = row * col;
//			pos[i] = { x0,y0,col };
//			tot += row * col;
//		}
//		for (int i = 1; i < n; ++i) w[i] += w[i - 1];
//	}
//	int mrand()
//	{
//		return rand() % maxn  * maxn + rand() % maxn;
//	}
//	vector<int> pick() {
//		int v = mrand() % tot;
//		auto i = upper_bound(w.begin(), w.end(), v) - w.begin();
//		int d = w[i];
//		if (i) v -= w[i - 1];
//		return { pos[i][0] + v % pos[i][2] , pos[i][1] + v / pos[i][2] };
//	}
//};
//881. Random Flip Matrix
//class Solution {
//public:
//	int n, m;
//	int tot;
//	unordered_map<int, int> s;
//	Solution(int n_rows, int n_cols) :n(n_rows), m(n_cols), tot(n*m) {
//
//	}
//	int mrand() {
//		return rand() % 10000 * 10000 + rand() % 10000;
//	}
//	vector<int> flip() {
//		int k = mrand() % tot;
//		tot--;
//		int val = k;
//		if (s.count(k)) val = s[k];
//		if (!s.count(tot)) s[k] = tot;
//		else s[k] = s[tot];
//		return { val / m,val %m };
//	}
//	void reset() {
//		s.clear();
//		tot = m * n;
//	}
//};
class Solution {
public:
	//700. Search in a Binary Search Tree
	TreeNode * searchBST(TreeNode* root, int val) {
		if (!root) return nullptr;
		if (root->val == val) return root;
		if (val < root->val) return searchBST(root->left, val);
		return searchBST(root->right, val);
	}
	//701. Insert into a Binary Search Tree
	TreeNode* insertIntoBST(TreeNode* root, int val) {
		if (!root) return new TreeNode(val);
		if (val < root->val)
			root->left = insertIntoBST(root->left, val);
		else
			root->right = insertIntoBST(root->right, val);
		return root;
	}

	//704. Binary Search
	int search(vector<int>& nums, int target) {
		int l = 0, r = nums.size() - 1;
		while (l <= r)
		{
			int m = (l + r) / 2;
			if (nums[m] == target) return m;
			else if (nums[m] < target) l = m + 1;
			else r = m - 1;
		}
		return -1;
	}

	//709. To Lower Case
	string toLowerCase(string str) {
		for (char &c : str) c = tolower(c);
		return str;
	}

	//868. Binary Gap 
	int binaryGap(int N) {
		bitset<32> b(N);
		int ans = 0;
		int last = -1;

		for (int i = 0; i < 32; ++i)
		{
			if (b[i] == 1)
			{
				if (last == -1)
				{
					last = i;
				}
				else {
					ans = max(ans, i - last);
					last = i;
				}
			}
		}
		return ans;
	}
	//869. Reordered Power of 2 
	bool reorderedPowerOf2(int N) {
		auto p = to_string(N);
		vector<int> d(10);
		for (char c : p)
			d[c - '0']++;

		for (int i = 1; i < 1e9; i <<= 1)
		{
			auto s = to_string(i);
			vector<int> d2(10);
			for (char c : s)
			{
				d2[c - '0']++;
			}
			if (d == d2)
			{
				return true;
			}
		}
		return false;
	}
	//870. Advantage Shuffle 
	vector<int> advantageCount(vector<int>& a, vector<int>& b) {
		int len = a.size();
		multiset<int> sa(a.begin(), a.end());
		vector<int> ans(len);

		vector<pair<int, int>>bb(len);
		for (int i = 0; i < len; ++i)
		{
			bb[i].first = b[i];
			bb[i].second = i;
		}
		sort(bb.begin(), bb.end());

		for (int i = 0; i < len; ++i)
		{
			auto idx = bb[i].second;
			int v = bb[i].first;
			auto it = sa.upper_bound(v);
			if (it == sa.end())
			{
				it = sa.begin();
			}
			ans[idx] = *it;
			sa.erase(it);
		}
		return ans;
	}
	//871. Minimum Number of Refueling Stops 
	int minRefuelStops(int t, int sf, vector<vector<int>>& st) {
		priority_queue<int> pq;
		ll ans = 0, cur = 0, tank = sf;
		st.push_back({ t,0 });
		for (int i = 0; i < st.size(); ++i)
		{
			ll dist = st[i][0] - cur;
			while (dist > tank)
			{
				if (pq.empty()) return -1;
				auto tmp = pq.top();
				pq.pop();
				tank += tmp;
				ans++;
			}
			tank -= dist;
			cur = st[i][0];
			pq.push(st[i][1]);
		}
		return ans;
	}

	//872. Implement Rand10() Using Rand7()
	int rand7() {
		return rand() % 7 + 1;
	}
	int rand10() {
		int d;
		while ((d = rand7()) == 4);
		if (d < 4)
		{
			while (((d = rand7()) > 5));
			return d;
		}
		else {
			while (((d = rand7()) > 5));
			return d + 5;
		}
	}

	void getleaf(TreeNode *root, vector<int> &a)
	{
		if (!root) return;
		if (!root->left && !root->right) {
			a.push_back(root->val);
			return;
		}
		getleaf(root->left, a);
		getleaf(root->right, a);
	}
	//872. Leaf-Similar Trees 
	bool leafSimilar(TreeNode* root1, TreeNode* root2) {
		vector<int> a, b;
		getleaf(root1, a);
		getleaf(root2, b);
		return a == b;
	}

	//874. Walking Robot Simulation 
	int robotSim(vector<int>& cmds, vector<vector<int>>& obs) {
		vector<vector<int>> dirs = { { 0,1 },{ 1,0 },{ 0,-1 },{ -1,0 } };
		set<pair<int, int>> s;
		for (auto &e : obs)
			s.insert({ e[0],e[1] });
		int d = 0;
		int x = 0, y = 0;
		int ans = 0;
		for (int c : cmds)
		{
			if (c == -2) {
				d = (d - 1 + 4) % 4;
			}
			else if (c == -1) {
				d = (d + 1) % 4;
			}
			else {
				for (int i = 0; i < c; ++i)
				{
					int nx = x + dirs[d][0], ny = y + dirs[d][1];
					if (s.count({ nx,ny })) break;
					x = nx, y = ny;
				}
				ans = max(ans, x*x + y * y);
			}
		}
		return ans;
	}
	int robotSim2(vector<int>& cmds, vector<vector<int>>& obs) {
		vector<vector<int>> dirs = { { 0,1 },{ 1,0 },{ 0,-1 },{ -1,0 } };

		map<int, set<int>> idx, idx2;
		for (auto ob : obs)
		{
			idx[ob[0]].insert(ob[1]);
			idx2[ob[1]].insert(ob[0]);
		}

		int d = 0;
		int x = 0, y = 0;
		int ans = 0;
		for (int c : cmds)
		{
			if (c == -2) {
				d = (d - 1 + 4) % 4;
			}
			else if (c == -1) {
				d = (d + 1) % 4;
			}
			else {
				int nx = x + dirs[d][0] * c, ny = y + dirs[d][1] * c;
				if (nx == x) {
					if (idx.count(x))
					{
						auto it = idx[x].lower_bound(y);
						int l, r;
						if (it == idx[x].begin())
							l = INT_MIN;
						else
						{
							l = *(--it);
							++it;
						}
						if (it == idx[x].end())
							r = INT_MAX;
						else
							r = *it;
						if (dirs[d][1] > 0)
							ny = min(ny, r - 1);
						else
							ny = max(ny, l + 1);
					}
				}
				else if (ny == y)
				{
					if (idx2.count(y))
					{
						auto it = idx2[y].lower_bound(x);
						int l, r;
						if (it == idx2[y].begin())
							l = INT_MIN;
						else
						{
							l = *(--it);
							++it;
						}
						if (it == idx2[y].end())
							r = INT_MAX;
						else
							r = *it;
						if (dirs[d][0] > 0)
							nx = min(nx, r - 1);
						else
							nx = max(nx, l + 1);
					}
				}
				ans = max(ans, nx * nx + ny * ny);
				x = nx, y = ny;
			}
		}
		return ans;
	}

	bool check_875(vector<int> &a, int k, int H)
	{
		int ans = 0;
		for (int e : a)
		{
			ans += ceil(double(e) / k);
		}
		return ans <= H;
	}
	//875. Koko Eating Bananas 
	int minEatingSpeed(vector<int>& a, int H) {
		int l = 1, r = INT_MIN;
		for (int e : a) r = max(r, e);
		while (l < r)
		{
			int m = l + (r - l) / 2;
			if (check_875(a, m, H))
				r = m;
			else
				l = m + 1;
		}
		return l;
	}

	int find(vector<int> &a, int s, int v)
	{
		if (s == a.size()) return -1;
		auto it = lower_bound(a.begin() + s, a.end(), v);
		if (it == a.end()) return -1;
		if (*it == v)
			return it - a.begin();
		return -1;
	}
	//873. Length of Longest Fibonacci Subsequence 
	int lenLongestFibSubseq(vector<int>& A) {
		int ans = 0, n = A.size();
		vector<vector<int>> dp(n, vector<int>(n));
		map<int, int> idx;
		for (int i = 0; i < n; ++i)
			idx[A[i]] = i;

		for (int i = 0; i < n; ++i)
			for (int j = 0; j < i; ++i)
			{
				dp[j][i] = 2;
				if (idx.count(A[i] - A[j]))
				{
					int k = idx[A[i] - A[j]];
					dp[j][i] = max(dp[j][i], dp[k][j] + 1);
				}
				ans = max(ans, dp[j][i]);
			}
		if (ans <= 2) return 0;
		return ans;
	}
	int lenLongestFibSubseq2(vector<int>& A)
	{
		int ans = 0, n = A.size();
		vector<vector<int>> dp(n, vector<int>(n));
		map<int, int> idx;
		for (int i = 0; i < n; ++i)
			idx[A[i]] = i;

		for (int i = 0; i < n; ++i)
			for (int j = 0; j < i; ++j)
			{
				dp[i][j] = 2;
				if (idx.count(A[i] - A[j]))
				{
					int k = idx[A[i] - A[j]];
					dp[i][j] = max(dp[i][j], dp[j][k] + 1);
				}
				ans = max(ans, dp[i][j]);
			}
		if (ans <= 2) return 0;
		return ans;
	}

	//876. Middle of the Linked List
	ListNode* middleNode(ListNode* head) {
		auto slow = head, fast = head;
		while (fast && fast->next)
		{
			slow = slow->next;
			fast = fast->next->next;
		}
		return slow;
	}

	//877. Stone Game
	bool stoneGame(vector<int>& a) {
		int n = a.size();
		vector<vector<int>> dp(n, vector<int>(n));
		for (int i = 0; i < n; ++i)
			dp[i][i] = a[i];
		for (int l = 2; l <= n; ++l)
		{
			for (int i = 0, j = i + l - 1; j < n; ++i, ++j)
				dp[i][j] = max(a[i] - dp[i + 1][j], a[j] - dp[i][j - 1]);
		}
		return dp[0][n - 1] > 0;
	}
	//878. Nth Magical Number
	ll gcd(ll a, ll b) { if (a < b) swap(a, b); ll t; while (b) { t = b; b = a % b; a = t; } return a; }
	int nthMagicalNumber(int N, int A, int B) {
		int const mod = 1e9 + 7;
		ll l = min(A, B), r = 100 * max((ll)N * A, (ll)N * B);
		ll lcm = (ll)A / gcd(A, B) * B;
		while (l < r)
		{
			ll m = l + (r - l) / 2;
			ll cnt = m / A + m / B - m / lcm;
			if (cnt < N) l = m + 1;
			else r = m;
		}
		return l % mod;
	}
	//879. Profitable Schemes
	//int dfs_879(int u, int g, int p, vector<int> &gs, vector<int> &ps, vector<vector<vector<int>>> &memo)
	//{
	//	static int const mod = 1e9 + 7;
	//	if (u == -1) return p <= 0;
	//	if (p < 0) p = 0;
	//	if (g < 0) return 0;
	//	if (memo[u][g][p] != -1) return memo[u][g][p];
	//	ll ans = dfs_879(u - 1, g, p, gs, ps, memo);
	//	if (g >= gs[u])
	//		ans += dfs_879(u - 1, g - gs[u], p - ps[u], gs, ps, memo);
	//	ans %= mod;
	//	memo[u][g][p] = ans;
	//	return ans;
	//}
	//int profitableSchemes(int G, int P, vector<int>& group, vector<int>& profit) {
	//	vector<vector<vector<int>>>
	//		memo(group.size(), vector<vector<int>>(G + 1, vector<int>(P + 1, -1)));
	//	return dfs_879(group.size() - 1, G, P, group, profit, memo);
	//}
	int profitableSchemes(int G, int P, vector<int>& group, vector<int>& profit) {
		vector<vector<int>> dp(P + 1, vector<int>(G + 1));
		int const mod = 1e9 + 7;
		for (int k = 0; k < group.size(); ++k)
		{
			int g = group[k], p = profit[k];
			for (int i = P; i >= 0; --i)
			{
				for (int j = G - p; j >= 0; --j)
				{
					dp[min(i + p, P)][j + g] = (dp[min(i + p, P)][j + g] + dp[i][j]) % mod;
				}
			}
		}
		int ans = 0;
		for (int e : dp[P])  ans = (ans + e) % mod;
		return ans;
	}
	//558. Quad Tree Intersection
	//Node * intersect(Node* q1, Node* q2) {
	//	if (!q1 && !q2) return nullptr;
	//	if (q1 && q2)
	//	{
	//		if (q1->isLeaf && q2->isLeaf)
	//		{
	//			q1->val = q1->val || q2->val;
	//			return q1;
	//		}
	//		if (q1->isLeaf && q1->val) return q1;
	//		if (q2->isLeaf && q2->val) return q2;

	//		auto r = new Node();
	//		r->topLeft = intersect(q1->topLeft, q2->topLeft);
	//		r->topRight = intersect(q1->topRight, q2->topRight);
	//		r->bottomLeft = intersect(q1->bottomLeft, q2->bottomLeft);
	//		r->bottomRight = intersect(q1->bottomRight, q2->bottomRight);
	//		r->isLeaf = false;
	//		if (r->topLeft && r->topLeft->isLeaf &&
	//			r->topRight && r->topRight->isLeaf &&
	//			r->bottomLeft && r->bottomLeft->isLeaf &&
	//			r->bottomRight && r->bottomRight->isLeaf
	//			)
	//		{
	//			if (r->topLeft->val && r->topRight->val && r->bottomLeft->val && r->bottomRight->val)
	//			{
	//				r->isLeaf = true;
	//				r->val = true;
	//			}
	//			else if (!r->topLeft->val && !r->topRight->val && !r->bottomLeft->val && !r->bottomRight->val)
	//			{
	//				r->isLeaf = true;
	//				r->val = false;
	//			}
	//		}
	//		return r;
	//	}
	//	else if (q1)
	//	{
	//		return q1;
	//	}
	//	else {
	//		return q2;
	//	}
	//}
	//427. Construct Quad Tree
	//	Node * build(int i, int j, int l, vector<vector<int>>& g)
	//	{
	//		if (l == 0) return nullptr;
	//		auto r = new Node(false, false, nullptr, nullptr, nullptr, nullptr);
	//		if (l == 1) {
	//			r->isLeaf = true;
	//			r->val = g[i][j];
	//		}
	//		else {
	//			auto tl = build(i, j, l / 2, g);
	//			auto tr = build(i, j + l / 2, l / 2, g);
	//			auto bl = build(i + l / 2, j, l / 2, g);
	//			auto br = build(i + l / 2, j + l / 2, l / 2, g);
	//			if (tl->isLeaf && tr->isLeaf && bl->isLeaf && br->isLeaf &&
	//				(tl->val && tr->val && bl->val && br->val || !tl->val && !tr->val && !bl->val && !br->val))
	//			{
	//				r->isLeaf = true;
	//				r->val = tl->val;
	//				delete tl;
	//				delete tr;
	//				delete bl;
	//				delete br;
	//			}
	//			else {
	//				r->topLeft = tl; r->topRight = tr;
	//				r->bottomLeft = bl; r->bottomRight = br;
	//			}
	//		}
	//		return r;
	//	}
	//
	//	Node* construct(vector<vector<int>>& grid) {
	//		return build(0, 0, grid.size(), grid);
	//	}
	//430. Flatten a Multilevel Doubly Linked List
	//Node * build(Node *pre, Node *p)
	//{
	//	while (p)
	//	{
	//		pre->next = p;
	//		p->prev = pre;
	//		pre = p;
	//		auto nx = p->next;
	//		if (p->child)
	//		{
	//			auto e = build(pre, p->child);
	//			p->child = nullptr;
	//			pre = e;
	//		}
	//		p = nx;
	//	}
	//	return pre;
	//}

	//Node * flatten(Node* head) {
	//	if (!head) return nullptr;
	//	auto dummy = new Node();
	//	auto r = build(dummy, head);
	//	r->next = nullptr;
	//	if (dummy->next)
	//		dummy->next->prev = nullptr;
	//	return dummy->next;
	//}

	//429. N - ary Tree Level Order Traversal
	//vector<vector<int>> levelOrder(Node* root) {
	//	if (!root) return {};
	//	vector<vector<int>> ans;
	//	queue<Node *>q;
	//	q.push(root);
	//	while (!q.empty())
	//	{
	//		int size = q.size();
	//		vector<int> cur;
	//		while (size--) {
	//			auto u = q.front();
	//			q.pop();
	//			cur.push_back(u->val);
	//			for (auto p : u->children)
	//				if (p) q.push(p);
	//		}
	//		ans.push_back(cur);
	//	}
	//	return ans;
	//}

	//589. N - ary Tree Preorder Traversal
	//vector<int> preorder(Node* root) {
	//	vector<int> ans;
	//	if (!root) return ans;
	//	ans.push_back(root->val);
	//	typedef vector<Node *>::iterator nit;
	//	stack<pair<Node*, nit>> stk;
	//	Node* cur = root;
	//	nit it = root->children.begin();
	//	while (cur && it != cur->children.end() || !stk.empty())
	//	{
	//		if (cur && it != cur->children.end())
	//		{
	//			ans.push_back((*it)->val);
	//			auto nx = ++it;
	//			--it;
	//			stk.push({ cur,nx });
	//			cur = *it;
	//			it = cur->children.begin();
	//		}
	//		else {
	//			auto top = stk.top(); stk.pop();
	//			cur = top.first;
	//			it = top.second;
	//		}
	//	}
	//	return ans;
	//}

	//590. N-ary Tree Postorder Traversal
	//vector<int> postorder(Node* root) {
	//	if (root == NULL) return {};
	//	vector<int> res;
	//	stack<Node*> stk;
	//	stk.push(root);
	//	while (!stk.empty())
	//	{
	//		Node* temp = stk.top();
	//		stk.pop();
	//		for (int i = 0; i < temp->children.size(); i++) stk.push(temp->children[i]);
	//		res.push_back(temp->val);
	//	}
	//	reverse(res.begin(), res.end());
	//	return res;
	//}
	//887. Projection Area of 3D Shapes
	int projectionArea(vector<vector<int>>& g) {
		int ans = 0, n = g.size(), m = g[0].size();
		vector<int> h(m), w(n);
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				if (g[i][j]) ans++;
				w[i] = max(w[i], g[i][j]);
				h[j] = max(h[j], g[i][j]);
			}
		}
		for (int e : h) ans += e;
		for (int e : w) ans += e;
		return ans;
	}
	//885. Boats to Save People
	int numRescueBoats(vector<int>& a, int limit) {
		multiset<int> pre;
		sort(a.begin(), a.end());
		int n = a.size();
		int ans = 0;
		for (int i = n - 1; i >= 0; --i)
		{
			auto it = pre.lower_bound(a[i]);
			if (it != pre.end())
				pre.erase(it);
			else {
				pre.insert(limit - a[i]);
				ans++;
			}
		}
		return ans;
	}
	//884. Decoded String at Index
	string decodeAtIndex(string s, int k) {
		ll len = 0;
		ll cur = 0;
		int sl = s.size();
		for (int i = 0; i < sl; ++i)
		{
			if (!isdigit(s[i]))
			{
				len++;
				cur++;
				if (cur == k)
					return string(1, s[i]);
			}
			else {
				ll d = s[i] - '0';
				if (k > d  * len)
				{
					cur = d * len;
					len *= d;
				}
				else {
					k = k % len;
					if (k == 0) k = len;
					i = -1;
					len = 0;
					cur = 0;
				}
			}
		}
	}
	//886. Reachable Nodes In Subdivided Graph
	int reachableNodes(vector<vector<int>>& edges, int M, int N) {
		unordered_map<int, unordered_map<int, int>> e;
		for (auto v : edges) e[v[0]][v[1]] = e[v[1]][v[0]] = v[2];
		priority_queue<pair<int, int>> pq;
		pq.push({ M, 0 });
		unordered_map<int, int> seen;
		while (pq.size()) {
			int moves = pq.top().first, i = pq.top().second;
			pq.pop();
			if (!seen.count(i)) {
				seen[i] = moves;
				for (auto j : e[i]) {
					int moves2 = moves - j.second - 1;
					if (!seen.count(j.first) && moves2 >= 0)
						pq.push({ moves2, j.first });
				}
			}
		}
		int res = seen.size();
		for (auto v : edges) {
			int a = seen.find(v[0]) == seen.end() ? 0 : seen[v[0]];
			int b = seen.find(v[1]) == seen.end() ? 0 : seen[v[1]];
			res += min(a + b, v[2]);
		}
		return res;
	}
	vector<string> split(string &s)
	{
		vector<string> ans;
		int len = s.size();
		for (int i = 0; i < len; ++i)
		{
			while (i < len && s[i] == ' ') i++;
			string tmp;
			while (i < len && s[i] != ' ') {
				tmp.push_back(s[i++]);
			}
			if (!tmp.empty())
			{
				ans.push_back(tmp);
			}
		}
		return ans;
	}
	//888. Uncommon Words from Two Sentences 
	vector<string> uncommonFromSentences(string A, string B) {
		auto a = split(A), b = split(B);
		map<string, int> cnta, cntb;
		for (auto &s : a)
			cnta[s]++;
		for (auto &s : b)
			cntb[s]++;
		vector<string> ans;
		for (auto &p : cnta)
		{
			if (p.second == 1 && !cntb.count(p.first))
				ans.push_back(p.first);
		}
		for (auto &p : cntb)
		{
			if (p.second == 1 && !cnta.count(p.first))
				ans.push_back(p.first);
		}
		return ans;
	}

	//889. Spiral Matrix III 
	vector<vector<int>> spiralMatrixIII(int R, int C, int r0, int c0) {
		vector<vector<int>> ans;
		vector<vector<int>> dirs = { { 0,1 },{ 1,0 },{ 0,-1 },{ -1,0 } };
		int d = 0, tot = 1, len = 1;
		int x = r0, y = c0;
		ans.push_back({ x,y });
		int turn = 0;
		while (tot < R * C)
		{
			for (int i = 0; i < len; ++i)
			{
				x = x + dirs[d][0];
				y = y + dirs[d][1];
				if (0 <= x && x < R && 0 <= y && y < C) {
					ans.push_back({ x,y });
					tot++;
				}
			}
			turn++;
			d = (d + 1) % 4;
			if (turn % 2 == 0)
				len++;
		}
		return ans;
	}
	bool dfs_890(int u, vector<int> &color, vector<vector<int>> &g)
	{
		vector<int> nx;
		for (int v : g[u])
		{
			if (color[v] == 0)
			{
				color[v] = -color[u];
				nx.push_back(v);
			}
			else
				if (color[v] == color[u])
					return false;
		}
		for (auto v : nx)
		{
			if (!dfs_890(v, color, g))
				return false;
		}
		return true;
	}

	//890. Possible Bipartition 
	bool possibleBipartition(int N, vector<vector<int>>& dislikes) {
		vector<vector<int>> g(N + 1);
		for (auto &e : dislikes)
		{
			int a = e[0], b = e[1];
			g[a].push_back(b);
			g[b].push_back(a);
		}
		vector<int> color(N + 1);
		for (int i = 1; i <= N; ++i)
		{
			if (color[i] == 0)
			{
				color[i] = 1;
				if (!dfs_890(i, color, g)) return false;
			}
		}
		return true;
	}

	//891. Super Egg Drop
	int superEggDrop(int K, int N) {
		int const maxn = INT_MAX / 16;
		vector<vector<int>> dp(K + 1, vector<int>(N + 1, maxn));
		dp[0][0] = 0;
		for (int i = 1; i <= K; ++i)
		{
			dp[i][1] = 1;
			dp[i][0] = 0;
		}
		for (int i = 1; i <= N; ++i)
		{
			dp[1][i] = i;
		}
		for (int k = 1; k <= K; ++k)
		{
			for (int n = 1; n <= N; ++n)
			{
				int l = 1, r = n;
				int res = n;
				while (l < r)
				{
					int m = (l + r) / 2;
					int left = dp[k - 1][m - 1], right = dp[k][n - m];
					res = min(res, max(left, right) + 1);
					if (left < right)
						l = m + 1;
					else
						r = m;
				}
				dp[k][n] = res;
			}
		}
		return dp[K][N];
	}

};

int main()
{
	return 0;
}