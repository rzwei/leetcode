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


};

int main()
{
	return 0;
}