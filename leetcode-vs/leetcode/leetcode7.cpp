//#include "stdafx.h"
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
};

int main()
{
	return 0;
}