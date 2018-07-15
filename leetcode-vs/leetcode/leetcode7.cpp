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
	//868. Binary Gap 
	int binaryGap(int N) {
		bitset<32> b(N);
		int ans = 0;
		int last = 0;

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
};

int main()
{
	return 0;
}