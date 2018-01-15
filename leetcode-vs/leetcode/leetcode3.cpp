#include "stdafx.h"
#include <sstream>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <queue>
#include <climits>
#include <stack>

using namespace std;
class AllOne {
public:
	/** Initialize your data structure here. */
	AllOne() {

	}

	/** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
	void inc(string key) {

	}

	/** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
	void dec(string key) {

	}

	/** Returns one of the keys with maximal value. */
	string getMaxKey() {

	}

	/** Returns one of the keys with Minimal value. */
	string getMinKey() {

	}
};


class LFUCache {
public:
	unordered_map<int, int> m;
	vector<int> nums;
	LFUCache(int capacity) {

	}

	int get(int key) {

	}

	void put(int key, int value) {

	}
};
class Solution {
public:
	bool isPrime(int a)
	{
		static set<int> prinmes{ 2,3,5,7,11,13,17,19,23,29,31 };
		int n = 0;
		while (a)
		{
			a &= (a - 1);
			n++;
		}
		return prinmes.find(n) != prinmes.end();
	}
	//762. Prime Number of Set Bits in Binary Representation 
	int countPrimeSetBits(int L, int R) {
		int ans = 0;
		for (int i = L; i <= R; i++)
			ans += isPrime(i);
		return ans;
	}
	//763. Partition Labels 
	vector<int> partitionLabels(string S) {
		map<char, vector<int>> idxes;
		int len = S.size();
		for (int i = 0; i < len; i++)
		{
			if (idxes.find(S[i]) == idxes.end())
				idxes[S[i]] = { i,i };
			else
				idxes[S[i]][1] = i;
		}

	}
	//764. Largest Plus Sign 
	int orderOfLargestPlusSign(int n, vector<vector<int>>& mines) {
		vector<vector<int>> matrix(n, vector<int>(n, 1));

		for (auto& zero : mines)
			matrix[zero[0]][zero[1]] = 0;

		vector<vector<int>> lf(n, vector<int>(n, 0));
		auto rt = lf, up = lf, dn = lf;

		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				if (matrix[i][j] == 1) {
					lf[i][j] = j == 0 ? 1 : lf[i][j - 1] + 1;
					up[i][j] = i == 0 ? 1 : up[i - 1][j] + 1;
				}
		int ans = 0;
		for (int i = n - 1; i >= 0; --i)
			for (int j = n - 1; j >= 0; --j)
				if (matrix[i][j] == 1) {
					rt[i][j] = j == n - 1 ? 1 : rt[i][j + 1] + 1;
					dn[i][j] = i == n - 1 ? 1 : dn[i + 1][j] + 1;

					ans = max(ans, min(min(lf[i][j], rt[i][j]), min(up[i][j], dn[i][j])));
				}
		return ans;
	}
	//765. Couples Holding Hands 
	int minSwapsCouples(vector<int>& row) {
		for (int& i : row)
			i /= 2;
		unsigned len = 0;
		for (auto it = row.begin(); it != row.end(); it += 2)
			if (*it != *(it + 1)) {
				auto toswap = find(it + 2, row.end(), *it);
				iter_swap(it + 1, toswap);
				++len;
			}
		return len;
	}

	void visit_findItinerary(vector<string> &path, string cur, map<string, vector<string>>&m)
	{
		while (!m[cur].empty())
		{
			string c = m[cur].back();
			m[cur].pop_back();
			visit_findItinerary(path, c, m);
		}
		path.push_back(cur);
	}
	//332. Reconstruct Itinerary
	vector<string> findItinerary(vector<pair<string, string>> tickets) {
		sort(tickets.begin(), tickets.end());
		map<string, vector<string>> m;
		for (auto&i : tickets)
			m[i.first].push_back(i.second);
		vector<string> path;
		visit_findItinerary(path, "JFK", / s m);
		reverse(path.begin(), path.end());
		return path;
	}
};
int main()
{
	Solution sol;
	vector<pair<string, string>> ticks{ { "MUC", "LHR" },{ "JFK", "MUC" },{ "SFO", "SJC" },{ "LHR", "SFO" } };
	auto r = sol.findItinerary(ticks);
	for (auto &i : r)
		cout << i << " ";
	cout << endl;
	return 0;
}