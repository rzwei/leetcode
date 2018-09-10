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
#include <stack>
#include <list>
#include <climits>
using namespace std;

//901. Online Stock Span
class StockSpanner {
public:
	StockSpanner() {

	}
	vector<int> a, b;
	int tot = 0;
	int next(int x) {
		if (a.empty() || x < a.back())
		{
			a.push_back(x);
			b.push_back(tot - 1);
			tot++;
			return 1;
		}
		else
		{
			int ans = 1;
			int i = tot - 1;
			while (i >= 0 && x >= a[i])
			{
				ans += i - b[i];
				i = b[i];
			}
			a.push_back(x);
			b.push_back(i);
			tot++;
			return ans;
		}
	}
};
//900. RLE Iterator
class RLEIterator {
public:
	int i = 0;
	vector<int> a;
	RLEIterator(vector<int> A) : a(A) {

	}

	int next(int n) {
		while (i < a.size() && n >= a[i])
		{
			n -= a[i];
			a[i] = 0;
			if (n == 0) break;
			i += 2;
		}
		if (i >= a.size()) return -1;
		a[i] -= n;
		return a[i + 1];
	}
};
class Solution {
public:
	int count_lower(char c, vector<int> &d)
	{
		int ans = 0;
		for (int i = 0; i < c - '0'; ++i)
			if (d[i]) ans++;
		return ans;
	}
	int dfs_902(int u, string &n, vector<int> &ds, vector<int> &m)
	{
		if (u == n.size() - 1)
		{
			int ans = 0;
			for (int i = 0; i <= n[u] - '0'; ++i)
				ans += ds[i];
			return ans;
		}
		int lower = count_lower(n[u], ds);
		int ans = lower * m[n.size() - u - 1];
		if (ds[n[u] - '0'])
			ans += dfs_902(u + 1, n, ds, m);
		return ans;
	}
	//902. Numbers At Most N Given Digit Set
	int atMostNGivenDigitSet(vector<string>& D, int N) {
		vector<int> ss(10);
		for (auto &s : D) {
			ss[s[0] - '0'] = 1;
		}
		string n = to_string(N);
		int len = n.size();
		vector<int> m(len + 1);
		m[0] = 1;
		for (int i = 1; i <= len; ++i)
			m[i] = m[i - 1] * D.size();
		int ans = 0;
		for (int l = len - 1; l >= 1; --l)
		{
			ans += m[l];
		}
		ans += dfs_902(0, n, ss, m);
		return ans;
	}
};
int main()
{
	return 0;
}