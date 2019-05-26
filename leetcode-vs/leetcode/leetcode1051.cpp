#include <queue>
#include <assert.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

class Solution
{
public:
	//1051. Height Checker
	int heightChecker(vector<int>& a) {
		int n = a.size();
		auto b = a;
		sort(a.begin(), a.end());
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			if (a[i] != b[i]) ans++;
		}
		return ans;
	}

	//1052. Grumpy Bookstore Owner
	int maxSatisfied(vector<int>& a, vector<int>& b, int x) {
		int n = a.size();
		vector<int> left(n), right(n);
		int u = 0;
		for (int i = 0; i < n; ++i)
		{
			left[i] = u;
			if (b[i] == 1) u += 0;
			else u += a[i];
		}
		u = 0;
		for (int i = n - 1; i >= 0; --i)
		{
			right[i] = u;
			if (b[i] == 1) u += 0;
			else u += a[i];
		}
		u = 0;
		int ans = 0;
		for (int i = 0; i < n; ++i)
		{
			u += a[i];
			if (i >= x) u -= a[i - x];
			int v = u + right[i];
			if (i - x + 1 >= 0)
				v += left[i - x + 1];
			ans = max(ans, v);
		}
		return ans;
	}

	//1053. Previous Permutation With One Swap
	vector<int> prevPermOpt1(vector<int>& a) {
		int n = a.size();
		map<int, int> pre;
		for (int i = n - 1; i >= 0; --i)
		{
			auto it = pre.lower_bound(a[i]);
			if (it != pre.begin() && (it == pre.end() || it->first > a[i])) --it;
			if (it != pre.begin() && (it == pre.end() || it->first == a[i])) --it;
			if (it != pre.end() && it->first < a[i])
			{
				int idx = it->second;
				swap(a[i], a[idx]);
				return a;
			}
			pre[a[i]] = i;
		}
		return a;
	}

	//1054. Distant Barcodes
	vector<int> rearrangeBarcodes(vector<int>& a) {
		priority_queue<pair<int, int>> pq;
		map<int, int> m;
		for (auto& e : a) m[e] ++;
		for (auto& it : m)
		{
			pq.push({ it.second, it.first });
		}

		int n = a.size();
		vector<int> ans(n);
		for (int i = 0; i < n; ++i)
		{
			if (i == 0)
			{
				auto u = pq.top(); pq.pop();
				a[i] = u.second;
				if (--u.first > 0) pq.push(u);
			}
			else
			{
				auto mx = pq.top(); pq.pop();
				if (a[i - 1] == mx.second)
				{
					if (pq.empty()) assert(0);
					auto smx = pq.top(); pq.pop();
					a[i] = smx.second;
					if (--smx.first > 0) pq.push(smx);
					pq.push(mx);
				}
				else
				{
					a[i] = mx.second;
					if (--mx.first > 0) pq.push(mx);
				}
			}
		}
		return a;
	}
};
int main()
{
	return 0;
}
