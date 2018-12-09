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
#include <numeric>
using namespace std;
typedef long long ll;

class Solution
{
public:
	bool less(string &a, string &b, vector<int> &order)
	{
		int i = 0, j = 0;
		int la = a.size(), lb = b.size();
		int l = max(la, lb);
		while (i < l)
		{
			int ai = 0, bj = 0;
			if (i < la) ai = order[a[i] - 'a'];
			if (i < lb) bj = order[b[i] - 'a'];
			if (ai == bj) i++;
			else return ai < bj;
		}
		return true;
	}

	//953. Verifying an Alien Dictionary
	bool isAlienSorted(vector<string>& words, string order) {
		vector<int> old(26);
		for (int i = 0; i < order.size(); ++i)
		{
			old[order[i] - 'a'] = i;
		}
		int n = words.size();
		for (int i = 1; i < n; ++i)
		{
			if (!less(words[i - 1], words[i], old)) return false;
		}
		return true;
	}

	//954. Array of Doubled Pairs
	bool canReorderDoubled(vector<int>& a) {
		if (accumulate(a.begin(), a.end(), 0) % 3) return false;
		auto cmp = [](int a, int b) {
			return abs(a) < abs(b);
		};
		sort(a.begin(), a.end(), cmp);
		int n = a.size();
		int cur = 0;
		map<int, int> cnt;
		for (int i = 0; i < n; ++i)
		{
			if (a[i] % 2 == 0 && cnt[a[i] / 2] > 0)
			{
				cur++;
				cnt[a[i] / 2]--;
			}
			else
			{
				cnt[a[i]]++;
			}
		}
		return cur >= n / 2;
	}

	//955. Delete Columns to Make Sorted II
	int minDeletionSize(vector<string>& a) {
		int ans = 0;
		int len = a[0].size(), n = a.size();
		vector<char> last(n);
		for (int j = 0; j < len; ++j)
		{
			bool f = false, con = false;
			for (int i = 1; !f && i < n; ++i)
			{
				if (a[i][j] < a[i - 1][j]) f = true;
			}
			if (f) ans += f;
			else
			{
				for (int i = 0; i < n; ++i) last[i] = a[i][j];
				j++;
				while (j < len)
				{
					bool f = false;
					for (int i = 1; !f && i < n; ++i)
					{
						if (last[i] == last[i - 1])
						{
							if (a[i][j] < a[i - 1][j]) f = true;
						}
					}
					if (f) ans++;
					else
					{
						for (int i = 0; i < n; ++i) last[i] = a[i][j];
					}
					j++;
				}
				break;
			}
		}
		return ans;
	}

	//956. Tallest Billboard
	int tallestBillboard(vector<int>& a) {
		int const offset = accumulate(a.begin(), a.end(), 0), maxn = 2 * offset + 1;
		int n = a.size();
		vector<int> dp(maxn, INT_MIN), pre(maxn, INT_MIN);
		pre[offset] = 0;
		for (int e : a)
		{
			for (int v = 0; v < maxn; ++v)
			{
				dp[v] = pre[v];
				if (v - e >= 0) dp[v] = max(dp[v], pre[v - e] + e);
				if (v + e < maxn) dp[v] = max(dp[v], pre[v + e] + e);
			}
			pre = dp;
		}
		return pre[offset] / 2;
	}
};

int main()
{

}