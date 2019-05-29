#include <queue>
#include <assert.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

template<typename T, typename V>
T first_less_than(T f, T l, V v)
{
    auto it = lower_bound(f, l, v);
    return it == f ? l : --it;
}
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

		auto first_less_than = [&](int val) {
			auto it = pre.lower_bound(val);
			if (it == pre.begin()) return pre.end();
			return --it;
		};

		for (int i = n - 1; i >= 0; --i)
		{
			auto it = first_less_than(a[i]);
			if (it != pre.end())
			{
				swap(a[i], a[it->second]);
				return a;
			}
			pre[a[i]] = i;
		}
		return a;
	}

	//1054. Distant Barcodes
    vector<int> rearrangeBarcodes(vector<int>& a) {
        int n = a.size();
        int const maxn = 10000 + 1;
        vector<int> cnt(maxn);
        int max_n = 0, max_cnt = 0;
        for (int &e : a)
        {
            if (++ cnt[e] > max_cnt)
            {
                max_cnt = cnt[e];
                max_n = e;
            }
        }
        int pos = 0;
        for (int i = 0; i < maxn; ++i)
        {
            int u = (i == 0 ? max_n : i);
            while (cnt[u] > 0)
            {
                a[pos] = u;
                pos += 2;
                if (pos >= n) pos = 1;
                cnt[u] --;
            }
        }
        return a;
    }
        
};
int main()
{
    Solution sol;
    vector<int> a;
    a = { 3, 1, 1, 3 };
    auto r = sol.prevPermOpt1(a);
    for (auto &e : r) cout << e << " ";
    cout << endl;
	return 0;
}
