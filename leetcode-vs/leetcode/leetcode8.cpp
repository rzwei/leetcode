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
	stack<pair<int, int>> stk;
	StockSpanner() {

	}

	int next(int price) {
		int w = 1;
		while (!stk.empty() && stk.top().first <= price)
		{
			w += stk.top().second;
			stk.pop();
		}
		stk.push({ price, w });
		return w;
	}
};
//900. RLE Iterator
class RLEIterator {
public:
	vector<int> a;
	int i = 0;
	RLEIterator(vector<int> A) :a(A) {

	}

	int next(int n) {
		while (i < a.size() && n > a[i])
		{
			n -= a[i];
			i += 2;
		}
		if (i >= a.size()) return -1;
		a[i] -= n;
		return a[i + 1];
	}
};
class Solution {
public:
	//902. Numbers At Most N Given Digit Set
	int atMostNGivenDigitSet(vector<string>& D, int N) {
		int k = D.size();
		vector<char> digits(k);
		for (int i = 0; i < k; ++i)
			digits[i] = D[i][0];

		string s = to_string(N);
		int n = s.size();
		vector<int> powk(n + 1);
		powk[0] = 1;
		for (int i = 1; i <= n; ++i)
			powk[i] = powk[i - 1] * k;

		int ans = 0;
		for (int i = 1; i < n; ++i)
		{
			ans += powk[i];
		}
		for (int i = 0; i < n; ++i)
		{
			int cnt = 0;
			bool same = false;
			for (char c : digits)
			{
				if (c < s[i]) cnt++;
				if (c == s[i]) same = true;
				if (c >= s[i]) break;
			}
			ans += cnt * powk[n - 1 - i];
			if (same == false) break;
			if (i == n - 1)
				ans += 1;
		}
		return ans;
	}

	//903. Valid Permutations for DI Sequence
	int numPermsDISequence(string S) {
		const int mod = 1e9 + 7;
		int n = S.size();
		vector<vector<int>> dp(n + 1, vector<int>(n + 1));
		for (int i = 0; i < n; ++i)
		{
			dp[0][i] = 1;
		}
		for (int i = 1; i <= n; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				if (S[i - 1] == 'D')
				{
					for (int k = j; k < i; ++k)
					{
						dp[i][j] = (dp[i][j] + dp[i - 1][k]) % mod;
					}
				}
				else
				{
					for (int k = 0; k < j; ++k)
					{
						dp[i][j] = (dp[i][j] + dp[i - 1][k]) % mod;
					}
				}
			}
		}
		int ans = 0;
		for (int e : dp[n])
			ans = (ans + e) % mod;
		return ans;
	}

	//905. Sort Array By Parity
	vector<int> sortArrayByParity(vector<int>& A) {
		vector<int> a, b;
		for (int e : A)
		{
			if (e % 2 == 0)
				a.push_back(e);
			else
				b.push_back(e);
		}
		for (int e : b)
			a.push_back(e);
		return a;
	}

	//904. Fruit Into Baskets 
	int totalFruit(vector<int>& tree) {
		map<int, int> cnt;
		int n = tree.size();
		int ans = 0, j = 0;
		for (int i = 0; i < n; ++i)
		{
			cnt[tree[i]]++;
			if (cnt.size() == 2)
				ans = max(ans, i - j + 1);
			else if (cnt.size() > 2)
			{
				while (cnt.size() > 2)
				{
					if (--cnt[tree[j]] == 0)
						cnt.erase(tree[j]);
					j++;
				}
				ans = max(ans, i - j + 1);
			}
		}
		return ans;
	}

	//907. Sum of Subarray Minimums 
	int sumSubarrayMins(vector<int>& A) {
		int n = A.size();
		vector<int> l(n, -1), r(n, n);
		stack<int> stk;
		for (int i = 0; i < n; ++i)
		{
			while (!stk.empty() && A[stk.top()] > A[i])
			{
				r[stk.top()] = i;
				stk.pop();
			}
			stk.push(i);
		}
		while (!stk.empty())
		{
			r[stk.top()] = n;
			stk.pop();
		}
		for (int i = n - 1; i >= 0; --i)
		{
			while (!stk.empty() && A[stk.top()] >= A[i])
			{
				l[stk.top()] = i;
				stk.pop();
			}
			stk.push(i);
		}
		while (!stk.empty())
		{
			l[stk.top()] = -1;
			stk.pop();
		}
		typedef long long ll;
		int const mod = 1e9 + 7;

		ll ans = 0;

		for (int i = 0; i < n; ++i)
		{
			ll left = i - l[i], right = r[i] - i;
			ans = (ans + A[i] * left * right) % mod;
		}
		return ans;
	}

	//906. Super Palindromes
	int superpalindromesInRange(string L, string R) {
		typedef long long ll;
		vector<ll> a = { 0,1,2,3,11,22,101,111,121,202,212,1001,1111,2002,10001,10101,10201,11011,11111,11211,20002,20102,100001,101101,110011,111111,200002,1000001,1001001,1002001,1010101,1011101,1012101,1100011,1101011,1102011,1110111,1111111,2000002,2001002,10000001,10011001,10100101,10111101,11000011,11011011,11100111,11111111,20000002,100000001,100010001,100020001,100101001,100111001,100121001,101000101,101010101,101020101,101101101,101111101,110000011,110010011,110020011,110101011,110111011,111000111,111010111,111101111,111111111,200000002,200010002,1000000001,1000110001,1001001001,1001111001,1010000101,1010110101,1011001101,1011111101,1100000011,1100110011,1101001011,1101111011,1110000111,1110110111,1111001111,2000000002 };

		auto left = stoll(L), right = stoll(R);
		int ans = 0;
		for (ll e : a)
		{
			if (left <= e * e && e * e <= right)
				ans++;
		}
		return ans;
	}
};
int main()
{
	return 0;
}