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
};
int main()
{
	return 0;
}