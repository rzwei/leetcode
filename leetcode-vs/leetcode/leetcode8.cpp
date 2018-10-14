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
typedef long long ll;
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

//911. Online Election
class TopVotedCandidate {
public:
	map<int, int> win;
	TopVotedCandidate(vector<int> a, vector<int> t) {
		map<int, int> cnt;
		int n = a.size();
		int mx = 0, p = -1;
		for (int i = 0; i < n; ++i)
		{
			if (++cnt[a[i]] >= mx)
			{
				mx = cnt[a[i]];
				p = a[i];
			}
			win[t[i]] = p;
		}
	}

	int q(int t) {
		auto it = win.upper_bound(t);
		--it;
		return it->second;
	}
};

struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

//919. Complete Binary Tree Inserter
class CBTInserter {
public:
	queue<TreeNode *> cur;
	TreeNode *ret;
	int maxdep(TreeNode *u)
	{
		if (!u) return 0;
		return max(maxdep(u->left), maxdep(u->right)) + 1;
	}
	CBTInserter(TreeNode* root) {
		ret = root;
		int dep = maxdep(root);
		queue<TreeNode *> q;
		q.push(root);
		int level = 0;
		while (!q.empty())
		{
			int size = q.size();
			level++;
			while (size--)
			{
				auto u = q.front();
				q.pop();
				if (level == dep - 1 && (u->left == nullptr || u->right == nullptr) || level == dep)
				{
					if (level == dep - 1)
					{
						if (u->left == nullptr)
							cur.push(u);
						if (u->right == nullptr)
							cur.push(u);
					}
					else
					{
						cur.push(u);
						cur.push(u);
					}
				}
				if (u->left) q.push(u->left);
				if (u->right) q.push(u->right);
			}
			if (level == dep) break;
		}
	}

	int insert(int v) {
		auto u = cur.front();
		cur.pop();
		if (u->left == nullptr)
		{
			u->left = new TreeNode(v);
			cur.push(u->left);
			cur.push(u->left);
			return u->val;
		}
		else if (u->right == nullptr)
		{
			u->right = new TreeNode(v);
			cur.push(u->right);
			cur.push(u->right);
			return u->val;
		}
	}

	TreeNode* get_root() {
		return ret;
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

	bool judge(ll n)
	{
		ll v2 = 0, t = n;
		while (t)
		{
			v2 = v2 * 10 + t % 10;
			t /= 10;
		}
		return v2 == n;
	}
	ll create(ll n)
	{
		ll ans = n;
		while (n)
		{
			ans = ans * 10 + n % 10;
			n /= 10;
		}
		return ans;
	}
	ll create2(ll n)
	{
		ll ans = n;
		n /= 10;
		while (n)
		{
			ans = ans * 10 + n % 10;
			n /= 10;
		}
		return ans;
	}
	//906. Super Palindromes
	int superpalindromesInRange(string L, string R) {
		vector<ll> a;
		int const BOUND = 1e9;
		for (int i = 1; i <= 3e4; ++i)
		{
			auto v1 = create(i), v2 = create2(i);
			if (v1 >= BOUND && v2 >= BOUND) break;
			if (v1 < BOUND &&  judge(v1 * v1)) a.push_back(v1 * v1);
			if (v2 < BOUND && judge(v2 * v2)) a.push_back(v2 * v2);
		}
		auto l = stoll(L), r = stoll(R);
		int ans = 0;
		//for (ll e : a)
		//	cout << e << endl;
		for (ll e : a)
			if (l <= e && e <= r)
				ans++;
		return ans;
	}

	// 909. Snakes and Ladders
	int snakesAndLadders(vector<vector<int>>& a) {
		int n = a.size();
		int const maxn = INT_MAX / 8;
		vector<int> s(n * n + 1);
		int cur = 1;
		for (int i = n - 1, f = 0; i >= 0; --i, f = 1 - f)
		{
			if (f == 0) for (int j = 0; j < n; ++j)
				s[cur++] = a[i][j];
			else for (int j = n - 1; j >= 0; --j)
				s[cur++] = a[i][j];
		}
		queue<int> q;
		vector<bool> vis(n * n + 1);
		vis[1] = 1;
		q.push(1);
		vector<int> dist(n * n + 1, maxn);
		dist[1] = 0;
		while (!q.empty())
		{
			int u = q.front();
			q.pop();
			for (int i = 1; i <= 6 && u + i <= n * n; ++i)
			{
				int v = u + i;
				if (s[v] != -1)
					v = s[v];
				if (dist[u] + 1 < dist[v])
				{
					dist[v] = dist[u] + 1;
					if (!vis[v])
					{
						vis[v] = 1;
						q.push(v);
					}
				}
			}
		}
		if (dist[n * n] == maxn) return -1;
		return dist[n * n];
	}
	//908. Smallest Range I
	int smallestRangeI(vector<int>& A, int K) {
		int mi = INT_MAX, mx = INT_MIN;
		for (int e : A)
		{
			mi = min(mi, e);
			mx = max(mx, e);
		}
		if (mx - mi <= K * 2)  return 0;
		else return mx - mi - K * 2;
	}

	//910. Smallest Range II
	int smallestRangeII(vector<int>& A, int K) {
		sort(A.begin(), A.end());
		int n = A.size();
		if (n == 1) return 0;
		int i = 0, j = n - 1;
		int ans = A.back() - A[0];
		for (int i = 0; i < n - 1; ++i)
		{
			int lx = A[i] + K, rx = A.back() - K;
			int mx = max(lx, rx);
			int li = A[0] + K, ri = A[i + 1] - K;
			int mi = min(li, ri);
			ans = min(ans, mx - mi);
		}
		return ans;
	}
	int gcd(int a, int b)
	{
		if (a < b) swap(a, b);
		int t;
		while (b)
		{
			t = b;
			b = a % b;
			a = t;
		}
		return a;
	}
	//914. X of a Kind in a Deck of Cards 
	bool hasGroupsSizeX(vector<int>& deck) {
		map<int, int> cnt;
		for (int e : deck) cnt[e]++;
		int pv = -1;
		for (auto it = cnt.begin(); it != cnt.end(); ++it)
		{
			if (pv == -1) pv = it->second;
			else pv = gcd(pv, it->second);
		}
		return pv >= 2;
	}

	//915. Partition Array into Disjoint Intervals
	int partitionDisjoint(vector<int>& A) {
		int n = A.size();
		vector<int> mis(n);
		mis[n - 1] = A[n - 1];
		for (int i = n - 2; i >= 0; --i)
		{
			mis[i] = min(mis[i + 1], A[i]);
		}
		int mx = -1;
		for (int i = 0; i + 1 < n; ++i)
		{
			mx = max(mx, A[i]);
			if (mx <= mis[i + 1])
				return i + 1;
		}
		return -1;
	}

	//916. Word Subsets
	vector<string> wordSubsets(vector<string>& A, vector<string>& B) {
		int n = B.size();
		vector<int> bb(26);
		for (int i = 0; i < n; ++i)
		{
			vector<int> cnt(26);
			for (int j = 0; j < B[i].size(); ++j)
				cnt[B[i][j] - 'a']++;
			for (int j = 0; j < 26; ++j)
				bb[j] = max(bb[j], cnt[j]);
		}
		vector<string> ans;
		int m = A.size();
		for (int i = 0; i < m; ++i)
		{
			vector<int> cnt(26);
			for (char c : A[i])
				cnt[c - 'a']++;
			bool f = true;
			for (int i = 0; f && i < 26; ++i)
			{
				if (cnt[i] < bb[i])
					f = false;
			}
			if (f) ans.push_back(A[i]);
		}
		return ans;
	}

	//913. Cat and Mouse
	int catMouseGame(vector<vector<int>>& graph) {
		int n = graph.size();
		int const DRAW = 0, MOUSE = 1, CAT = 2;
		vector<vector<vector<int>>> color(n, vector<vector<int>>(n, vector<int>(3)));
		auto outdegree = color;
		//cat, mouse, turn, win, mouse -> 0;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				outdegree[i][j][0] = graph[j].size();
				outdegree[i][j][1] = graph[i].size();
				for (int k : graph[i])
				{
					if (k == 0)
					{
						outdegree[i][j][1]--;
						break;
					}
				}
			}

		}
		queue<vector<int>> q;

		for (int k = 1; k < n; ++k)
		{
			for (int m = 0; m < 2; ++m)
			{
				color[k][0][m] = 1;         //mouse
				q.push({ k, 0, m, 1 });
				color[k][k][m] = 2;         //cat
				q.push({ k, k, m, 2 });
			}
		}

		while (!q.empty())
		{
			auto u = q.front();
			q.pop();
			int cat = u[0], mouse = u[1], turn = u[2], c = u[3];
			if (cat == 2 && mouse == 1 && turn == 0) return c;
			int pre = 1 - turn;
			for (int v : graph[pre == 1 ? cat : mouse])
			{
				int preCat = pre == 1 ? v : cat;
				int preMouse = pre == 1 ? mouse : v;
				if (preCat == 0) continue;
				if (color[preCat][preMouse][pre] > 0) continue;
				if (pre == 1 && c == 2 || pre == 0 && c == 1
					|| --outdegree[preCat][preMouse][pre] == 0) {
					color[preCat][preMouse][pre] = c;
					q.push({ preCat, preMouse, pre, c });
				}
			}
		}
		return color[2][1][0];
	}

	//917. Reverse Only Letters
	string reverseOnlyLetters(string S) {
		int i = 0, j = S.size() - 1;
		while (i < j)
		{
			while (i < j && !isalpha(S[i])) i++;
			while (i < j && !isalpha(S[j])) j--;
			if (i >= j) break;
			swap(S[i], S[j]);
			i++;
			j--;
		}
		return S;
	}

	//918. Maximum Sum Circular Subarray
	int maxSubarraySumCircular(vector<int>& A) {
		int n = A.size();
		vector<int> sums(n + n + 1);
		int ans = INT_MIN;
		for (int i = 0; i < n; ++i)
		{
			sums[i + 1] = sums[i + n + 1] = A[i];
			ans = max(ans, A[i]);
		}
		for (int i = 1; i <= n + n; ++i)
			sums[i] += sums[i - 1];
		deque<int> q;
		q.push_back(0);
		for (int i = 1; i <= n + n; ++i)
		{
			while (!q.empty() && sums[q.back()] > sums[i]) q.pop_back();
			while (!q.empty() && i - q.front() > n) q.pop_front();
			if (!q.empty())
				ans = max(ans, sums[i] - sums[q.front()]);
			q.push_back(i);
		}
		return ans;
	}

	//920. Number of Music Playlists
	int numMusicPlaylists(int N, int L, int K) {
		long mod = 1e9 + 7;
		vector <vector<long>> dp(N + 1, vector<long>(L + 1));
		vector<long> factorial(N + 1);
		factorial[0] = 1;
		for (int i = 1; i <= N; ++i)
			factorial[i] = factorial[i - 1] * i % mod;

		for (int i = K + 1; i <= N; ++i)
			for (int j = i; j <= L; ++j)
				if ((i == j) || (i == K + 1))
					dp[i][j] = factorial[i];
				else
					dp[i][j] = (dp[i - 1][j - 1] * i + dp[i][j - 1] * (i - K)) % mod;
		return (int)dp[N][L];
	}

	//922. Sort Array By Parity II
	vector<int> sortArrayByParityII(vector<int>& A) {
		int n = A.size(), i = 0, j = 1;
		for (int i = 0; i < n; i += 2)
		{
			if (A[i] % 2)
			{
				while (j < n && A[j] % 2) j += 2;
				swap(A[i], A[j]);
				j += 2;
			}
		}
		return A;
	}

	//921. Minimum Add to Make Parentheses Valid
	int minAddToMakeValid(string S) {
		int cnt = 0, ans = 0;
		for (char c : S)
		{
			if (c == '(') cnt++;
			else
			{
				if (cnt > 0) cnt--;
				else
					ans++;
			}
		}
		ans += cnt;
		return ans;
	}

	//923. 3Sum With Multiplicity
	int threeSumMulti(vector<int>& A, int target) {
		int const N = 100, mod = 1e9 + 7;
		vector<long long> cnt(N + 1);
		for (int e : A) cnt[e]++;
		long long ans = 0;
		for (int a = 0; a <= N; ++a)
		{
			for (int b = a; b <= N; ++b)
			{
				int c = target - a - b;
				if (b <= c && c <= N)
				{
					if (a == b && a != c)
					{
						long long v = cnt[a];
						ans = (ans + (v * (v - 1)) / 2 * cnt[c]) % mod;
					}
					else if (a == b && a == c)
					{
						long long v = cnt[a];
						ans = (ans + v * (v - 1) * (v - 2) / 6) % mod;
					}
					else if (a != b && b == c)
					{
						long long v = cnt[b];
						ans = (ans + v * (v - 1) / 2 * cnt[a]) % mod;
					}
					else if (a != b && a != c)
					{
						ans = (ans + cnt[a] * cnt[b] * cnt[c]) % mod;
					}
				}
			}
		}
		return ans;
	}

	void dfs_924(int u, vector<vector<int>> &g, vector<int> &color, int c)
	{
		color[u] = c;
		for (int v = 0; v < g.size(); ++v)
		{
			if (g[u][v] && color[v] == -1)
				dfs_924(v, g, color, c);
		}
	}

	//924. Minimize Malware Spread
	int minMalwareSpread(vector<vector<int>>& g, vector<int>& init) {
		int n = g.size();
		vector<int>color(n, -1);
		int c = 0;
		for (int i = 0; i < n; ++i)
			if (color[i] == -1) dfs_924(i, g, color, c++);
		vector<int> colorsize(c);
		for (int x : color) colorsize[x]++;
		vector<int> colorcnt(n);
		for (int e : init) colorcnt[color[e]]++;
		int ans = -1;
		for (int e : init)
		{
			int x = color[e];
			if (colorcnt[x] == 1)
			{
				if (ans == -1) ans = e;
				else if (colorsize[x] > colorsize[color[ans]]) ans = e;
				else if (colorsize[x] == colorsize[color[ans]] && e < ans) ans = e;
			}
		}
		if (ans == -1)
		{
			for (int e : init) if (ans == -1) ans = e;
			else if (e < ans) ans = e;
		}
		return ans;
	}
};
int main()
{
	Solution sol;
	return 0;
}