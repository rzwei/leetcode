#include <sstream>
#include <functional>
#include <iostream>
#include <bitset>
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

//933. Number of Recent Calls
class RecentCounter {
public:
	vector<int> a;
	RecentCounter() {

	}

	int ping(int t) {
		a.push_back(t);
		return a.end() - lower_bound(a.begin(), a.end(), t - 3000);
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

	//925. Long Pressed Name
	bool isLongPressedName(string name, string typed) {
		int i = 0, j = 0, n = name.size(), m = typed.size();
		while (j < m && i < n)
		{
			while (j > 0 && j < m && typed[j] != name[i] && typed[j] == typed[j - 1]) j++;
			if (name[i] == typed[j])
			{
				i++;
				j++;
			}
			else return false;
		}
		return i == n;
	}

	//926. Flip String to Monotone Increasing
	int minFlipsMonoIncr(string S) {
		int ans = INT_MAX, n = S.size();
		int left = 0, right = 0;
		for (int i = 0; i < n; ++i)
			right += S[i] == '0';
		for (int i = 0; i < n; ++i)
		{
			ans = min(ans, left + right);
			if (S[i] == '0') right--;
			else left++;
		}
		ans = min(ans, left + right);
		return ans;
	}

	//927. Three Equal Parts
	vector<int> threeEqualParts(vector<int>& A) {
		int cnt = 0;
		for (int e : A) cnt += e == 1;
		int n = A.size();
		if (cnt == 0) return { 0, n - 1 };
		if (cnt % 3) return { -1, -1 };
		int k = cnt / 3;
		vector<int> st(3, -1), ed(3, -1);
		int f = 0;
		for (int i = 0; i < n; ++i)
		{
			if (A[i] == 1)
			{
				if (f % k == 0)
					st[f / k] = i;
				f++;
				if (f % k == 0)
					ed[(f - 1) / k] = i;
			}
		}

		for (int e : st) if (e == -1) return { -1, -1 };


		for (int i = 1; i < 3; ++i)
		{
			if (ed[i] - st[i] != ed[i - 1] - st[i - 1])
				return { -1, -1 };
		}

		int endzeros = n - 1 - ed[2];
		int len = ed[0] - st[0] + 1;

		for (int i = 1; i < 3; ++i) if (st[i] - ed[i - 1] - 1 < endzeros) return { -1, -1 };

		for (int i = 0; i < len; ++i)
		{
			int val = A[st[0] + i];
			for (int j = 1; j < 3; ++j) if (A[st[j] + i] != val) return { -1, -1 };
		}

		return { ed[0] + endzeros, ed[1] + endzeros + 1 };
	}

	void bfs_928(int u, vector<int> &color, vector<vector<int>> &g)
	{
		queue<int> q;
		q.push(u);
		color[u] = 1;
		while (!q.empty())
		{
			int u = q.front(); q.pop();
			for (int v = 0; v < g.size(); ++v)
			{
				if (g[u][v])
				{
					if (color[v] == 0)
					{
						color[v] = 1;
						q.push(v);
					}
				}
			}
		}
	}
	//928. Minimize Malware Spread II
	int minMalwareSpreadII(vector<vector<int>>& graph, vector<int>& initial) {
		int n = graph.size();
		sort(initial.begin(), initial.end());
		int ans = INT_MAX, idx = -1;
		vector<int> bk(n);
		for (int i = 0; i < initial.size(); i++)
		{
			int e = initial[i];
			bk = graph[e];
			fill(graph[e].begin(), graph[e].end(), 0);
			vector<int> color(n);
			for (int j = 0; j < initial.size(); ++j)
			{
				if (i == j) continue;
				if (color[initial[j]] == 0)
				{
					bfs_928(initial[j], color, graph);
				}
			}
			int cur = 0;
			for (int i = 0; i < n; ++i) cur += color[i];

			cur -= color[e];

			// cout << initial[i] << " " << cur << endl;

			if (cur < ans)
			{
				ans = cur;
				idx = initial[i];
			}
			graph[e] = bk;
		}
		return idx;
	}
	//929. Unique Email Addresses
	int numUniqueEmails(vector<string>& emails) {
		set<string> ans;
		for (string &s : emails)
		{
			string n;
			bool f = true;
			for (int i = 0; i < s.size(); ++i)
			{
				if (f && s[i] == '.') continue;
				if (s[i] == '+')
				{
					int j = i + 1;
					while (j < s.size() && s[j] != '@') j++;
					if (s[j] == '@')
						j--;
					i = j;
				}
				else
				{
					n.push_back(s[i]);
					if (s[i] == '@') f = false;
				}
			}
			ans.insert(n);
		}
		// for (auto &s : ans) cout << s << endl;
		return ans.size();
	}

	//930. Binary Subarrays With Sum
	int numSubarraysWithSum(vector<int>& A, int S) {
		map<int, int> cnt;
		int n = A.size(), ans = 0, cur = 0;
		cnt[0] = 1;
		for (int i = 0; i < n; ++i)
		{
			cur += A[i];
			ans += cnt[cur - S];
			cnt[cur]++;
		}
		return ans;
		//int ans = 0;
		//int n = A.size();
		//vector<int> b(n);
		//int cur = 0;
		//for (int i = 0; i < n; ++i)
		//{
		//	if (A[i] == 0)
		//	{
		//		cur++;
		//		b[i] = cur;
		//	}
		//	else
		//	{
		//		b[i] = cur;
		//		cur = 0;
		//	}
		//}
		//cur = 0;
		//int j = 0;
		//for (int i = 0; i < n; ++i)
		//{
		//	cur += A[i];
		//	while (cur > S)
		//	{
		//		cur -= A[j++];
		//	}
		//	if (cur == S && j <= i)
		//	{
		//		while (j < i && A[j] == 0) j++;
		//		if (A[j] != 0)
		//			ans += b[j] + 1;
		//		else
		//			ans += b[j];
		//	}
		//}
		//return ans;
	}

	//int dfs_931(int u, int pre, vector<vector<int>>& a, int n, int m, vector<vector<int>> &memo)
	//{
	//	if (u == n) return 0;
	//	if (memo[u][pre] != -1) return memo[u][pre];
	//	int ans = INT_MAX;
	//	for (int d = -1; d <= 1; d++)
	//	{
	//		int np = pre + d;
	//		if (0 <= np && np < m)
	//		{
	//			ans = min(ans, dfs_931(u + 1, np, a, n, m, memo) + a[u][np]);
	//		}
	//	}
	//	memo[u][pre] = ans;
	//	return ans;
	//}
	//931. Minimum Falling Path Sum
	int minFallingPathSum(vector<vector<int>>& a) {
		//int n = a.size();
		//int m = a[0].size();
		//vector<vector<int>> memo(n, vector<int>(m, -1));
		//int ans = INT_MAX;
		//for (int i = 0; i < n; ++i)
		//{
		//	ans = min(ans, dfs_931(1, i, a, n, m, memo) + a[0][i]);
		//}
		//return ans;
		int n = a.size(), m = a[0].size();
		for (int i = n - 2; i >= 0; --i)
		{
			for (int j = 0; j < m; ++j)
			{
				int val = a[i + 1][j];
				if (j + 1 < m) val = min(val, a[i + 1][j + 1]);
				if (j - 1 >= 0) val = min(val, a[i + 1][j - 1]);
				a[i][j] += val;
			}
		}
		return *min_element(a[0].begin(), a[0].end());
	}
	//932. Beautiful Array
	vector<int> beautifulArray(int N) {
		vector<int> res = { 1 };
		while (res.size() < N) {
			vector<int> tmp;
			for (int i : res) if (i * 2 - 1 <= N) tmp.push_back(i * 2 - 1);
			for (int i : res) if (i * 2 <= N) tmp.push_back(i * 2);
			res = tmp;
		}
		return res;
	}

	//935. Knight Dialer
	int knightDialer(int N) {
		int n = 4, m = 3;
		vector<vector<int>> valid = { {1, 1, 1},{1, 1, 1},{1, 1, 1},{0, 1, 0}, };
		int dr[] = { 1,-1,1,-1,2,-2,2,-2 };
		int dc[] = { 2,2,-2,-2,1,1,-1,-1 };
		int const mod = 1e9 + 7;
		vector<vector<long long>> dp(n, vector<long long>(m));
		auto pre = dp;
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j)
				pre[i][j] = valid[i][j];
		for (int t = 0; t < N - 1; ++t)
		{
			for (auto it = dp.begin(); it != dp.end(); ++it)
				fill(it->begin(), it->end(), 0);
			for (int i = 0; i < n; ++i)
				for (int j = 0; j < m; ++j)
				{
					if (!valid[i][j]) continue;
					for (int d = 0; d < 8; ++d)
					{
						int nx = i + dr[d], ny = j + dc[d];
						if (0 <= nx && nx < n && 0 <= ny && ny < m && valid[nx][ny])
						{
							dp[nx][ny] = (dp[nx][ny] + pre[i][j]) % mod;
						}
					}
				}
			swap(dp, pre);
		}
		long long ans = 0;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				if (valid[i][j]) ans = (ans + pre[i][j]) % mod;
			}
		}
		return ans;
	}

	void bfs_934(int i, int j, vector<vector<int>> &a, int c)
	{
		static int dr[4] = { 0, 1, 0, -1 };
		static int dc[4] = { 1, 0, -1, 0 };
		int n = a.size(), m = a[0].size();
		queue<pair<int, int>> q;
		q.emplace(i, j);
		a[i][j] = c;
		while (!q.empty())
		{
			i = q.front().first, j = q.front().second;
			q.pop();
			for (int d = 0; d < 4; ++d)
			{
				int ni = i + dr[d], nj = j + dc[d];
				if (0 <= ni && ni < n && 0 <= nj && nj < m && a[ni][nj] == 1)
				{
					a[ni][nj] = c;
					q.emplace(ni, nj);
				}
			}
		}
	}

	//934. Shortest Bridge
	int shortestBridge(vector<vector<int>>& a) {
		static int dr[4] = { 0, 1, 0, -1 };
		static int dc[4] = { 1, 0, -1, 0 };
		int n = a.size(), m = a[0].size();
		vector<pair<int, int>> left, right;
		int c = 100;
		queue<pair<int, int>> q;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				if (a[i][j] == 1)
					bfs_934(i, j, a, c++);
			}
		}
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j)
				if (a[i][j] == 100) q.emplace(i, j);
		int ans = 0;
		while (!q.empty())
		{
			int size = q.size();
			ans++;
			while (size--)
			{
				auto i = q.front().first, j = q.front().second;
				q.pop();
				for (int d = 0; d < 4; ++d)
				{
					int ni = i + dr[d], nj = j + dc[d];
					if (0 <= ni && ni < n && 0 <= nj && nj < m)
					{
						if (a[ni][nj] == c - 1)
							return ans - 1;
						else if (a[ni][nj] == 0)
						{
							a[ni][nj] = 100;
							q.emplace(ni, nj);
						}
					}
				}
			}
		}
		return -1;
	}

	//936. Stamping The Sequence
	vector<int> movesToStamp(string stamp, string target) {
		class Node
		{
		public:
			set<int> made, todo;
			Node(set<int> &made_, set<int> &todo_) : made(made_), todo(todo_) {}
		};
		int m = stamp.size(), n = target.size();
		queue<int> q;
		vector<bool> vis(n);
		vector<int> ans;
		vector<Node> a;
		for (int i = 0; i <= n - m; ++i)
		{
			set<int> made, todo;
			for (int j = 0; j < m; ++j)
			{
				if (target[i + j] == stamp[j])
					made.insert(i + j);
				else
					todo.insert(i + j);
			}
			a.emplace_back(made, todo);
			if (todo.empty())
			{
				ans.push_back(i);
				for (int j = i; j < i + m; ++j)
				{
					if (!vis[j])
					{
						vis[j] = 1;
						q.push(j);
					}
				}
			}
		}
		while (!q.empty())
		{
			int i = q.front();
			q.pop();
			for (int j = max(0, i - m + 1); j <= min(n - m, i); ++j)
			{
				if (a[j].todo.count(i))
				{
					a[j].todo.erase(i);
					if (a[j].todo.empty())
					{
						ans.push_back(j);
						for (int m : a[j].made)
						{
							if (!vis[m])
							{
								q.push(m);
								vis[m] = 1;
							}
						}
					}
				}
			}
		}
		for (int i = 0; i < n; ++i)
			if (!vis[i]) return {};
		reverse(ans.begin(), ans.end());
		return ans;
	}

	//937. Reorder Log Files
	vector<string> reorderLogFiles(vector<string>& logs) {
		vector<string> ans;
		vector<string> tmp, tmp2;
		for (auto &s : logs)
		{
			bool f = true;
			for (int i = 0; i < s.size(); ++i)
			{
				if (s[i] == ' ')
				{
					if (isdigit(s[i + 1]))
					{
						f = false;
					}
					break;
				}
			}
			if (f)
				tmp.push_back(s);
			else tmp2.push_back(s);
		}
		auto cmp = [](string &a, string &b) {

			int i = 0, j = 0;
			while (a[i] != ' ') i++;
			while (b[j] != ' ') j++;
			return a.substr(i) < b.substr(j);
		};
		sort(tmp.begin(), tmp.end(), cmp);
		for (string &s : tmp2)
			tmp.push_back(s);
		return tmp;
	}

	//938. Range Sum of BST
	int rangeSumBST(TreeNode* root, int L, int R) {
		if (!root) return 0;
		int ans = 0;
		if (L <= root->val && root->val <= R)
			ans += root->val;
		ans += rangeSumBST(root->left, L, R);
		ans += rangeSumBST(root->right, L, R);
		return ans;
	}

	//939. Minimum Area Rectangle
	int minAreaRect(vector<vector<int>>& a) {
		int n = a.size();
		map<int, set<int>> plane;
		for (auto &e : a)
		{
			plane[e[0]].insert(e[1]);
		}
		int ans = INT_MAX;
		for (auto i = plane.begin(); i != plane.end(); ++i)
		{
			for (auto j = plane.begin(); j != i; ++j)
			{
				int last = -1, dx = i->first - j->first;
				for (int y : i->second)
				{
					if (j->second.count(y))
					{
						if (last == -1) last = y;
						else
						{
							ans = min(ans, dx * (y - last));
							last = y;
						}
					}
				}
			}
		}
		if (ans != INT_MAX)
			return ans;
		return 0;
	}

	//940. Distinct Subsequences II
	int distinctSubseqII(string S) {
		int const mod = 1e9 + 7;
		vector<int> last(128, -1);
		int n = S.size();
		vector<ll> dp(n + 1);
		dp[0] = 1;
		for (int i = 1; i <= n; ++i)
		{
			dp[i] = dp[i - 1] * 2 % mod;
			if (last[S[i - 1]] != -1)
			{
				dp[i] = (dp[i] - dp[last[S[i - 1]]] + 2 * mod) % mod;
			}
			last[S[i - 1]] = i - 1;
		}
		return dp[n] - 1;
	}

	//941. Valid Mountain Array
	bool validMountainArray(vector<int>& a) {
		if (a.size() < 3) return false;
		int pv = -1, n = a.size();
		for (int i = 0; i + 1 < n; ++i)
		{
			if (a[i] < a[i + 1]) continue;
			pv = i;
			break;
		}
		for (int i = pv + 1; i < n; ++i)
		{
			if (a[i] < a[i - 1]) continue;
			return false;
		}
		return pv != -1 && 1 <= pv && pv <= n - 2;
	}

	//944. Delete Columns to Make Sorted
	int minDeletionSize(vector<string>& a)
	{
		int n = a.size(), m = a[0].size();
		int ans = 0;
		for (int j = 0; j < m; ++j)
		{
			bool f0 = true, f1 = true;
			for (int i = 1; f0 && i < n; ++i)
			{
				if (a[i][j] < a[i - 1][j]) f0 = false;
			}
			if (f0) continue;
			ans++;
		}
		return ans;
	}

	//942. DI String Match
	vector<int> diStringMatch(string S) {
		int n = S.size();
		vector<int> ans(n + 1);
		ans[0] = 0;
		int l = 0, h = 0;
		for (int i = 0; i < n; ++i)
		{
			if (S[i] == 'I')
				ans[i + 1] = ++h;
			else
				ans[i + 1] = --l;
		}
		for (int &e : ans)
			e += -l;
		return ans;
	}
	//943. Find the Shortest Superstring
	string shortestSuperstring(vector<string>& a)
	{
		if (a.size() == 1) return a[0];
		int n = a.size();
		vector<vector<int>> g(n, vector<int>(n));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (i == j) continue;
				for (int l = min(a[i].size(), a[j].size()); l >= 0; --l)
				{
					if (a[i].substr(a[i].size() - l) == a[j].substr(0, l))
					{
						g[i][j] = l;
						break;
					}
				}
			}
		}
		int const maxn = INT_MAX / 8;
		vector<vector<int>> dp(1 << n, vector<int>(n, maxn));
		for (int i = 0; i < n; ++i)
		{
			dp[1 << i][i] = a[i].size();
		}
		vector<vector<int>> path(1 << n, vector<int>(n, -1));
		int BOUND = 1 << n;
		for (int s = 0; s < BOUND; ++s)
		{
			for (int i = 0; i < n; ++i)
			{
				if (s & (1 << i))
				{
					int pre = s ^ (1 << i);
					for (int j = 0; j < n; ++j)
					{
						if (pre & (1 << j))
						{
							int v = dp[pre][j] + (int)a[i].size() - g[j][i];
							if (dp[s][i] > v)
							{
								dp[s][i] = v;
								path[s][i] = j;
							}
						}
					}
				}
			}
		}
		int ans = maxn, x = -1;
		for (int i = 0; i < n; ++i)
		{
			if (ans > dp[BOUND - 1][i])
			{
				ans = dp[BOUND - 1][i];
				x = i;
			}
		}
		//cout << ans << endl;
		string ret = a[x];
		int s = (BOUND - 1) ^ (1 << x), pre = x;
		x = path[BOUND - 1][x];
		while (x != -1)
		{
			ret = a[x].substr(0, a[x].size() - g[x][pre]) + ret;

			int ns = s ^ (1 << x);
			pre = x;
			x = path[s][x];
			s = ns;
		}
		return ret;
	}
	//int dfs_943(bitset<12> &s, int i, vector<string> &a, vector<vector<int>> &memo, vector<vector<int>> &path, vector<vector<int>> &g)
	//{
	//	if (s.count() == a.size()) return 0;
	//	int &ans = memo[s.to_ulong()][i];
	//	if (ans != -1) return ans;
	//	ans = INT_MAX;
	//	int &p = path[s.to_ulong()][i];
	//	int nx = -1;
	//	for (int k = 0; k < a.size(); ++k)
	//	{
	//		if (s[k] == 0)
	//		{
	//			s[k] = 1;
	//			int v = dfs_943(s, k, a, memo, path, g) + a[k].size() - g[i][k];
	//			if (v < ans)
	//			{
	//				ans = v;
	//				nx = k;
	//			}
	//			s[k] = 0;
	//		}
	//	}
	//	p = nx;
	//	return ans;
	//}
	//string shortestSuperstring(vector<string>& a)
	//{
	//	if (a.size() == 1) return a[0];
	//	int n = a.size();
	//	vector<vector<int>> g(n, vector<int>(n));
	//	for (int i = 0; i < n; ++i)
	//	{
	//		for (int j = 0; j < n; ++j)
	//		{
	//			if (i == j) continue;
	//			for (int l = min(a[i].size(), a[j].size()); l >= 0; --l)
	//			{
	//				if (a[i].substr(a[i].size() - l) == a[j].substr(0, l))
	//				{
	//					g[i][j] = l;
	//					break;
	//				}
	//			}
	//		}
	//	}
	//	vector<vector<int>>memo(1 << n, vector<int>(n, -1));
	//	vector<vector<int>>path(1 << n, vector<int>(n, -1));
	//	int ans = INT_MAX;
	//	int x = -1;
	//	for (int i = 0; i < n; ++i)
	//	{
	//		bitset<12> s;
	//		s[i] = 1;
	//		int v = dfs_943(s, i, a, memo, path, g) + a[i].size();
	//		if (ans > v)
	//		{
	//			ans = v;
	//			x = i;
	//		}
	//	}
	//	// cout << ans << endl;
	//	int cnt = 1;
	//	string ret = a[x];
	//	int s = (1 << x);
	//	while (cnt < n)
	//	{
	//		int nx = path[s][x];
	//		ret = ret + a[nx].substr(g[x][nx]);
	//		x = nx;
	//		s |= (1 << nx);
	//		cnt++;
	//	}
	//	return ret;
	//}
};
int main()
{
	Solution sol;
	vector<string> a;
	a = { "catg","ctaagt","gcta","ttca","atgcatc" };
	cout << sol.shortestSuperstring(a) << endl;
	//vector<vector<int>> a;
	//a = { {0,1},{1,0} };
	//cout << sol.shortestBridge(a) << endl;
	return 0;
}