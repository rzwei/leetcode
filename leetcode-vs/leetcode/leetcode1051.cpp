#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <set>
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

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class MountainArray {
public:
	int get(int index)
	{
		return 0;
	}
	int length()
	{
		return 0;
	}
	
};
/*int memo[11][1 << 11];

class Solution {
public:
	int dfs(int u, vector<vector<int>> &wk, bitset<12> &vis, vector<vector<int>> &bk)
	{
		if (u == wk.size()) return 0;
		if (memo[u][vis.to_ulong()] != -1) return memo[u][vis.to_ulong()];
		int ans = INT_MAX / 2;
		for (int i = 0; i < bk.size(); ++i)
		{
			if (vis[i] == 0)
			{
				vis[i] = 1;
				int dist = abs(wk[u][0] - bk[i][0]) + abs(wk[u][1] - bk[i][1]);
				ans = min(ans, dfs(u + 1, wk, vis, bk) + dist);
				vis[i] = 0;
			}
		}
		return memo[u][vis.to_ulong()] = ans;
	}
	//1057. Campus Bikes II
	int assignBikes(vector<vector<int>>& workers, vector<vector<int>>& bikes) {
		bitset<12>vis;
		memset(memo, -1, sizeof(memo));
		return dfs(0, workers, vis, bikes);
	}
};
*/


/*
int memo[11][2][11][2];

class Solution {
public:
	int dfs(int i, int lim, int pre, int num, int d, vector<int> &a)
	{
		if (i == a.size())
		{
			return pre;
		}
		if (memo[i][lim][pre][num] != -1) return memo[i][lim][pre][num];
		int ans = 0;
		if (lim)
		{
			for (int v = 0; v <= a[i]; ++v)
			{
				int nx = (num | (v != 0));
				ans += dfs(i + 1, v == a[i], nx ? pre + (v == d) : 0, nx, d, a);
			}
		}
		else
		{
			for (int v = 0; v < 10; ++v)
			{
				int nx = (num | (v != 0));
				ans += dfs(i + 1, 0, nx ? pre + (v == d) : 0, nx, d, a);
			}
		}
		return memo[i][lim][pre][num] = ans;
	}

	//1 - v , count d
	int count(int v, int d)
	{
		vector<int> digits;
		while (v)
		{
			digits.push_back(v % 10);
			v /= 10;
		}
		if (digits.size() == 0) return 0;
		reverse(digits.begin(), digits.end());
		int ans = 0;
		memset(memo, -1, sizeof(memo));
		for (int i = 0; i <= digits[0]; ++i)
		{
			int nx = (i != 0);
			ans += dfs(1, digits[0] == i, nx ? i == d : 0, nx, d, digits);
		}
		return ans;
	}
	//1058. Digit Count in Range
	int digitsCount(int d, int low, int high) {
		return count(high, d) - count(low - 1, d);
	}
};
*/
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
	//1055. Fixed Point
	int fixedPoint(vector<int>& A) {
		int n = A.size();
		for (int i = 0; i < n; ++i)
		{
			if (i == A[i]) return i;
		}
		return -1;
	}

	//1056. Index Pairs of a String
	vector<vector<int>> indexPairs(string s, vector<string>& words) {
		set<string> ss(words.begin(), words.end());
		int n = s.size();
		vector<vector<int>> ans;
		for (int i = 0; i < n; ++i)
		{
			for (int j = i; j < n; ++j)
			{
				if (ss.count(s.substr(i, j - i + 1)))
				{
					ans.push_back({ i, j });
				}
			}
		}
		return ans;
	}

	//1066. Campus Bikes II
	int assignBikes(vector<vector<int>>& workers, vector<vector<int>>& bikes) {
		int n = workers.size(), m = bikes.size();
		int B = (1 << m);
		int const maxn = INT_MAX / 2;
		vector<int> dp(B, maxn), nx(B, maxn);
		dp[0] = 0;
		for (int i = 0; i < n; ++i)
		{
			fill(nx.begin(), nx.end(), maxn);
			for (int s = 0; s < B; ++s)
			{
				for (int j = 0; j < m; ++j)
				{
					// nx[s] = INT_MAX;
					if ((s >> j) & 1)
					{
						nx[s] = min(nx[s], dp[s ^ (1 << j)] + abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1]));
					}
				}
			}
			dp = nx;
		}
		return *min_element(dp.begin(), dp.end());
	}

	//1074. Number of Submatrices That Sum to Target
	int numSubmatrixSumTarget(vector<vector<int>>& a, int t) {
		int n = a.size(), m = a[0].size();

		vector<vector<int>> sums(n, vector<int>(m + 1));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				sums[i][j + 1] = sums[i][j] + a[i][j];
			}
		}

		int ans = 0;
		unordered_map<int, int> pre;
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				pre.clear();
				pre[0] = 1;
				int s = 0;
				for (int k = 0; k < n; ++k)
				{
					int v = sums[k][i + 1] - sums[k][j];
					s += v;
					if (pre.count(s - t)) ans += pre[s - t];
					pre[s] ++;
				}
			}
		}
		return ans;
	}

	bool check_1071(string& p, string t)
	{
		if (p.size() % t.size()) return false;
		int n = p.size(), m = t.size();
		for (int i = 0; i < n; ++i)
		{
			if (p[i] != t[i % m]) return false;
		}
		return true;
	}

	//1071. Greatest Common Divisor of Strings
	string gcdOfStrings(string str1, string str2) {
		if (str1.size() < str2.size()) swap(str1, str2);
		int n = str2.size();
		for (int i = n; i >= 1; --i)
		{
			if (check_1071(str1, str2.substr(0, i)) && check_1071(str2, str2.substr(0, i)))
			{
				return str2.substr(0, i);
			}
		}
		return "";
	}

	int count_1072(vector<vector<int>>& a, vector<int>& fp)
	{
		int n = a.size(), m = a[0].size();
		int ans = 0, ans2 = 0;
		for (int i = 0; i < n; ++i)
		{
			bool f = true, f2 = true;
			for (int j = 1; (f || f2) && j < m; ++j)
			{
				if ((a[i][j] ^ fp[j]) != (a[i][0] ^ fp[0]))
					f = false;
				if ((a[i][j] ^ (~fp[j])) != (a[i][0] ^ (~fp[0])))
					f2 = false;
			}
			ans += f;
			ans2 += f2;
		}
		return max(ans, ans2);
	}

	//1072. Flip Columns For Maximum Number of Equal Rows
	int maxEqualRowsAfterFlips(vector<vector<int>>& a) {
		int n = a.size(), m = a[0].size();
		vector<int> fp(m);
		int ans = 0;
		sort(a.begin(), a.end());
		for (int i = 0; i < n; ++i)
		{
			if (i > 0 && a[i] == a[i - 1]) continue;
			fp = a[i];
			ans = max(ans, count_1072(a, fp));
		}
		return ans;
	}

	////316. Remove Duplicate Letters
	//string removeDuplicateLetters(string s) {
	//	int n = s.size();
	//	vector<int> cnt(26);
	//	vector<bool> vis(26);
	//	for (auto& c : s) cnt[c - 'a'] ++;
	//	string ans;
	//	for (auto& c : s)
	//	{
	//		cnt[c - 'a'] --;
	//		if (vis[c - 'a']) continue;
	//		while (!ans.empty() && c < ans.back() && cnt[ans.back() - 'a'])
	//		{
	//			vis[ans.back() - 'a'] = 0;
	//			ans.pop_back();
	//		}
	//		ans.push_back(c);
	//		vis[c - 'a'] = 1;
	//	}
	//	return ans;
	//}

	//1081. Smallest Subsequence of Distinct Characters
	string smallestSubsequence(string s) {
		int n = s.size();
		vector<int> dp(n + 1);
		for (int i = n - 1; i >= 0; --i)
		{
			dp[i] = dp[i + 1] | (1 << (s[i] - 'a'));
		}
		string ans;
		int cur = dp[0];
		int u = 0;
		while (cur && u < n)
		{
			char mi = 'z';
			for (int i = u; i < n; ++i)
			{
				if (cur & (1 << (s[i] - 'a')))
				{
					if ((cur | dp[i]) == dp[i])
					{
						if (s[i] < mi)
						{
							mi = s[i];
							u = i;
						}
					}
				}
			}
			ans.push_back(mi);
			cur ^= (1 << (mi - 'a'));
			u++;
		}
		return ans;
	}

	vector<string> split(string& s)
	{
		vector<string> ans;
		string u;
		for (auto& c : s)
		{
			if (c == ' ')
			{
				if (!u.empty())
				{
					ans.push_back(u);
					u.clear();
				}
			}
			else
			{
				u.push_back(c);
			}
		}
		if (!u.empty()) ans.push_back(u);
		return ans;
	}
	//1078. Occurrences After Bigram
	vector<string> findOcurrences(string text, string first, string second) {
		auto s = split(text);
		int n = s.size();
		vector<string> ans;
		for (int i = 0; i + 2 < n; ++i)
		{
			if (s[i] == first && s[i + 1] == second)
			{
				ans.push_back(s[i + 2]);
			}
		}
		return ans;
	}

	int dfs_1079(vector<int>& cnt)
	{
		int ans = 0;
		for (int i = 0; i < 26; ++i)
		{
			if (cnt[i] == 0) continue;
			ans++;
			cnt[i] --;
			ans += dfs_1079(cnt);
			cnt[i] ++;
		}
		return ans;
	}
	//1079. Letter Tile Possibilities
	int numTilePossibilities(string s) {
		vector<int> cnt(26);
		for (auto& c : s)
		{
			cnt[c - 'A'] ++;
		}
		return dfs_1079(cnt);
	}
	//1080. Insufficient Nodes in Root to Leaf Paths
	TreeNode* sufficientSubset(TreeNode* root, int limit) {
		if (root->left == root->right)
			return root->val < limit ? NULL : root;
		if (root->left)
			root->left = sufficientSubset(root->left, limit - root->val);
		if (root->right)
			root->right = sufficientSubset(root->right, limit - root->val);
		return root->left == root->right ? NULL : root;
	}

	//1085. Sum of Digits in the Minimum Number
	int sumOfDigits(vector<int>& a) {
		int v = *min_element(a.begin(), a.end());
		int ans = 0;
		while (v)
		{
			ans += v % 10;
			v /= 10;
		}
		return ans % 2 ? 0 : 1;
	}

	//1086. High Five
	vector<vector<int>> highFive(vector<vector<int>>& a) {
		map<int, vector<int>> x;
		for (auto& e : a)
		{
			int id = e[0], sc = e[1];
			x[id].push_back(sc);
		}
		for (auto& e : x)
		{
			sort(e.second.begin(), e.second.end(), greater<int>());
		}
		vector<vector<int>> ans;
		for (auto& e : x)
		{
			int id = e.first;
			int tot = 0;
			for (int i = 0; i < 5; ++i)
			{
				tot += e.second[i];
			}
			ans.push_back({ id, tot / 5 });
		}
		return ans;
	}

	void dfs_1087(int u, string& s, string& cur, vector<string>& ans)
	{
		if (u == s.size())
		{
			ans.push_back(cur);
			return;
		}
		if (s[u] == '{')
		{
			vector<char> values;
			int i = u + 1;
			while (s[i] != '}')
			{
				char v = s[i];
				values.push_back(v);
				if (s[i + 1] == '}')
				{
					i += 2;
					break;
				}
				else if (s[i + 1] == ',')
				{
					i += 2;
				}
			}
			sort(values.begin(), values.end());
			for (auto& e : values)
			{
				cur.push_back(e);
				dfs_1087(i, s, cur, ans);
				cur.pop_back();
			}
		}
		else
		{
			cur.push_back(s[u]);
			dfs_1087(u + 1, s, cur, ans);
			cur.pop_back();
		}
	}
	//1087. Permutation of Letters
	vector<string> permute(string S) {
		vector<string> ans;
		string cur;
		dfs_1087(0, S, cur, ans);
		return ans;
	}

	/*
	//1088. Confusing Number II
	typedef long long int64;
	class Solution {
	public:
		map<int, int> A;
		int n, ret;
		bool check(int64 n) {
			int64 ret = 0;
			for (int64 m = n; m; m /= 10) {
				ret = ret * 10 + A[m % 10];
			}
			return ret != n;
		}
		void search(int pos, int64 cur) {
			//cout << "search:" << pos << " " << cur << endl;
			if (cur > n) return;
			if (cur && check(cur)) {
				//cout << cur << endl;
				++ret;
			}
			for (auto& it : A) {
				if (pos == 0 && it.first == 0) continue;
				search(pos + 1, cur * 10 + it.first);
			}
		}
		int confusingNumberII(int n) {
			A.clear();
			A[0] = 0;
			A[1] = 1;
			A[8] = 8;
			A[6] = 9;
			A[9] = 6;
			this->n = n;
			ret = 0;
			search(0, 0LL);
			return ret;
		}
	};
	*/

	//1090. Largest Values From Labels
	int largestValsFromLabels(vector<int>& values, vector<int>& labels, int num_wanted, int use_limit) {
		int n = values.size();
		vector<pair<int, int>> a(n);
		for (int i = 0; i < n; ++i)
		{
			a[i].first = values[i];
			a[i].second = labels[i];
		}
		sort(a.begin(), a.end());
		map<int, int> cnt;
		int ans = 0;
		for (int i = n - 1; i >= 0 && num_wanted; --i)
		{
			if (cnt[a[i].second] == use_limit)
			{
				continue;
			}
			else
			{
				cnt[a[i].second]++;
				ans += a[i].first;
				num_wanted--;
			}
		}
		return ans;
	}

	//1091. Shortest Path in Binary Matrix
	int shortestPathBinaryMatrix(vector<vector<int>>& a) {
		int n = a.size();
		if (a[0][0] == 1 || a[n - 1][n - 1] == 1) return -1;
		if (n == 1) return 1;
		queue<pair<int, int>> q;
		int ans = 1;
		vector<vector<int>> d = { {0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1,}, {1, -1}, {-1, 1}, {-1, -1} };
		q.push({ 0, 0 });
		a[0][0] = 1;
		while (!q.empty())
		{
			ans++;
			int size = q.size();
			while (size--)
			{

				int x = q.front().first, y = q.front().second;
				q.pop();

				for (int i = 0; i < d.size(); ++i)
				{
					int nx = x + d[i][0], ny = y + d[i][1];
					if (0 <= nx && nx < n && 0 <= ny && ny < n && a[nx][ny] == 0)
					{
						q.push({ nx, ny });
						a[nx][ny] = 1;
						if (nx == n - 1 && ny == n - 1)
						{
							return ans;
						}
					}
				}
			}
		}
		return -1;
	}

	//1092. Shortest Common Supersequence
	string shortestCommonSupersequence(string s, string p) {
		int n = s.size(), m = p.size();
		vector<vector<int>> dp(n + 1, vector<int>(m + 1));
		vector<vector<int>> path(n + 1, vector<int>(m + 1, -1));

		// 0 match;
		// 1 i,j + 1
		// 2 i + 1, j;

		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				if (s[i] == p[j])
				{
					dp[i + 1][j + 1] = dp[i][j] + 1;
					path[i + 1][j + 1] = 0;
				}
				else
				{
					if (dp[i][j + 1] > dp[i + 1][j])
					{
						dp[i + 1][j + 1] = dp[i][j + 1];
						path[i + 1][j + 1] = 1;
					}
					else
					{
						dp[i + 1][j + 1] = dp[i + 1][j];
						path[i + 1][j + 1] = 2;
					}
				}
			}
		}
		int i = n, j = m;
		string ans;
		int lasti = n - 1, lastj = m - 1;
		while (i >= 0 && j >= 0 && path[i][j] != -1)
		{
			if (path[i][j] == 0)
			{
				while (lasti > i - 1)
				{
					ans.push_back(s[lasti--]);
				}
				while (lastj > j - 1)
				{
					ans.push_back(p[lastj--]);
				}
				ans.push_back(s[i - 1]);
				i--;
				j--;
				lasti--;
				lastj--;
			}
			else if (path[i][j] == 1)
			{
				i--;
			}
			else if (path[i][j] == 2)
			{
				j--;
			}
		}
		while (lasti >= 0)
		{
			ans.push_back(s[lasti--]);
		}
		while (lastj >= 0)
		{
			ans.push_back(p[lastj--]);
		}
		reverse(ans.begin(), ans.end());
		return ans;
	}

	//1089. Duplicate Zeros
	void duplicateZeros(vector<int>& a) {
		int n = a.size();
		int len = 0;
		int s = 0;
		for (int i = 0; len < n && i < n; ++i)
		{
			len++;
			len += a[i] == 0;
			s = i;
		}
		for (int j = n - 1; j > s; --s)
		{
			if (a[s] == 0)
			{
				if (j == n - 1 && len > n)
				{
					a[j--] = 0;

				}
				else
				{
					a[j--] = 0;
					a[j--] = 0;
				}
			}
			else
			{
				a[j--] = a[s];
			}
		}
	}

	vector<string> unique(vector<string>& s)
	{
		sort(s.begin(), s.end());
		vector<string> ans;
		for (int i = 0; i < s.size(); ++i)
		{
			if (ans.empty() || ans.back() != s[i]) ans.push_back(s[i]);
		}
		return ans;
	}

	vector<string> parse_1096(int& i, string& s)
	{
		if (i == s.size()) return {};
		vector<string> ret;
		vector<string> cur;
		while (i < s.size())
		{
			if (s[i] == '{')
			{
				auto r = parse_1096(++i, s);
				if (cur.empty()) cur = r;
				else
				{
					vector<string> tmp;
					for (auto& v : r)
					{
						for (auto& e : cur)
						{
							tmp.push_back(e + v);
						}
					}
					cur = tmp;
				}
			}
			else if (s[i] == ',')
			{
				if (ret.empty())
					ret = cur;
				else
				{
					for (auto& e : cur)
						ret.push_back(e);
				}
				cur.clear();
				i++;
			}
			else if (s[i] == '}')
			{
				++i;
				if (ret.empty()) ret = cur;
				else
				{
					for (auto& e : cur) ret.push_back(e);
				}
				return unique(ret);
			}
			else
			{
				if (cur.empty())
				{
					cur.push_back(string(1, s[i]));
				}
				else
				{
					for (auto& e : cur)
						e.push_back(s[i]);
				}
				i++;
			}
		}
		if (ret.empty()) ret = cur;
		else
		{
			for (auto& e : cur) ret.push_back(e);
		}
		return unique(ret);
	}
	//1096. Brace Expansion II
	vector<string> braceExpansionII(string s) {
		int i = 0;
		return parse_1096(i, s);
	}

	//1095. Find in Mountain Array
	int findInMountainArray(int t, MountainArray& a) {
		int n = a.length();
		int l = 0, r = n - 1;
		int pos = 0;
		while (l < r)
		{
			int mid = (l + r) / 2;
			int mid1 = mid + 1;
			int mid_v = a.get(mid), mid1_v = a.get(mid1);
			if (mid1_v < mid_v)
			{
				r = mid;
			}
			else
			{
				l = mid1;
			}
		}
		int mx = a.get(l);
		pos = l;
		l = 0, r = pos;
		while (l <= r)
		{
			int mid = (l + r) / 2;
			int v = a.get(mid);
			if (v == t)
			{
				return mid;
			}
			else if (v < t)
			{
				l = mid + 1;
			}
			else
			{
				r = mid - 1;
			}
		}
		l = pos, r = n - 1;
		while (l <= r)
		{
			int mid = (l + r) / 2;
			int v = a.get(mid);
			if (v == t)
			{
				return mid;
			}
			else if (v < t)
			{
				r = mid - 1;
			}
			else
			{
				l = mid + 1;
			}
		}
		return -1;
	}

	//1094. Car Pooling
	bool carPooling(vector<vector<int>>& a, int cap) {
		int cur = 0, n = a.size();
		vector<pair<int, int>> b(n + n);
		for (int i = 0; i < n; ++i)
		{
			b[2 * i] = make_pair(a[i][1], a[i][0]);
			b[2 * i + 1] = make_pair(a[i][2], -a[i][0]);
		}
		sort(b.begin(), b.end());
		for (int i = 0; i < n + n; ++i)
		{
			cur += b[i].second;
			if (cur > cap) return false;
		}
		return true;
	}
	

	//1093. Statistics from a Large Sample
	vector<double> sampleStats(vector<int>& count) {
		int n = count.size();
		double mi = 1e9 + 10, mx = 0, common = -1;
		int tot = 0;
		double sum = 0;
		for (int i = 0; i < n; ++i)
		{
			tot += count[i];
			if (common == -1 || count[i] > count[common])
			{
				common = i;
			}
			if (count[i])
			{
				mi = min(mi, (double)i);
				mx = max(mx, (double)i);
			}
			sum += count[i] * i;
		}

		double zw = 0;
		// cout << tot << endl;
		if (tot & 1)
		{
			int mid = (tot + 1) / 2;
			int cur = 0;
			for (int i = 0; i < n; ++i)
			{
				cur += count[i];
				if (cur >= mid)
				{
					zw = i;
					break;
				}
			}
		}
		else
		{

			int cur = 0, mid = tot / 2;

			bool pre = false;

			for (int i = 0; i < n; ++i)
			{
				cur += count[i];
				if (!pre && cur >= mid + 1)
				{
					zw += i + i;
					break;
				}
				else if (cur >= mid && cur < mid + 1)
				{
					zw += i;
					pre = true;
				}
				else if (pre && cur >= mid + 1)
				{
					zw += i;
					break;
				}
				if (cur >= mid + 1) break;
			}
			zw /= 2;
		}
		return { mi, mx, sum / tot, zw, common };
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
