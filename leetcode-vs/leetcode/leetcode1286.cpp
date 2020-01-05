#include <numeric>
#include <future>
#include <stack>
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
#include <thread>
#include <functional>
#include <mutex>
#include <string>

#include "common.h"

using namespace std;


//1286. Iterator for Combination
class CombinationIterator {
    string s;
    int n, len;
    vector<int> pos;
    bool valid = true;
public:
    CombinationIterator(string s, int n) : pos(len, 0), s(s), n(n), len(s.size()) {
        sort(s.begin(), s.end());
        for (int i = 0; i < n; ++i)
        {
            pos[i] = 1;
        }
    }

    string next() {
        string ret;
        ret.clear();
        for (int i = 0; i < len; ++i)
        {
            if (pos[i])
            {
                ret.push_back(s[i]);
            }
        }

        int i = len - 1;
        int cnt = 0;
        valid = true;
        if (pos[i] == 1)
        {
            for (; i >= 0 && pos[i] == 1; --i)
            {
                cnt++;
            }
            if (cnt == n)
            {
                valid = false;
            }
            for (int j = len - 1; j > i; --j)
            {
                pos[j] = 0;
            }
            for (; i >= 0; --i)
            {
                if (pos[i] == 1)
                {
                    pos[i] = 0;
                    pos[i + 1] = 1;
                    for (int j = 0; j < cnt; ++j)
                    {
                        pos[i + 2 + j] = 1;
                    }
                    break;
                }
            }
        }
        else
        {
            for (; i >= 0 && pos[i] == 0; --i);
            pos[i] = 0;
            pos[i + 1] = 1;
        }
        return ret;
    }

    bool hasNext() {
        return valid;
    }
};

class Solution
{
    //1288. Remove Covered Intervals
    int removeCoveredIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());
        int last = 0, ans = 1;
        for (int i = 0; i < intervals.size(); i++) {
            if (intervals[last][0] <= intervals[i][0] &&
                intervals[i][1] <= intervals[last][1])
                continue;
            ans++;
            last = i;
        }
        return ans;
    }

    //1289. Minimum Falling Path Sum II
    int minFallingPathSum(vector<vector<int>>& a) {
        int n = a.size(), m = a[0].size();
        vector<vector<int>> dp(n, vector<int>(m, INT_MAX));
        for (int j = 0; j < m; ++j) dp[0][j] = a[0][j];
        for (int i = 1; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                for (int k = 0; k < m; ++k)
                {
                    if (j == k) continue;
                    dp[i][j] = min(dp[i][j], dp[i - 1][k] + a[i][j]);
                }
            }
        }
        int ans = INT_MAX;
        for (int j = 0; j < m; ++j)
        {
            ans = min(ans, dp[n - 1][j]);
        }
        return ans;
    }
    
    //1287. Element Appearing More Than 25% In Sorted Array
    int findSpecialInteger(vector<int>& a) {
        int n = a.size();
        for (int i = 0; i < n; )
        {
            int v = a[i];
            int cnt = 0;
            while (i < n && a[i] == v)
            {
                cnt++;
                i++;
            }
            if (cnt > n / 4.0)
                return v;
        }
        return -1;
    }


    //1290. Convert Binary Number in a Linked List to Integer
	int getDecimalValue(ListNode* head) {
		int ans = 0;
		while (head)
		{
			ans = ans * 2 + head->val;
			head = head->next;
		}
		return ans;
	}

    vector<int> solve_len_1291(int n, int low, int high)
    {
        vector<int> ans;
        for (int i = 1; i + n - 1 < 10; ++i)
        {
            int t = 0;
            for (int j = 0; j < n; ++j)
            {
                t = t * 10 + i + j;
            }
            if (low <= t && t <= high)
                ans.push_back(t);
        }
        return ans;
    }
    //1291. Sequential Digits
    vector<int> sequentialDigits(int low, int high) {
        vector<int> ans;
        for (int l = 1; l <= 9; ++l)
        {
            for (auto& e : solve_len_1291(l, low, high))
            {
                ans.push_back(e);
            }
        }
        return ans;
    }

    //1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
	int maxSideLength(vector<vector<int>>& mat, int t) {
		int n = mat.size(), m = mat[0].size();
		NumMatrix nm(mat);
		int l = 0, r = max(n, m) + 1;
		while (l < r)
		{
			int mid = (l + r) / 2;
			bool valid = false;
			for (int i = 0; i + mid - 1 < n; ++i)
			{
				for (int j = 0; j + mid - 1 < m; ++j)
				{
					int val = nm.sumRegion(i, j, i + mid - 1, j + mid - 1);
					if (val <= t)
					{
						valid = true;
					}
				}
			}
			if (!valid)
			{
				r = mid;
			}
			else
			{
				l = mid + 1;
			}
		}
		return l - 1;
	}

    /*
    //1293. Shortest Path in a Grid with Obstacles Elimination
    int dr[] = { 0, 1, 0, -1 };
    int dc[] = { 1, 0, -1, 0 };

	int const maxn = 40 + 1;
	int dp[maxn][maxn][maxn * maxn];
	class Solution {
	public:
		int n, m;

        int dfs(int i, int j, int left, vector<vector<int>>& g)
        {
            if (i == n - 1 && j == m - 1) return 0;
            int& ans = dp[i][j][left];
            if (ans != -1) return ans;
            ans = INT_MAX / 4;
            for (int d = 0; d < 4; ++d)
            {
                int ni = i + dr[d], nj = j + dc[d];
                if (0 <= ni && ni < n && 0 <= nj && nj < m)
                {
                    if (g[ni][nj] == 1 && left > 0)
                    {
                        ans = min(ans, dfs(ni, nj, left - 1, g) + 1);
                    }
                    else if (g[ni][nj] == 0)
                    {
                        ans = min(ans, dfs(ni, nj, left, g) + 1);
                    }
                }
            }
            return ans;
        }
        int shortestPath(vector<vector<int>>& grid, int k) {
            if (grid[0][0] == 1)
            {
                if (k == 0) return -1;
                k--;
                grid[0][0] = 0;
            }
            memset(dp, -1, sizeof(dp));
            n = grid.size(), m = grid[0].size();
            int ret = dfs(0, 0, k, grid);
            if (ret == INT_MAX / 4) return -1;
            return ret;
        }
    };
    */

	//1295. Find Numbers with Even Number of Digits
	int findNumbers(vector<int>& a) {
		int ans = 0;
		for (auto e : a)
		{
			int len = 0;
			while (e)
			{
				len++;
				e /= 10;
			}
			ans += (len % 2 == 0);
		}
		return ans;
	}

	//1296. Divide Array in Sets of K Consecutive Numbers
	bool isPossibleDivide(vector<int>& nums, int k) {
		map<int, int> cnt;
		for (auto& e : nums)
			cnt[e] ++;
		while (!cnt.empty())
		{
			int u = cnt.begin()->first;
			for (int i = 0; i < k; ++i)
			{
				int v = i + u;
				if (cnt.count(v))
				{
					if (--cnt[v] == 0)
					{
						cnt.erase(v);
					}
				}
				else
				{
					return false;
				}
			}
		}
		return true;
	}

	//1297. Maximum Number of Occurrences of a Substring
	int maxFreq(string s, int maxLetters, int minSize, int maxSize) {
		unordered_map<string, int> cnt;
		int ans = 0;
		int diff = 0, i = 0, j = 0;
		int char_count[256] = { 0 };
		for (; j < minSize; ++j) {
			if (++char_count[s[j]] == 1) {
				diff++;
			}
		}
		do {
			if (diff <= maxLetters) {
				ans = max(ans, ++cnt[s.substr(i, minSize)]);
			}
			if (j == s.length()) break;
			if (--char_count[s[i]] == 0) {
				diff--;
			}
			if (++char_count[s[j]] == 1) {
				diff++;
			}
			i++; j++;
		} while (1);
		return ans;
	}


	//1298. Maximum Candies You Can Get from Boxes
	int maxCandies(vector<int>& status, vector<int>& candies, vector<vector<int>>& keys, vector<vector<int>>& containedBoxes, vector<int>& initialBoxes) {
		queue<int> q;
		for (auto& e : initialBoxes) q.push(e);
		int ans = 0;
		int n = status.size();

		vector<int> vis(n);     // hasCandies
		vector<int> hasKey(n);  // left keys
		vector<int> hasBox(n);  // need key

		while (!q.empty())
		{
			auto u = q.front(); q.pop();
			hasBox[u] = 1;

			if (!vis[u])
			{
				if (status[u] == 1)
				{
					ans += candies[u];
					vis[u] = 1;
				}
				else
				{
					hasBox[u] = 1;
				}
			}

			for (auto& box : containedBoxes[u])
			{
				hasBox[box] = 1;
			}

			for (auto& key : keys[u])
			{
				hasKey[key] = 1;
			}

			for (int box = 0; box < n; ++box)
			{
				if (hasBox[box] && status[box] && vis[box] == 0)
				{
					ans += candies[box];
					q.push(box);
					vis[box] = 1;
				}
			}

			for (int key = 0; key < n; ++key)
			{
				if (hasBox[key] && hasKey[key] && vis[key] == 0)
				{
					vis[key] = 1;
					q.push(key);

					ans += candies[key];
					hasBox[key] = 0;
				}
			}
		}
		return ans;
	}

    //1299. Replace Elements with Greatest Element on Right Side
    vector<int> replaceElements(vector<int>& arr) {
        int mx = -1;
        for (int i = arr.size() - 1; i >= 0; --i)
        {
            int t = arr[i];
            arr[i] = mx;
            mx = max(mx, t);
        }
        return arr;
    }

    //1300. Sum of Mutated Array Closest to Target
    int findBestValue(vector<int>& a, int t) {
        int n = a.size();
        int l = 0, r = 1e5 + 1;

        int ret = 0;

        while (l < r)
        {
            int m = (l + r) / 2;
            int sum = 0;
            for (int i = 0; i < n; ++i)
            {
                sum += min(m, a[i]);
            }

            if (sum >= t)
            {
                r = m;
            }
            else
            {
                l = m + 1;
            }
        }
        int ans_less = 0, ans_big = 0;
        for (auto& e : a)
        {
            ans_less += min(e, l - 1);
            ans_big += min(e, l);
        }

        if (abs(ans_less - t) <= abs(ans_big - t))
        {
            return l - 1;
        }
        return l;
    }

    //1301. Number of Paths with Max Score
    vector<int> pathsWithMaxScore(vector<string>& board) {
        int n = board.size(), m = board[0].size(), mod = 1e9 + 7;
        vector<pair<int, int>> dp(m + 1);
        dp[0] = { 0, 1 };
        for (int i = 0; i < n; i++) {
            auto last = dp;
            dp[0] = { 0, 0 };
            for (int j = 0; j < m; j++) {
                if (board[i][j] == 'X') {
                    dp[j + 1] = { 0, 0 };
                    continue;
                }
                auto [a0, a1] = dp[j];
                auto [b0, b1] = last[j];
                auto [c0, c1] = last[j + 1];
                if (!a1 && !b1 && !c1)
                    dp[j + 1] = { 0, 0 };
                else {
                    int64_t maxsum = max({ a0, b0, c0 }), ways = 0;
                    ways += a0 == maxsum ? a1 : 0;
                    ways += b0 == maxsum ? b1 : 0;
                    ways += c0 == maxsum ? c1 : 0;;
                    maxsum += isdigit(board[i][j]) ? board[i][j] - '0' : 0;
                    dp[j + 1] = { maxsum % mod, ways % mod };
                }
            }
        }
        return { dp[m].first, dp[m].second };
    }


    void dfs_1302(int d, TreeNode* u, vector<int> &dep)
    {
        if (!u) return;
        if (dep.size() < d + 1) dep.push_back(u->val);
        else dep[d] += u->val;
        dfs_1302(d + 1, u->left, dep);
        dfs_1302(d + 1, u->right, dep);
    }
    //1302. Deepest Leaves Sum
    int deepestLeavesSum(TreeNode* root) {
        vector<int> dep;
        dfs_1302(0, root, dep);
        return dep.back();
    }

    //1304. Find N Unique Integers Sum up to Zero
    vector<int> sumZero(int n) {
        vector<int> ans;
        if (n % 2) ans.push_back(0);
        for (int i = 0; i < n / 2; ++i)
        {
            ans.push_back(i + 1);
            ans.push_back(-i - 1);
        }
        return ans;
    }


    void dfs_1305(TreeNode* u, vector<int>& a)
    {
        if (!u) return;
        dfs_1305(u->left, a);
        a.push_back(u->val);
        dfs_1305(u->right, a);
    }
    //1305. All Elements in Two Binary Search Trees
    vector<int> getAllElements(TreeNode* root1, TreeNode* root2) {
        vector<int> a, b;
        dfs_1305(root1, a);
        dfs_1305(root2, b);
        vector<int> c(a.size() + b.size());
        merge(a.begin(), a.end(), b.begin(), b.end(), c.begin());
        return c;
    }
    
    //1306. Jump Game III
    bool canReach(vector<int>& a, int st) {
        int n = a.size();
        queue<int> q;
        vector<bool> vis(n);
        q.push(st);
        vis[st] = 1;
        if (a[st] == 0) return true;

        while (!q.empty())
        {
            int size = q.size();
            while (size--)
            {
                auto u = q.front();
                q.pop();
                if (a[u] == 0) return true;
                if (u + a[u] < n)
                {
                    if (!vis[u + a[u]])
                    {
                        vis[u + a[u]] = 1;
                        q.push(u + a[u]);
                    }
                }
                if (u - a[u] >= 0)
                {
                    if (!vis[u - a[u]])
                    {
                        vis[u - a[u]] = 1;
                        q.push(u - a[u]);
                    }
                }
            }
        }
        return false;
    }


    int calc_left_5298(vector<pair<char, int>>& left, int carry, vector<int>& test)
    {
        int value_left = carry;
        for (auto& e : left)
        {
            value_left += test[e.first - 'A'] * e.second;
        }
        return value_left;
    }

    bool calc_5298(vector<pair<char, int>>& left, int carry, char right, vector<int>& test)
    {
        if (calc_left_5298(left, carry, test) % 10 == test[right - 'A'])
        {
            return true;
        }
        return false;
    }


    void dfs_1307(int u, vector<pair<char, int>>& left, int carry, char right, vector<int>& test,
        vector<vector<int>>& out, vector<int>& pre, vector<bool>& vis)
    {
        if (u == left.size())
        {
            if (pre[right - 'A'] != -1)
            {
                test[right - 'A'] = pre[right - 'A'];
                if (calc_5298(left, carry, right, test))
                {
                    out.push_back(test);
                }
            }
            else
            {
                for (int i = 0; i < 10; ++i)
                {
                    if (vis[i] == 0)
                    {
                        vis[i] = 1;
                        test[right - 'A'] = i;
                        if (calc_5298(left, carry, right, test))
                        {
                            out.push_back(test);
                        }
                        vis[i] = 0;
                    }
                }
            }
            return;
        }


        if (pre[left[u].first - 'A'] != -1)
        {
            test[left[u].first - 'A'] = pre[left[u].first - 'A'];
            dfs_1307(u + 1, left, carry, right, test, out, pre, vis);
        }
        else
        {
            for (int i = 0; i < 10; ++i)
            {
                if (vis[i] == 0)
                {
                    vis[i] = 1;
                    test[left[u].first - 'A'] = i;
                    dfs_1307(u + 1, left, carry, right, test, out, pre, vis);
                    vis[i] = 0;
                }
            }
        }
    }

    vector<vector<int>> generate_5298(vector<pair<char, int>>& left, int carry, char right, vector<int>& pre)
    {
        vector<int> test(26, -1);
        vector<vector<int>> out;

        vector<bool> vis(10);
        for (int i = 0; i < 26; ++i)
        {
            if (pre[i] != -1) vis[pre[i]] = 1;
        }

        dfs_1307(0, left, carry, right, test,
            out, pre, vis);
        return out;
    }

    bool dfs_1307(vector<int>& pre, int pos, int carry, vector<string>& ws, string& res)
    {
        if (pos == res.size()) return true;
        vector<int> cnt(26);
        for (int i = 0; i < ws.size(); ++i)
        {
            if (pos < ws[i].size())
            {
                cnt[ws[i][pos] - 'A'] ++;
            }
        }
        vector<pair<char, int>> left;
        for (char i = 'A'; i <= 'Z'; ++i)
        {
            if (cnt[i - 'A'])
            {
                left.push_back({ i, cnt[i - 'A'] });
            }
        }
        char right = res[pos];
        auto test = generate_5298(left, carry, right, pre);
        for (auto& e : test)
        {
            vector<int> er;
            for (int i = 0; i < 26; ++i)
            {
                if (e[i] == -1) continue;
                if (pre[i] == -1) er.push_back(i);
                pre[i] = e[i];
            }
            int value_left = calc_left_5298(left, carry, e);
            if (dfs_1307(pre, pos + 1, value_left / 10, ws, res))
                return true;
            for (auto& i : er)
            {
                pre[i] = -1;
            }
        }
        return false;
    }

    //1307. Verbal Arithmetic Puzzle
    bool isSolvable(vector<string>& words, string result) {
        for (auto& e : words) reverse(e.begin(), e.end());
        reverse(result.begin(), result.end());
        vector<int> pre(26, -1);
        return dfs_1307(pre, 0, 0, words, result);
    }

    //1309. Decrypt String from Alphabet to Integer Mapping
    string freqAlphabets(string s) {
        string ans;
        int n = s.size();
        for (int i = 0; i < n; )
        {

            if (i + 2 < n && s[i + 2] == '#')
            {
                ans.push_back((s[i] - '0') * 10 + s[i + 1] - '0' - 10 + 'j');
                i += 3;
            }
            else
            {
                ans.push_back(s[i] - '0' - 1 + 'a');
                i++;
            }
        }
        return ans;
    }

    //1310. XOR Queries of a Subarray
    vector<int> xorQueries(vector<int>& a, vector<vector<int>>& qs) {
        int n = a.size();
        vector<vector<int>> sums(n + 1, vector<int>(32));
        for (int mask = 0; mask < 31; ++mask)
        {
            for (int i = 0; i < n; ++i)
            {
                sums[i + 1][mask] = sums[i][mask] + ((a[i] & (1 << mask)) != 0);
            }
        }
        int m = qs.size();
        vector<int> ans;
        for (auto& e : qs)
        {
            int l = e[0], r = e[1];
            int val = 0;
            for (int i = 0; i < 31; ++i)
            {
                if ((sums[r + 1][i] - sums[l][i]) & 1)
                {
                    val += (1 << i);
                }
            }
            ans.push_back(val);
        }
        return ans;
    }

    //1311. Get Watched Videos by Your Friends
    vector<string> watchedVideosByFriends(vector<vector<string>>& watchedVideos, vector<vector<int>>& g, int id, int level) {
        queue<int> q;
        map<string, int> freq;
        q.push(id);
        int n = g.size();
        vector<bool> vis(n);
        vis[id] = 1;
        while (!q.empty() && level)
        {
            freq.clear();
            int size = q.size();
            while (size--)
            {
                auto u = q.front();
                q.pop();
                for (auto& v : g[u])
                {
                    if (!vis[v])
                    {
                        vis[v] = 1;
                        q.push(v);
                        for (auto& e : watchedVideos[v])
                        {
                            freq[e] ++;
                        }
                    }
                }
            }
            if (--level == 0)
            {
                vector<pair<int, string>> ord;
                for (auto& e : freq)
                {
                    ord.push_back({ e.second, e.first });
                }
                sort(ord.begin(), ord.end());
                vector<string> ans;
                for (auto& e : ord)
                {
                    ans.push_back(e.second);
                }
                return ans;
            }
        }
        return {};
    }

    //1312. Minimum Insertion Steps to Make a String Palindrome
    int minInsertions(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n));
        for (int i = 0; i < n; ++i)
        {
            dp[i][i] = 1;
            if (i + 1 < n)
            {
                dp[i][i + 1] = 1 + (s[i] == s[i + 1]);
            }
        }
        for (int l = 3; l <= n; ++l)
        {
            for (int i = 0, j = i + l - 1; j < n; ++i, ++j)
            {
                if (s[i] == s[j])
                {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                }
                else
                {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return n - dp[0][n - 1];
    }
};
int main()
{

}
