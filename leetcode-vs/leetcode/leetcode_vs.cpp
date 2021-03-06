﻿// cpptemp.cpp: 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include <sstream>
#include <math.h>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <queue>

using namespace std;


struct Interval {
    int start;
    int end;

    Interval() : start(0), end(0) {}

    Interval(int s, int e) : start(s), end(e) {}
};


class TrieNode {
public:
    char v;
    bool isword;
    TrieNode *next[26];

    TrieNode(char v_ = 0) : v(v_), isword(false) {
        for (int i = 0; i < 26; i++)
            next[i] = nullptr;
    }

};

class Trie {
    TrieNode *root;
public:
    Trie() {
        root = new TrieNode();
    }

    void add(string word) {
        TrieNode *cur = root;
        for (char i : word) {
            if (cur->next[i - 'a'] == nullptr)
                cur->next[i - 'a'] = new TrieNode(i);
            cur = cur->next[i - 'a'];
        }
        cur->isword = true;
    }

    string prefix(string word) {
        string ret = "";
        TrieNode *cur = root;
        for (char i : word) {
            cur = cur->next[i - 'a'];
            if (cur == nullptr)
                return word;
            ret += i;
            if (cur->isword)
                return ret;
        }
        return ret;
    }

};

class Node {
public:
    Node *next[2];

    Node() {
        next[0] = nullptr;
        next[1] = nullptr;
    }
};

//class NodeTree {
//	Node *root;
//public:
//	NodeTree(Node *root_) {
//		root = root_;
//	}
//	void add(unsigned int v)
//	{
//		Node *cur = root;
//		while (cur&&v)
//		{
//			if (!cur->next[v & 1])
//				cur->next[v & 1] = new Node(v & 1);
//			cur = cur->next[v & 1];
//			v >>= 1;
//		}
//		cur->isWord = true;
//	}
//};



void fun_xx() {
    int T;
    cin >> T;
    while (T--) {
        int n, k;
        cin >> n >> k;
        vector<int> dogs(n);
        for (int i = 0; i < n; i++) {
            int ti = 0;
            cin >> ti;
            dogs[i] = ti;
        }
        sort(dogs.begin(), dogs.end());
        vector<int> diff;
        for (int i = 1; i < n; i++)
            diff.push_back(dogs[i] - dogs[i - 1]);
        sort(diff.begin(), diff.end());
        int ans = 0;
        for (int i = 0; i < n - k; i++)
            ans += diff[i];
        cout << ans << endl;

    }
}

//402
string removeKdigits(string num, int k) {
    int last = 0;
    int remaining = num.size() - k;
    string ret = "";
    for (int i = 1; i <= remaining; i++) {
        int minv = num[last], min_index = last;
        for (int j = last + 1; j < num.size() - k + i - 1; j++) {
            if (num[j] < minv) {
                minv = num[j];
                min_index = j;
            }
        }
        ret += minv;
        last = min_index + 1;
    }
    return ret;
}

int add(int a, int b) {
    return a + b;
}


struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};


class Solution {
    map<int, int> cache;
public:
    Solution() {
        cache[0] = 1;
        cache[1] = 0;
        cache[2] = 1;
        cache[3] = 2;
    }

    int integerReplacement(long long n) {
        if (cache.find(n) != cache.end())
            return cache[n];
        int r = 0;
        if (n & 1) {
            r = min(integerReplacement(n - 1), integerReplacement(n + 1)) + 1;
            cache[n] = r;
            return r;
        }
        return integerReplacement(n >> 1) + 1;

    }

    int kthSmallest(vector<vector<int>> &matrix, int k) {
        int n = matrix.size();
        int le = matrix[0][0], ri = matrix[n - 1][n - 1];
        int mid = 0;
        while (le < ri) {
            mid = le + (ri - le) / 2;
            int num = 0;
            for (int i = 0; i < n; i++) {
                int pos = upper_bound(matrix[i].begin(), matrix[i].end(), mid) - matrix[i].begin();
                num += pos;
            }
            cout << mid << " " << num << endl;
            if (num < k) {
                le = mid + 1;
            } else {
                ri = mid;
            }
        }
        return le;
    }

    bool isPowerOfTwo(int a) {
        long long n = a;
        if (n <= 0)
            return false;
        while (n != 1) {
            if (n & 1)
                return false;
            n >>= 1;
        }
        return true;
    }

    int minDepth(TreeNode *root) {
        if (!root)
            return 0;
        int left = minDepth(root->left);
        int right = minDepth(root->right);
        return (left == 0 || right == 0) ? left + right + 1 : min(left, right) + 1;
    }

    bool canPartition(vector<int> &nums) {
        int sums = 0;
        for (int &a : nums)
            sums += a;
        if (sums & 1)
            return false;
        sums >>= 1;
        int n = nums.size();
        vector<vector<bool>> dp(n + 1, vector<bool>(sums + 1, false));
        dp[0][0] = true;
        for (int i = 1; i < n + 1; i++)
            dp[i][0] = true;
        for (int i = 1; i < sums + 1; i++)
            dp[0][i] = false;
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < sums + 1; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j > nums[i - 1])
                    dp[i][j] = dp[i][j] || dp[i - 1][j - nums[i - 1]];
            }
        }
        return dp[n][sums];
    }

    int nthSuperUglyNumber(int n, vector<int> &primes) {
        return 0;
    }


    //648
    string replaceWords(vector<string> &dict, string sentence) {
        sentence += " ";
        Trie trie;
        for (string word : dict)
            trie.add(word);
        string ret = "";
        string token = "";

        token += sentence[0];

        int last = 0;

        for (int i = 1; i < sentence.size(); i++) {
            if (sentence[i] == ' ' || i == sentence.size() - 1) {
                //cout << token << endl;
                if (last == 0)
                    ret += trie.prefix(token);
                else
                    ret += " " + trie.prefix(token);
                token = "";
                last = i + 1;
            } else
                token += sentence[i];
        }
        return ret;
    }

    int maxProduct(vector<string> &words) {
        int ret = 0, len = words.size();
        vector<unsigned int> values(len, 0);
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < words[i].size(); j++)
                values[i] |= 1 << (words[i][j] - 'a');
        }
        for (int i = 0; i < len; i++)
            for (int j = i + 1; j < len; j++)
                if (!(values[i] & values[j]) && words[i].length() * words[j].length() > ret)
                    ret = words[i].length() * words[j].length();
        return ret;
    }

    int bulbSwitch(int n) {
        return sqrt(n);
    }

    vector<int> plusOne(vector<int> &digits) {
        if (digits[0] == 0) {
            digits[0] = 1;
            return digits;
        }
        int carry = 1;
        for (int i = digits.size() - 1; i >= 0; i--) {
            carry = digits[i] + carry;
            digits[i] = carry % 10;
            carry = carry / 10;
        }
        if (carry) {
            digits.resize(digits.size() + 1);
            for (int i = digits.size() - 1; i >= 1; i--) {
                digits[i] = digits[i - 1];
            }
            digits[0] = carry;
        }
        return digits;
    }


    int lengthOfLIS(vector<int> &nums) {
        vector<int> res;
        for (int i = 0; i < nums.size(); i++) {
            auto it = std::lower_bound(res.begin(), res.end(), nums[i]);
            if (it == res.end()) res.push_back(nums[i]);
            else *it = nums[i];

            cout << nums[i] << endl;
            for (auto a : res) {
                cout << a << " ";
            }
            cout << endl;
        }
        return res.size();
    }

    int rob(vector<int> &nums) {
        if (nums.size() <= 2) {
            int ret = 0;
            for (auto &i : nums)
                ret = max(ret, i);
            return ret;
        }
        vector<int> dp(nums.size(), 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);

        for (int i = 2; i < nums.size(); i++)
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);

        return dp[nums.size() - 1];
    }


    //int inoreder(TreeNode *p, int pre, int prepre)
    //{
    //	if (p == nullptr)
    //	{
    //		int ret = 0;
    //		ret = max(pre, prepre);
    //		return ret;
    //	}
    //	if (p->val + prepre > pre)
    //		return inoreder(p->left, p->val + prepre, pre) + inoreder(p->right, p->val + prepre, pre);
    //	else
    //		return inoreder(p->left, p->val, pre) + inoreder(p->right, p->val, pre);
    //}

    //int rob(TreeNode* root) {
    //	return inoreder(root, 0, 0);
    //}

    //void travel(TreeNode*p, int dep, vector<int> &nums)
    //{
    //	if (p == nullptr)
    //		return;
    //	if (dep > nums.size())
    //		nums.push_back(0);
    //	nums[dep] += p->val;
    //	travel(p->left, dep + 1, nums);
    //	travel(p->right, dep + 1, nums);
    //}

    int rob(TreeNode *root) {
        if (!root)
            return 0;
        int val = 0;
        if (root->left)
            val += rob(root->left->left) + rob(root->left->right);
        if (root->right)
            val += rob(root->right->left) + rob(root->right->right);
        return max(root->val + val, rob(root->left) + rob(root->right));
    }

    bool isPowerOfFour(int num) {
        if (num <= 0)
            return false;
        if (num == 1)
            return true;

        unsigned n = num;
        int t = 0;
        while (n) {
            if (n == 1)
                break;
            if (n & 1)
                return false;
            n >>= 1;
            t++;
        }
        return (t & 1) == 0;
    }

    int integerBreak(int n) {
        if (n == 2)
            return 1;
        if (n == 3)
            return 2;
        int ret = 1;
        while (n > 4) {
            ret *= 3;
            n -= 3;
        }
        ret *= n;
        return ret;
    }

    bool isVowel(char a) {
        if (a == 'a' || a == 'e' || a == 'o' || a == 'i' || a == 'u' || a == 'A' || a == 'E' || a == 'O' || a == 'I' ||
            a == 'U')
            return true;
        return false;

    }

    string reverseVowels(string s) {
        int i = 0, j = s.length() - 1;

        while (i < j) {
            while (i < s.length() && !isVowel(s[i]))
                i++;
            while (j >= 0 && !isVowel(s[j]))
                j--;
            if (i < j) {
                swap(s[i], s[j]);
                i++;
                j--;
            }
        }
        return s;
    }

    vector<int> largestDivisibleSubset(vector<int> &nums) {
        sort(nums.begin(), nums.end());

        int m = 0, mi = 0;

        vector<int> T(nums.size(), 0), parent(nums.size(), 0);

        for (int i = nums.size() - 1; i >= 0; i--)
            for (int j = i; j < nums.size(); j++)
                if (nums[j] % nums[i] == 0 && T[i] < 1 + T[j]) {
                    T[i] = T[j] + 1;

                    parent[i] = j;

                    if (T[i] > m) {
                        m = T[i];
                        mi = i;
                    }
                }
        vector<int> ret;
        for (int i = 0; i < m; i++) {
            ret.push_back(nums[mi]);
            mi = parent[mi];
        }
        return ret;
    }

    int wiggleMaxLength(vector<int> &nums) {
        if (nums.size() <= 1)
            return nums.size();

        int ret = 0, cur = 1;;
        while (cur < nums.size() && nums[cur - 1] == nums[cur])
            cur++;
        if (cur >= nums.size())
            return 1;
        bool flag = nums[cur] < nums[cur - 1];

        cur++;
        ret = 2;
        while (cur < nums.size()) {
            if (flag && nums[cur] > nums[cur - 1]) {
                ret++;
                flag = !flag;
            } else if (!flag && nums[cur] < nums[cur - 1]) {
                ret++;
                flag = !flag;
            }
            cur++;
        }
        return ret;
    }

    int longestSubstring(string s, int k) {
        //395
        int count[26] = {0};
        for (int i = 0; i < s.length(); i++) {
            count[s[i] - 'a'] += 1;
            if (count[s[i] - 'a'] == k + 1)
                return i + 1;
        }
        return s.length();
    }

    bool validUtf8(vector<int> &data) {
        int count = 0;
        for (auto i : data) {
            if (count == 0) {

                if ((i >> 5) == 0b110) count = 1;
                else if ((i >> 4) == 0b1110) count = 2;
                else if ((i >> 3) == 0b11110) count = 3;
                else if ((i >> 7)) return false;
            } else {
                count--;
                if ((i >> 6) != 0b10)
                    return false;
            }
        }
        return count == 0;
    }

    int maxRotateFunction(vector<int> &A) {
        int sums = 0, F = 0, ret = 0;
        for (int i = 0; i < A.size(); i++) {
            sums += A[i];
            F += i * A[i];
        }
        ret = F;
        for (int i = A.size() - 1; i >= 0; i--) {
            F = F + sums - A[i] * A.size();
            ret = max(F, ret);
        }
        return ret;
    }

    string toHex(int num) {
        string ret = "";
        int v;
        for (int i = 0; num && i < 32; i += 4, num >>= 4) {
            v = num & 0xf;
            if (v >= 10)
                ret = char('a' + v - 10) + ret;
            else
                ret = char('0' + v) + ret;
        }
        return ret.empty() ? "0" : ret;
    }

    void dfs(int x, int y, int preh, vector<vector<int>> &table, vector<vector<int>> &matrix, int v,
             vector<pair<int, int>> &dirs, vector<pair<int, int>> &ret) {
        if (x < 0 || x >= matrix.size() || y < 0 || y >= matrix[0].size())
            return;
        if (table[x][y] & v || matrix[x][y] < preh)
            return;

        table[x][y] |= v;
        if (x == 2 && y == 2) {
            cout << "bk" << endl;
        }
        if (v == 2 && table[x][y] == 3)
            ret.push_back({x, y});
        for (auto &p : dirs) {
            int nx = x + p.first, ny = y + p.second;
            if (nx < 0 || nx >= matrix.size() || ny < 0 || ny >= matrix[0].size())
                continue;
            dfs(nx, ny, matrix[x][y], table, matrix, v, dirs, ret);
        }
    }

    //417
    vector<pair<int, int>> pacificAtlantic(vector<vector<int>> &matrix) {
        int m = matrix.size();
        vector<pair<int, int>> ret;
        if (!m)
            return ret;
        int n = matrix[0].size();
        vector<vector<int>> table(m, vector<int>(n, 0));

        vector<pair<int, int>> dirs;
        dirs.push_back({0, 1});
        dirs.push_back({1, 0});
        dirs.push_back({-1, 0});
        dirs.push_back({0, -1});


        for (int i = 0; i < m; i++)
            dfs(i, 0, matrix[i][0], table, matrix, 1, dirs, ret);
        cout << endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                cout << table[i][j] << " ";
            cout << endl;
        }
        for (int j = 0; j < n; j++)
            dfs(0, j, matrix[0][j], table, matrix, 1, dirs, ret);
        cout << endl;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                cout << table[i][j] << " ";
            cout << endl;
        }
        dirs.clear();
        dirs.push_back({0, -1});
        dirs.push_back({-1, 0});

        for (int i = 0; i < m; i++)
            dfs(i, n - 1, matrix[i][n - 1], table, matrix, 2, dirs, ret);
        cout << endl;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                cout << table[i][j] << " ";
            cout << endl;
        }
        for (int j = 0; j < n; j++)
            dfs(m - 1, j, matrix[m - 1][j], table, matrix, 2, dirs, ret);
        cout << endl;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                cout << table[i][j] << " ";
            cout << endl;
        }
        return ret;
    }


    int findMaximumXOR_(vector<int> &nums) {
        int mask = 0, ret = 0;
        for (int i = 31; i >= 0; i--) {
            mask |= 1 << i;
            unordered_set<int> S;
            for (auto &num : nums)
                S.insert(num & mask);
            int tmp = ret | (1 << i);
            for (auto prefix : S)
                if (S.find(tmp ^ prefix) != S.end()) {
                    ret = tmp;
                    break;
                }
        }
        return ret;
    }


    void buildTree(Node *root, int v) {
        Node *cur = root;
        for (int i = 31; i >= 0; i--) {
            int idx = (v >> i) & 1;
            if (!cur->next[idx])
                cur->next[idx] = new Node();
            cur = cur->next[idx];
        }
    }

    int findMaximumXOR(vector<int> &nums) {
        Node *root = new Node();
        for (auto n : nums)
            buildTree(root, n);
        int ret = 0;

        for (auto n : nums) {
            Node *cur = root;
            int res = 0;
            for (int i = 31; i >= 0; i--) {
                int idx = (n >> i) & 1 ? 0 : 1;
                res <<= 1;

                if (cur->next[idx]) {
                    res |= 1;
                    cur = cur->next[idx];
                } else
                    cur = cur->next[idx ? 0 : 1];
            }
            if (res > ret)
                ret = res;
        }
        return ret;
    }

    int characterReplacement(string s, int k) {
        int ans = 0, curCount = 0, count[26] = {0}, start = 0, end = 0;
        while (end < s.length()) {
            count[s[end] - 'A']++;
            curCount = max(curCount, count[s[end] - 'A']);
            while (end - start + 1 - curCount > k) {
                count[s[start] - 'A']--;
                start++;
            }
            ans = max(ans, end - start + 1);
            end++;
        }
        return ans;
    }

    vector<int> findRightInterval_(vector<Interval> &intervals) {
        map<int, int> hash;
        for (int i = 0; i < intervals.size(); i++)
            hash[intervals[i].start] = i;
        vector<int> ret;
        for (auto &i : intervals) {
            auto p = hash.lower_bound(i.end);
            if (p != hash.end())
                ret.push_back(p->second);
            else
                ret.push_back(-1);
        }
        return ret;
    }

    int low_bound(vector<int> &nums, int target) {
        int i = 0, j = nums.size() - 1;
        while (i <= j) {
            int mid = (i + j) / 2;
            if (nums[mid] >= target)
                j = mid - 1;
            else
                i = mid + 1;
        }
        return i;
    }

    vector<int> findRightInterval(vector<Interval> &intervals) {
        map<int, int> hash;
        vector<int> starts(intervals.size(), 0);
        for (int i = 0; i < intervals.size(); i++) {
            hash[intervals[i].start] = i;
            starts[i] = intervals[i].start;
        }
        sort(starts.begin(), starts.end());
        vector<int> ret;
        for (auto &i : intervals) {
            int idx = low_bound(starts, i.end);
            if (idx >= intervals.size())
                ret.push_back(-1);
            else
                ret.push_back(hash[starts[idx]]);
        }
        return ret;
    }

    int compress(vector<char> &chars) {
        int i = 0, cur = 0;
        while (i < chars.size()) {
            int c = 0;
            char t = chars[i];
            while (i < chars.size() && chars[i] == t) {
                i++;
                c++;
            }
            if (c > 1) {
                string n = to_string(c);
                chars[cur++] = t;
                for (auto cc : n)
                    chars[cur++] = cc;
            } else
                chars[cur++] = t;
        }
        return cur;
    }

    TreeNode *deleteNode(TreeNode *root, int key) {
        if (!root)
            return nullptr;
        if (key < root->val)
            root->left = deleteNode(root->left, key);
        else if (key > root->val)
            root->right = deleteNode(root->right, key);
        else {
            if (root->left == nullptr)
                return root->right;
            else if (root->right == nullptr)
                return root->left;

            TreeNode *mid = root->right;

            while (mid->left)
                mid = mid->left;

            root->val = mid->val;
            root->right = deleteNode(root->right, root->val);
        }
        return root;
    }

    int findMaxForm(vector<string> &strs, int m, int n) {
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        for (auto &s : strs) {
            int n0 = 0, n1 = 0;
            for (int i = 0; i < s.size(); i++)
                if (s[i] == '0')
                    n0++;
                else
                    n1++;
            for (int i = m; i >= n0; i--)
                for (int j = n; j >= n1; j--)
                    dp[i][j] = max(dp[i][j], dp[i - n0][j - n1] + 1);

        }
        return dp[m][n];
    }

    int findSubstringInWraproundString(string p) {
        vector<int> dp(p.size() + 1, 0), cur(26, 0);
        int ans = 0, len = 0;

        for (int i = 1; i < p.size(); i++) {
            int c = p[i] - 'a', last;
            if (p[i] - p[i - 1] == 1 || p[i - 1] - p[i] == 25)
                len++;
            else
                len = 1;
            cur[p[i] - 'a'] = max(cur[p[i] - 'a'], len);
        }
        for (int i = 0; i < 26; i++)
            ans += cur[i];
        return ans;
    }

    bool dfs(vector<int> &nums, int sums[], int index, int target) {
        if (index == nums.size())
            return sums[0] == target && sums[1] == target && sums[2] == target;
        for (int i = 0; i < 4; i++) {
            if (sums[i] + nums[index] > target)
                continue;
            int j = i;
            while (--j >= 0)
                if (sums[j] == sums[i])
                    break;
            if (j != -1)
                continue;
            sums[i] += nums[index];
            if (dfs(nums, sums, index + 1, target))
                return true;
            sums[i] -= nums[index];
        }
        return false;
    }

    bool makesquare(vector<int> &nums) {
        if (nums.size() < 4)
            return false;
        int sums = 0;
        for (int i : nums)
            sums += i;
        if (sums % 4 != 0)
            return false;
        sort(nums.begin(), nums.end());
        int i = 0, j = nums.size() - 1;
        while (i < j) {
            swap(nums[i], nums[j]);
            i++;
            j--;
        }
        return dfs(nums, new int[4]{0}, 0, sums / 4);
    }

    string licenseKeyFormatting(string S, int K) {
        string ret;
        int k = K;
        for (int i = S.size() - 1; i >= 0; i--) {
            if (S[i] == '-')
                continue;
            if (k == 0) {
                //ret = '-' + ret;
                ret.push_back('-');
                k = K;
            }
            if ('a' <= S[i] && S[i] <= 'z')
                S[i] += 'A' - 'a';
            ret.push_back(S[i]);
            //ret = S[i] + ret;
            k--;
        }
        reverse(ret.begin(), ret.end());
        return ret;
    }

    int magicalString(int n) {
        string S = "122";
        int i = 2;
        while (S.size() < n)
            S += string(S[i++] - '0', S.back() ^ 3);
        return count(S.begin(), S.begin() + n, '1');
    }

    int findMaxConsecutiveOnes(vector<int> &nums) {
        int count = 0, Len = nums.size();

        for (int i = 0; i < Len; i++) {
            if (nums[i] == 0)
                continue;
            int t = 0;
            while (i < Len && nums[i] == 1) {
                t++;
                i++;
            }
            i--;
            count = max(count, t);
        }
        return count;
    }

    int gcd(int a, int b) {
        if (a < b) {
            a ^= b;
            b ^= a;
            a ^= b;
        }
        int t;
        while (b) {
            t = b;
            b = a % b;
            a = t;
        }
        return a;
    }

    string fractionAddition(string expression) {
        stringstream ss(expression);
        int A = 0, B = 1, a, b;
        char _;
        while (ss >> a >> _ >> b) {
            A = A * b + B * a;
            B *= b;
            int g = abs(gcd(A, B));
            A /= g;
            B /= g;
            cout << A << " " << B << endl;
        }
        return to_string(A) + '/' + to_string(B);
    }

    TreeNode *addOneRow(TreeNode *root, int v, int d) {
        if (d == 1) {
            TreeNode *n = new TreeNode(v);
            n->left = root;
            return n;
        }
        queue<TreeNode *> q;
        q.push(root);
        int dep = d - 2;
        while (!q.empty() && dep--) {
            int len = q.size();
            while (len--) {
                TreeNode *cur = q.front();
                q.pop();
                if (cur->left)
                    q.push(cur->left);
                if (cur->right)
                    q.push(cur->right);
            }
        }
        while (!q.empty()) {
            TreeNode *cur = q.front();
            q.pop();
            TreeNode *n = new TreeNode(v);
            n->left = cur->left;
            cur->left = n;
            n = new TreeNode(v);
            n->right = cur->right;
            cur->right = n;
        }
        return root;
    }

    int maxCoins(vector<int> &nums) {
        //312. Burst Balloons
        int len = nums.size();
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        vector<vector<int>> dp(nums.size(), vector<int>(nums.size(), 0));
        for (int l = 1; l <= len; l++)
            for (int i = 1; i <= len - l + 1; i++) {
                int j = i + l - 1;
                for (int k = i; k <= j; k++) {
                    dp[i][j] = max(dp[i][j], nums[i - 1] * nums[k] * nums[j + 1] + dp[i][k - 1] + dp[k + 1][j]);
                }
            }
        return dp[1][len];
    }
};


int main_1() {

    Solution sol;
    vector<int> nums{3, 1, 5, 8};
    cout << sol.maxCoins(nums) << endl;
    //string s = "1234567";
    //cout << s.substr(0) << ' ' << s.substr(1) << " " << s.substr(1, 1) << endl;
    //vector<int>nums{ 1,1,0,1,1,1 };
    //cout << sol.findMaxConsecutiveOnes(nums) << endl;
    //pair<int, pair<int, int>> t;
    //t.first = 1;
    //t.second.first = 2;
    //t.second.second = 3;
    //cout << sol.magicalString(6) << endl;
    //cout << sol.licenseKeyFormatting("2-4A0r7-4k", 3) << endl;
    //cout << sol.licenseKeyFormatting("2-4A0r7-4k", 4) << endl;
    //vector<string > strs;
    //strs.push_back("10");
    //strs.push_back("0001");
    //strs.push_back("111001");
    //strs.push_back("1");
    //strs.push_back("0");
    //cout << sol.findMaxForm(strs, 5, 3) << endl;
    //TreeNode* root = new TreeNode(5);
    //root->left = new TreeNode(3);
    //root->left->left = new TreeNode(2);
    //root->left->right = new TreeNode(4);

    //root->right = new TreeNode(6);
    //root->right->right = new TreeNode(7);

    //auto r = sol.deleteNode(root, 5);

    //cout << r->val << endl;

    //vector<char> chars;
    ////chars.push_back('a');
    ////chars.push_back('a');
    ////chars.push_back('b');
    ////chars.push_back('b');
    ////chars.push_back('c');
    ////chars.push_back('c');
    ////chars.push_back('c');

    //chars.push_back('a');
    //chars.push_back('a');
    //chars.push_back('a');
    //chars.push_back('a');
    //chars.push_back('a');
    //chars.push_back('a');
    //chars.push_back('b');

    //int r = sol.compress(chars);

    ////vector<Interval> intervals;
    ////intervals.push_back(Interval(1, 4));
    ////intervals.push_back(Interval(2, 3));
    ////intervals.push_back(Interval(3, 4));

    ////auto ret = sol.findRightInterval(intervals);
    //cout << r << endl;
    //for (auto &i : chars)
    //{
    //	cout << i << " ";
    //}
    //cout << endl;
    //cout << sol.characterReplacement("AABABBA", 1) << endl;

    //cout << sol.integerReplacement(2147483647) << endl;
    //vector<vector<int>> matrix{ {1,5,9},{10,11,13},{12,13,15} };
    //cout << sol.kthSmallest(matrix, 8) << endl;
    //int a[10];
    //cout << "你好" << endl;
    //vector<int> array{1, 2, 3, 4};
    //map<int, int> map_t;
    //map_t[1] = 1;
    //int v = sol.isPowerOfTwo(29);
    //cout << sol.isPowerOfTwo(29) << endl;
    //cout << (int)0x80000000 << endl;
    //cout << removeKdigits("1432219", 3) << endl;
    //cout << removeKdigits("10200", 1) << endl;
    //vector<string> words;
    //words.push_back("cat");
    //words.push_back("bat");
    //words.push_back("rat");
    //cout << sol.replaceWords(words, "the cattle was rattled by the battery") << endl;
    //vector<int> digits;
    //for (int i : {8, 0, 9, 9, 9, 9})
    //	digits.push_back(i);
    //sol.plusOne(digits);
    //for (int i : digits)
    //	cout << i ;
    //cout << endl;
    //int(*fun)(int, int);
    //fun = add;
    //cout << fun(1, 2) << endl;
    //return 0;
    //vector<string> words = { "abcw", "baz", "foo", "bar", "xtfn", "abcdef","a" };
    //vector<string> words = { "abc","def","aaa" };

    //cout << sol.maxProduct(words) << endl;
    //vector<int >nums{ 1, 3, 54, 3, 2, 54, 7 };
    //cout << sol.lengthOfLIS(nums) << endl;
    //vector<int >nums{ 2,1,6,3,8,4 };

    //cout << sol.wiggleMaxLength(nums) << endl;
    //cout << sol.getMoneyAmount(4) << endl;
    //cout << sol.longestSubstring("aaabb", 3) << endl;
    //cout << sol.longestSubstring("ababbc", 2) << endl;
    //vector<int> nums{ 197,130,1 };
    //vector<int> nums{ 235,140,4 };
    //int r = 0;
    //if (sol.validUtf8(nums))
    //	r = 1;
    //else
    //	r = 0;
    //cout << r << endl;
    //vector<int> nums{ 4,3,2,6 };
    //cout << sol.maxRotateFunction(nums) << endl;
    //int a;
    //while (cin >> a)
    //{
    //	cout << sol.toHex(a) << endl;
    //}
    //vector<vector<int>> matrix{
    //	{ 1, 2, 2, 3, 5 },
    //	{ 3, 2, 3, 4, 4 },
    //	{ 2, 4, 5, 3, 1 },
    //	{ 6, 7, 1, 4, 5 },
    //	{ 5, 1, 1, 2, 4 }
    //};
    //vector<vector<int>> matrix{
    //{ 1,2,3 },
    //{ 8,9,4 },
    //{ 7,6,5 }
    //};
    //for (auto p : sol.pacificAtlantic(matrix)) {
    //	cout << p.first << " " << p.second << endl;
    //}
    //vector<int> nums{ 3, 10, 5, 25, 2, 8 };
    //cout << sol.findMaximumXOR(nums) << endl;
    return 0;
}