#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <queue>
#include <set>

using namespace std;

class TrieNode {
public:
    char v;
    bool isword;
    TrieNode *next[26];

    TrieNode(char v_ = 0) : v(v_), isword(false), next{nullptr} {
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


struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

struct ListNode {
    int val;
    ListNode *next;

    ListNode(int x) : val(x), next(NULL) {}
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

    TreeNode *constructMaximumBinaryTree(vector<int> &nums) {

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

    ListNode *detectCycle(ListNode *head) {
        if (!head) {
            return nullptr;
        }
        ListNode *fast = head->next, *slow = head;
        set<ListNode *> cyc;
        while (fast && slow != fast) {
            if (fast->next)
                fast = fast->next->next;
            else
                return nullptr;
            slow = slow->next;
        }

        if (!fast) {
            return nullptr;
        }

        ListNode *endpos = slow;
        do {
            cyc.insert(slow);
            slow = slow->next;
        } while (slow != endpos);

        ListNode *dummy = new ListNode(-1);
        dummy->next = head;
        fast = dummy;
        while (fast->next) {
            if (cyc.find(fast->next) != cyc.end()) {
                return fast->next;
            } else {
                fast = fast->next;
            }
        }
    }

};

ListNode *buildList(vector<int> nums) {
    ListNode *dummy = new ListNode(-1), *cur;
    cur = dummy;
    for (int &i:nums) {
        cur->next = new ListNode(i);
        cur = cur->next;
    }
    return dummy->next;
}

void fun_xx() {
    cout << "fun_xx" << endl;
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

int add(int a, int b){
    return a + b;
}
int main() {
    Solution sol;
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
    ListNode *nums = buildList({1, 2});
    nums->next->next->next = nums;
//    int n = 6;
//    ListNode *head = nums, *cur;
//    head->next = head;
//    cur = head;
//    while (n--) {
//        cur = cur->next;
//    }
//    cur->next = head->next;
//    ListNode *ret = sol.detectCycle(new ListNode(1));
//    cout << ret->val << endl;
    int (*fun)(int, int);
    fun=add;
    cout << fun(1,2) << endl;
    return 0;
}

