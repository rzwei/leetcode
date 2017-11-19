import math


# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# class TrieNode:
#     # Initialize your data structure here.
#     def __init__(self):
#         self.children = collections.defaultdict(TrieNode)
#         self.is_word = False
#
#
# class Trie:
#     def __init__(self):
#         self.root = TrieNode()
#
#     def insert(self, word):
#         current = self.root
#         for letter in word:
#             current = current.children[letter]
#         current.is_word = True
#
#     def search(self, word):
#         current = self.root
#         for letter in word:
#             current = current.children.get(letter)
#             if current is None:
#                 return False
#         return current.is_word
#
#     def startsWith(self, prefix):
#         current = self.root
#         for letter in prefix:
#             current = current.children.get(letter)
#             if current is None:
#                 return False
#         return True

class Node:
    def __init__(self):
        self.is_word = False
        self.children = {}


class Trie(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        cur = self.root
        for letter in word:
            if letter not in cur.children:
                cur.children[letter] = Node()
            cur = cur.children[letter]
        cur.is_word = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        cur = self.root
        for letter in word:
            if letter not in cur.children:
                return False
            cur = cur.children[letter]
        return cur.is_word

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        cur = self.root
        for letter in prefix:
            if letter not in cur.children:
                return False
            cur = cur.children[letter]
        return True


class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num == 1:
            return True
        i = 1
        j = num // 2
        while i <= j:
            # print(i, j)
            mid = (i + j) // 2
            t = mid ** 2
            if t == num:
                return True
            if t < num:
                i = mid + 1
            else:
                j = mid - 1

        return False

    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        d = {}
        for i in moves:
            d[i] = d.get(i, 0) + 1
        return d.get('L', 0) == d.get('R', 0) and d.get('U', 0) == d.get('D', 0)

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False

        n = len(matrix)
        m = len(matrix[0])

        def geti(i):
            x = i // m
            y = i % m
            return matrix[x][y]

        i = 0
        j = n * m - 1
        while i <= j:
            mid = (i + j) // 2
            vm = geti(mid)
            if vm == target:
                return True
            elif vm < target:
                i = mid + 1
            else:
                j = mid - 1
        return False

    # public List<Integer> findAnagrams(String s, String p) {
    #     List<Integer> list = new ArrayList<>();
    # if (s == null || s.length() == 0 || p == null || p.length() == 0) return list;
    # int[] hash = new int[256]; //character hash
    # //record each character in p to hash
    # for (char c : p.toCharArray()) {
    #     hash[c]++;
    # }
    # //two points, initialize count to p's length
    # int left = 0, right = 0, count = p.length();
    # while (right < s.length()) {
    # //move right everytime, if the character exists in p's hash, decrease the count
    # //current hash value >= 1 means the character is existing in p
    # if (hash[s.charAt(right++)]-- >= 1) count--;
    #
    # //when the count is down to 0, means we found the right anagram
    # //then add window's left to result list
    # if (count == 0) list.add(left);
    #
    # //if we find the window's size equals to p, then we have to move left (narrow the window) to find the new match window
    # //++ to reset the hash because we kicked out the left
    # //only increase the count if the character is in p
    # //the count >= 0 indicate it was original in the hash, cuz it won't go below 0
    # if (right - left == p.length() && hash[s.charAt(left++)]++ >= 0) count++;
    # }
    # return list;
    # }


    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        if not s or len(p) > len(s):
            return []
        pd = {}
        for i in p:
            pd[i] = pd.get(i, 0) + 1

        window = {}
        window_size = len(p)

        ls = len(s)

        ret = []
        for i in range(window_size):
            window[s[i]] = window.get(s[i], 0) + 1

        f = 1
        for i in pd:
            if pd[i] != window.get(i, 0):
                f = 0
                break
        if f:
            ret.append(0)

        for i in range(1, ls - window_size + 1):
            old_letter = s[i - 1]

            if old_letter in window:
                window[old_letter] -= 1
                if window[old_letter] <= 0:
                    del window[old_letter]

            new_letter = s[i + window_size - 1]
            window[new_letter] = window.get(new_letter, 0) + 1
            f = 1
            for j in pd:
                if pd[j] != window.get(j, 0):
                    f = 0
                    break
            if f:
                ret.append(i)
        return ret

    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        ni = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[ni] = nums[i]
                ni += 1
        return ni

    def findRestaurant(self, list1, list2):
        """
        :type list1: List[str]
        :type list2: List[str]
        :rtype: List[str]
        """
        Aindex = {v: i for i, v in enumerate(list1)}
        best, ans = 1e9, []
        for j, v in enumerate(list2):
            i = Aindex.get(v, 1e9)
            if i + j < best:
                best = i + j
                ans = [v]
            elif i + j == best:
                ans.append(v)
        return ans

    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return nums[0]

        def subfun(nums):

            if not nums:
                return 0
            if len(nums) <= 2:
                return max(nums)
            dp = [0 for i in nums]
            dp[0] = nums[0]
            dp[1] = max(nums[0], nums[1])

            for i in range(2, len(nums)):
                dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
            return dp[len(nums) - 1]

        return max(subfun(nums[1:]), subfun(nums[:-1]))

    def evalRPN(self, tokens: list):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        while tokens:
            v = tokens.pop(0)
            if v.isdigit() or v.startswith('-') and len(v) > 1:
                stack.append(int(v))
            else:
                op2 = stack.pop()
                op1 = stack.pop()
                if v == '+':
                    stack.append(op1 + op2)
                elif v == '-':
                    stack.append(op1 - op2)
                elif v == '*':
                    stack.append(op1 * op2)
                elif v == '/':
                    vv = abs(op1) // abs(op2)
                    if op1 * op2 < 0:
                        vv = -vv
                    stack.append(vv)

        return stack[0]

    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d = {}
        for i in nums:
            if i not in d:
                d[i] = 0
            d[i] += 1
            if d[i] > len(nums) / 2:
                return i

    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        tokens1 = version1.split('.')
        tokens2 = version2.split('.')
        for i in range(len(tokens1)):
            tokens1[i] = (int(tokens1[i]))
        for i in range(len(tokens2)):
            tokens2[i] = (int(tokens2[i]))
        i = 0
        while i < len(tokens1) and i < len(tokens2):
            if tokens1[i] < tokens2[i]:
                return -1
            elif tokens1[i] > tokens2[i]:
                return 1
            else:
                i += 1
        if i < len(tokens2):
            return -1
        elif i < len(tokens1):
            return 1
        else:
            return 0

    def findRelativeRanks(self, nums: list):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        d = {}
        for i, v in enumerate(nums):
            d[v] = i
        nums.sort(reverse=True)

        award = ["Gold Medal", "Silver Medal", "Bronze Medal"]

        ret = [0 for i in nums]
        for i in range(len(nums)):
            if i < 3:
                ret[d[nums[i]]] = award[i]
            else:
                ret[d[nums[i]]] = str(i + 1)
        return ret

    def constructRectangle(self, area):
        """
        :type area: int
        :rtype: List[int]
        """
        s = math.sqrt(area)
        s = math.ceil(s)
        s = int(s)
        for i in range(s, 0, -1):
            if area % i == 0:
                a = [area // i, i]
                a.sort(reverse=True)
                return a

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        pre = strs[0]
        for i in strs:
            while pre not in i:
                pre = pre[:-1]
        return pre

    def lastRemaining(self, n):
        """
        :type n: int
        :rtype: int
        """
        left = True
        remaining = n
        step = 1
        head = 1
        while remaining > 1:
            if left or remaining % 2 == 1:
                head += step
            remaining //= 2
            step *= 2
            left = not left
        return head

    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = set()

        def dfs(now: list):
            li = now[-1]

            if li >= len(nums):
                return

            t = [nums[i] for i in now]
            t = tuple(t)
            if t in ans:
                return

            lv = nums[li]

            for i in range(li + 1, len(nums)):
                v = nums[i]
                if v >= lv:
                    now.append(i)
                    t = [nums[i] for i in now]
                    t = tuple(t)
                    if t not in ans:
                        dfs(now)
                        ans.add(t)
                    now.pop(-1)

        for i in range(len(nums)):
            dfs([i])
        ret = []
        for i in ans:
            ret.append(list(i))
        return ret

    def oddEvenList(self, head: ListNode):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        odd = head
        even = head.next
        evenHead = even
        while even and even.next:
            odd.next = odd.next.next
            even.next = even.next.next
            odd = odd.next
            even = even.next
        odd.next = evenHead
        return head

    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        B = [A[i] - A[i - 1] for i in range(1, len(A))]
        ans = 0
        lengthB = len(B)

        i = 0
        while i < lengthB:
            ti = 1
            j = i + 1
            while j < lengthB and B[j] == B[i]:
                j += 1
                ti += 1
            if ti >= 2:
                ans += ti * (ti - 1) // 2

            i = j
        return ans

    def travel_inorder(self, root: TreeNode):
        if not root:
            return
        self.travel_inorder(root.left)
        print(root.val)
        self.travel_inorder(root.right)

    def buildTree(self, inorder: list, postorder: list):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not inorder:
            return None
        pv = postorder[-1]
        pvi = inorder.index(pv)
        node = TreeNode(pv)
        curinorder = inorder[:pvi]
        curpost = postorder[:pvi]
        node.left = self.buildTree(curinorder, curpost)
        curinorder = inorder[pvi + 1:]
        curpost = postorder[pvi:-1]
        node.right = self.buildTree(curinorder, curpost)
        return node

    def isPossible(self, nums):
        """
        #659
        :type nums: List[int]
        :rtype: bool
        """
        pass

    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        # 617
        if t1 and t2:
            head = TreeNode(t1.val + t2.val)
            head.left = self.mergeTrees(t1.left, t2.left)
            head.right = self.mergeTrees(t1.right, t2.right)
        elif t1:
            head = TreeNode(t1.val)
            head.left = self.mergeTrees(t1.left, None)
            head.right = self.mergeTrees(t1.right, None)
        elif t2:
            head = TreeNode(t2.val)
            head.left = self.mergeTrees(None, t2.left)
            head.right = self.mergeTrees(None, t2.right)
        else:
            return None

        return head

    def largestPalindrome(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 479
        pass

    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ans = []

        def fun(s, left, right):
            if len(s) == n + n:
                ans.append(s)
                return
            if left < n:
                fun(s + '(', left + 1, right)
            if right < left:
                fun(s + ')', left, right + 1)

        fun("", 0, 0)
        return ans

    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        ln = len(needle)
        if len(needle) > len(haystack):
            return -1
        for i in range(len(haystack) - len(needle) + 1):
            j = 0
            flag = True
            while j < ln:
                if haystack[i] == needle[j]:
                    i += 1
                    j += 1
                else:
                    flag = False
                    break
            if flag:
                return i - ln
        return -1

    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        k -= 1
        nums = [i for i in range(1, n + 1)]
        ans = []
        while nums:
            ln = len(nums)
            offset = int(math.factorial(len(nums) - 1))
            lk = k // offset
            ans.append(nums[lk])
            nums.pop(lk)
            k = k % offset
        ret = ''
        for i in ans:
            ret += str(i)
        return ret

    def findLongestWord(self, s, d: list):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        # 524

        # d.sort(key=lambda x: (-len(x), x))

        for i in d:
            ti = i
            j = 0
            i = 0

            flag = False

            while j < len(s) and i < len(ti):
                if ti[i] == s[j]:
                    if i == len(ti) - 1:
                        flag = True
                    i += 1
                j += 1

            if flag:
                return ti
        return ""

    def checkEqualTree(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root and not root.left and not root.right:
            return False
        d = {}

        def travel(p: TreeNode):
            if not p:
                return 0
            if not p.left and not p.right:
                # p.s = p.val
                d[p.val] = d.get(p.val, 0) + 1
                return p.val
            s = travel(p.left) + travel(p.right) + p.val
            d[s] = d.get(s, 0) + 1
            return s

        s = travel(root)

        if s % 2 != 0:
            return False

        return s / 2 in d

    def decodeString(self, s: str):
        """
        :type s: str
        :rtype: str
        """
        # 394
        curi = s.find(']')
        while curi != -1:
            j = curi - 1

            si = ''
            while j >= 0:
                if s[j] == '[':
                    break
                si = s[j] + si

                j -= 1
            n = ''
            tj = j - 1
            while tj >= 0 and s[tj].isdigit():
                n = s[tj] + n
                tj -= 1

            s = s[:tj + 1] + si * int(n) + s[curi + 1:]

            curi = s.find(']')
        return s

    def findNthDigit(self, n):
        """
        :type n: int
        :rtype: int
        """
        i = 0
        s = 0
        while s < n:
            i += 1
            s += 9 * 10 ** (i - 1) * i

        s -= 9 * 10 ** (i - 1) * i

        offset = n - s
        num = 10 ** (i - 1) - 1 + int(math.ceil(offset / (i + 0.0)))
        num = str(num)
        ni = offset % i - 1
        return int(num[ni])

    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        # 331
        diff = 1
        tokens = preorder.split(',')
        for t in tokens:
            diff -= 1
            if diff < 0:
                return False
            if t != '#':
                diff += 2
        return diff == 0

    def connect(self, root):
        if not root:
            return
        pre = root
        while pre.left:
            cur = pre
            while cur:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                cur = cur.next
            pre = pre.left

    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """

        maxx = [[], 0]
        cur = [[], 0]

        def travel(p: TreeNode):
            if not p:
                return

            travel(p.left)

            if cur[0] != None and p.val == cur[0]:
                cur[1] += 1
            else:
                cur[0] = p.val
                cur[1] = 1

            if cur[1] > maxx[1]:
                maxx[1] = cur[1]
                maxx[0] = [p.val]

            elif cur[1] == maxx[1]:
                maxx[0].append(p.val)

            travel(p.right)

        travel(root)
        return maxx[0]

    def insert(self, intervals, newInterval: Interval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        ret = []

        i = 0

        while i < len(intervals) and intervals[i].end < newInterval.start:
            ret.append(intervals[i])
            i += 1

        while i < len(intervals) and intervals[i].start <= newInterval.end:
            if intervals[i].start < newInterval.start:
                newInterval.start = intervals[i].start
            if intervals[i].end > newInterval.end:
                newInterval.end = intervals[i].end
            i += 1
        ret.append(newInterval)
        while i < len(intervals) and intervals[i].start > newInterval.end:
            ret.append(intervals[i])
            i += 1

        return ret

    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        # 329

        d = {}

        m = len(matrix)
        n = len(matrix[0])

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        notInGraph = lambda x, y: x >= m or x < 0 or y >= n or y < 0

        def search(i, j):
            if (i, j) in d:
                return d[(i, j)]

            if notInGraph(i, j):
                return -1

            v = 1
            for di in dirs:
                x = i + di[0]
                y = j + di[1]
                if notInGraph(x, y) or matrix[x][y] <= matrix[i][j]:
                    continue
                v = max(search(x, y) + 1, v)
            d[(i, j)] = v
            return v

        maxx = 1
        for i in range(m):
            for j in range(n):
                maxx = max(maxx, search(i, j))
        return maxx

    def palindromePairs(self, words):
        """
        :type words: List[str]
        :rtype: List[List[int]]
        """

        # 336
        cache = {}
        ans = []

        def reverse(s):
            return s[::-1]

        def isPalindrome(s):
            if not s:
                return False
            if s in cache:
                return cache[s]
            i = 0
            j = len(s) - 1
            while i < j:
                if s[i] == s[j]:
                    i += 1
                    j -= 1
                else:
                    cache[s] = False
                    return False

            cache[s] = True
            return True

        d = {}
        for i, word in enumerate(words):
            d[word] = i

        for i, word in enumerate(words):

            if not word:
                continue

            if isPalindrome(word):
                if '' in d:
                    ans.append([i, d['']])
                    ans.append([d[''], i])

            t = reverse(word)
            if t in d:
                if i != d[t]:
                    ans.append([i, d[t]])
                    # ans.append([d[t], i])

            for cur in range(1, len(word)):
                curw = word[:cur]
                curw2 = word[cur:]

                t = reverse(curw2)

                if isPalindrome(curw) and t in d:
                    if i != d[t]:
                        ans.append([d[t], i])

                t = reverse(curw)
                if isPalindrome(curw2) and t in d:
                    if i != d[t]:
                        ans.append([i, d[t]])
        return ans

    def findMinHeightTrees(self, n, edges: list):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        # 310
        if n == 1:
            return [0]

        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)

        leaves = [i for i in range(n) if len(adj[i]) == 1]

        while n > 2:
            n -= len(leaves)
            newLeaves = []
            for i in leaves:
                j = adj[i].pop()
                adj[j].remove(i)
                if len(adj[j]) == 1:
                    newLeaves.append(j)
            leaves = newLeaves
        return leaves

    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ans = 0
        for i in nums:
            ans = ans ^ i
        return ans

    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        d = {}
        for i in s:
            d[i] = d.get(i, 0) + 1
        s = list(s)
        s.sort(key=lambda x: (d[x], x), reverse=True)
        return ''.join(s)


if __name__ == '__main__':
    sol = Solution()
    #
    # print(sol.isPerfectSquare(104976))
    # print(sol.isPerfectSquare(16))
    # print(sol.isPerfectSquare(1))
    # print(sol.isPerfectSquare(4))
    # a = Trie()
    # a.insert("a")
    # print(a.search("a"))
    # print(a.startsWith("a"))
    # print(sol.searchMatrix([[1, 3, 5]], 2))
    # print(sol.findAnagrams('abab', 'ab'))
    # print(sol.findRestaurant(["Shogun", "Tapioca Express", "Burger King", "KFC"],
    #                          ["KFC", "Burger King", "Tapioca Express", "Shogun"]))
    # print(sol.evalRPN(["3", "-4", "+"]))
    # print(sol.evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]))

    ##    print(sol.compareVersion('1.1', '1.02'))
    ##    print(sol.compareVersion("1.2", "1.10"))
    # print(sol.findRelativeRanks([5, 4, 3, 2, 1]))
    # print(sol.findRelativeRanks([10, 3, 8, 9, 4]))
    # print(sol.constructRectangle(20))
    # print(sol.lastRemaining(24))
    # print(sol.findSubsequences([4, 6, 7, 7, 12, 124124, 12, 321, 31, 24, 124, 1, 3, 8, 12]))
    # nums = [1, 2, 3, 4, 5, 6, 7, 8]
    # nums = [1, 2, 3, 4]
    # head = ListNode(1)
    # p = head
    # for i in nums:
    #     p.next = ListNode(i)
    #     p = p.next
    # p.next = None
    # ans = sol.oddEvenList(head.next)
    # while ans:
    #     print(ans.val)
    #     ans = ans.next
    # root = sol.buildTree([4, 2, 5, 1, 7, 3], [4, 5, 2, 7, 3, 1])
    # root = sol.buildTree([10, 5, 2, 10, 3], [10, 2, 3, 10, 5])
    # sol.travel_inorder(root)
    # print(sol.numberOfArithmeticSlices([1, 2, 3, 4]))
    # print(sol.generateParenthesis(3))
    # print(sol.strStr('123456789', '123456789'))
    # print(sol.getPermutation(3, 6))
    # print(sol.findLongestWord('abpcplea', ["ale", "apple", "monkey", "plea"]))
    # print(sol.findLongestWord('000', ["a", "b", "c", "d"]))
    # print(sol.checkEqualTree(root))
    # print(sol.decodeString("3[a]2[bc]"))
    # print(sol.decodeString("3[a2[c]]"))
    # print(sol.decodeString("2[abc]3[cd]ef"))

    # print(sol.findNthDigit(10))
    # print(sol.findNthDigit(11))
    # print(sol.findNthDigit(190))
    # print(sol.findNthDigit(191))
    # print(sol.findNthDigit(195))
    # print(sol.findNthDigit(1000))
    # v = TreeLinkNode(1)
    # v.left = TreeLinkNode(2)
    # v.right = TreeLinkNode(3)
    # s = sol.connect(v)
    # print(s)

    # root = TreeNode(0)
    # root.right = TreeNode(0)
    # print(sol.findMode(root))
    # v = sol.insert([Interval(1, 3), Interval(6, 9)], Interval(2, 5))
    # for i in v:
    #     print(i.start, i.end)
    # nums = [
    #     [3, 4, 5],
    #     [3, 2, 6],
    #     [2, 2, 1]
    # ]
    # nums = [
    #     [9, 9, 4],
    #     [6, 6, 8],
    #     [2, 1, 1]
    # ]
    # print(sol.longestIncreasingPath(nums))

    # print(sol.palindromePairs(["a", "b", "c", "ab", "ac", "aa"]))
    # print(sol.palindromePairs(["bat", "tab", "cat"]))
    # print(sol.palindromePairs(["a", ""]))
    # print(sol.findMinHeightTrees(4, [[1, 0], [1, 2], [1, 3]]))
    # print(sol.findMinHeightTrees(6, [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]))

    print(sol.frequencySort("loveleetcode"))
