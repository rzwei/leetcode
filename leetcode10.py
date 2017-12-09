import collections
import heapq
from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


# Employee info
# class Employee(object):
#     def __init__(self, id, importance, subordinates):
#         # It's the unique id of each node.
#         # unique id of this employee
#         self.id = id
#         # the importance value of this employee
#         self.importance = importance
#         # the id of direct subordinates
#         self.subordinates = subordinates
#

# class TreeNode:
#     def __init__(self, s, e):
#         self.start = s
#         self.end = e
#         self.right = None
#         self.left = None
#
#
# class MyCalendar:
#     def __init__(self):
#         self.times = {}
#         self.books = TreeNode(-1, -1)
#
#     def search(self, p, s, e):
#         if p.start >= e:
#             if not p.left:
#                 p.left = TreeNode(s, e)
#                 return True
#             else:
#                 return self.search(p.left, s, e)
#         if p.end <= s:
#             if not p.right:
#                 p.right = TreeNode(s, e)
#                 return True
#             else:
#                 return self.search(p.right, s, e)
#         return False
#
#     def book(self, start, end):
#         """
#         :type start: int
#         :type end: int
#         :rtype: bool
#         """
#         return self.search(self.books, start, end)


# class MyCalendar:
#     def __init__(self):
#         self.start = []
#         self.end = []
#
#     def book(self, start, end):
#         i = bisect.bisect_right(self.end, start)
#         j = bisect.bisect_left(self.start, end)
#         if i == j:
#             self.start.index(i, start)
#             self.end.index(i, end)
#             return True
#         return False


def buildList(nums: List):
    dummy = ListNode(-1)
    p = dummy
    for i in nums:
        p.next = ListNode(i)
        p = p.next
    return dummy.next


def showList(head: ListNode):
    while head:
        print(head.val, end='->')
        head = head.next
    print()


class Solution:
    def areSentencesSimilar(self, words1, words2, pairs):
        """
        :type words1: List[str]
        :type words2: List[str]
        :type pairs: List[List[str]]
        :rtype: bool
        """
        # 734. Sentence Similarity
        if len(words1) != len(words2):
            return False
        d = {}
        for l, r in pairs:
            if l not in d:
                d[l] = {r}
            else:
                d[l].add(r)
            if r not in d:
                d[r] = {l}
            else:
                d[r].add(l)
        for w1, w2 in zip(words1, words2):
            if w1 == w2 or w1 in d.get(w2, []):
                continue
            return False
        return True

    def areSentencesSimilarTwo(self, words1, words2, pairs):
        """
        :type words1: List[str]
        :type words2: List[str]
        :type pairs: List[List[str]]
        :rtype: bool
        """
        if len(words1) != len(words2):
            return False
        d = {}

        def father(x):
            if x != d[x]:
                return father(d[x])
            else:
                return x

        def union(x, y):
            x = father(x)
            y = father(y)
            d[x] = y

        for w1, w2 in pairs:
            if w1 not in d:
                d[w1] = w1
            if w2 not in d:
                d[w2] = w2
            union(w1, w2)
        for w1, w2 in zip(words1, words2):
            if w1 == w2:
                continue
            if w1 not in d or w2 not in d:
                return False
            if father(w1) == father(w2):
                continue
            print(w1, w2)
            return False
        return True

    def smallestRange(self, nums):
        """
        :type nums: List[List[int]]
        :rtype: List[int]
        """
        # 632. Smallest Range
        pq = [(row[0], i, 0) for i, row in enumerate(nums)]
        heapq.heapify(pq)

        right = max(row[0] for row in nums)

        ans = -1e9, 1e9

        while pq:
            left, i, j = heapq.heappop(pq)
            if right - left < ans[1] - ans[0]:
                ans = left, right
            if j + 1 == len(nums[i]):
                return ans
            v = nums[i][j + 1]
            right = max(right, v)
            heapq.heappush(pq, (v, i, j + 1))

    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 697. Degree of an Array
        c = collections.Counter(nums)
        degree = max(c.values())
        if degree == 1:
            return 1
        first, last = {}, {}
        for i, v in enumerate(nums):
            if v not in first:
                first[v] = i
            last[v] = i
        return min(last[v] - first[v] + 1 for v in c if c[v] == degree)

    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        # 739. Daily Temperatures
        stack = []
        d = {}
        for i, v in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < v:
                top = stack.pop()
                d[top] = i - top
            stack.append(i)
        ret = [0] * len(temperatures)
        for i in range(len(temperatures)):
            ret[i] = d.get(i, 0)
        return ret

    # def getImportance(self, employees: List[Employee], id) -> int:
    #     """
    #     :type employees: Employee
    #     :type id: int
    #     :rtype: int
    #     """
    #     # 690. Employee Importance
    #     d = {}
    #     for employee in employees:
    #         d[employee.id] = employee
    #     s = [id]
    #     res = 0
    #     while s:
    #         cur = d[s.pop()]
    #         res += cur.importance
    #         for sub in cur.subordinates:
    #             s.append(sub)
    #     return res

    def hasAlternatingBits(self, n: int) -> bool:
        """
        :type n: int
        :rtype: bool
        """
        # 693. Binary Number with Alternating Bits
        f = n & 1
        n >>= 1
        while n:
            if not f ^ (n & 1):
                return False
            f = n & 1
            n >>= 1
        return True

    def findLength(self, A: List[int], B: List[int]) -> int:
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        # 718. Maximum Length of Repeated Subarray
        LenA = len(A)
        LenB = len(B)
        dp = [[0] * (LenA + 1) for _ in range(LenB + 1)]
        ret = 0
        for i in range(1, LenA + 1):
            for j in range(1, LenB + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    ret = max(ret, dp[i][j])
        return ret

    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        d = {}
        for word in words:
            d[word] = d.get(word, 0) + 1
        q = [(-v, word) for word, v in d.items()]
        heapq.heapify(q)
        return [heapq.heappop(q)[1] for _ in range(k)]

    def knightProbability(self, N, K, r, c):
        """
        :type N: int
        :type K: int
        :type r: int
        :type c: int
        :rtype: float
        """
        # 688. Knight Probability in Chessboard
        dirs = [(1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (-2, 1), (2, -1), (-2, -1)]
        cache = {}

        def dp(x, y, k):
            if not (0 <= x < N and 0 <= y < N):
                return 0
            if (x, y, k) in cache:
                return cache[x, y, k]
            if k == 0:
                return 1
            r = 0
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                r += dp(nx, ny, k - 1)
            cache[(x, y, k)] = r
            return r

        r = dp(r, c, K)
        return r / (8 ** K)

    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        # 6. ZigZag Conversion
        if numRows == 1 or numRows >= len(s):
            return s

        L = [''] * numRows
        index, step = 0, 1

        for x in s:
            L[index] += x
            if index == 0:
                step = 1
            elif index == numRows - 1:
                step = -1
            index += step

        return ''.join(L)

    def isValidSudoku(self, board: List[List[str]]):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # 36. Valid Sudoku
        L1 = [[0] * 10 for _ in range(10)]
        L2 = [[0] * 10 for _ in range(10)]
        L3 = [[0] * 10 for _ in range(10)]
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    n = int(board[i][j])
                    if L1[i][n] == 1:
                        return False
                    L1[i][n] = 1

                    if L2[j][n] == 1:
                        return False
                    L2[j][n] = 1
                    k = i // 3 * 3 + j // 3

                    if L3[k][n] == 1:
                        return False
                    L3[k][n] = 1
        return True

    #     def valid(r, c):
    #         L = [0] * 10
    #         for i in range(3):
    #             for j in range(3):
    #                 nx = r + i
    #                 ny = j + c
    #                 #print(nx, ny)
    #                 if board[nx][ny] != '.':
    #                     n = int(board[nx][ny])
    #                     if L[n] == 1:
    #                         return False
    #                     L[n] = 1
    #         return True
    #
    #     for i in range(9):
    #         L1 = [0] * 10
    #         L2 = [0] * 10
    #         for j in range(9):
    #             if board[i][j] != '.':
    #                 n = int(board[i][j])
    #                 if L1[n] == 1:
    #                     return False
    #
    #                 L1[n] = 1
    #             if board[j][i]!='.':
    #                 n = int(board[j][i])
    #                 if L2[n] == 1:
    #                     return False
    #                 L2[n] = 1
    #     for i in range(3):
    #         for j in range(3):
    #             if not valid(i, j):
    #                 return False
    #     return True

    def solveSudoku(self, board: List[List[str]]):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """

        # 37. Sudoku Solver

        def valid(i, j, v):
            v = str(v)
            for k in range(9):
                if board[i][k] == v:
                    return False

                if board[k][j] == v:
                    return False

                nx = i // 3 * 3 + k // 3
                ny = j // 3 * 3 + k % 3
                if board[nx][ny] == v:
                    return False

            return True

        def dfs(x, y):
            if x == 9 and y == 0:
                return True

            if board[x][y] != '.':
                if x == 8 and y == 8:
                    return True
                if y + 1 == 9:
                    return dfs(x + 1, 0)
                else:
                    return dfs(x, y + 1)

            L = [0] * 9
            for i in range(9):
                if board[x][i] != '.':
                    n = int(board[x][i]) - 1
                    L[n] = 1
                if board[i][y] != '.':
                    n = int(board[i][y]) - 1
                    L[n] = 1
                nx = x // 3 * 3 + i // 3
                ny = y // 3 * 3 + i % 3
                if board[nx][ny] != '.':
                    n = int(board[nx][ny]) - 1
                    L[n] = 1
            for i, li in enumerate(L):
                if li == 0:
                    if not valid(x, y, i + 1):
                        continue
                    board[x][y] = str(i + 1)
                    nx = x
                    ny = y + 1
                    if ny == 9:
                        nx += 1
                        ny = 0

                    if dfs(nx, ny):
                        return True

                    board[x][y] = '.'
            return False

        dfs(0, 0)
        # return board

    def reverseKGroup(self, head: ListNode, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """

        # 25. Reverse Nodes in k-Group

        def work(p, K):
            if not p:
                return

            k = K
            h = p
            tail = None
            while k and p:
                tail = p
                p = p.next
                k -= 1
            if k != 0:
                return h

            pre = work(tail.next, K)
            k = K
            p = h
            while k:
                t = p.next
                p.next = pre
                pre = p
                p = t
                k -= 1
            return pre

        return work(head, k)

    def firstMissingPositive(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        # 41. First Missing Positive

        Len = len(A)

        for i in range(Len):
            while 0 < A[i] <= Len and A[A[i] - 1] != A[i]:
                t = A[A[i] - 1]
                A[A[i] - 1] = A[i]
                A[i] = t
                # A[i], A[A[i] - 1] = A[A[i] - 1], A[i]

        for i in range(Len):
            if i + 1 != A[i]:
                return i + 1
        return Len + 1

    def trap(self, height: List[int]):
        """
        :type height: List[int]
        :rtype: int
        """
        # 42. Trapping Rain Water

        i = 0
        j = len(height) - 1
        maxLeft = 0
        maxRight = 0
        res = 0
        while i <= j:
            if height[i] <= height[j]:
                if height[i] >= maxLeft:
                    maxLeft = height[i]
                else:
                    res += maxLeft - height[i]
                i += 1
            else:
                if height[j] >= maxRight:
                    maxRight = height[j]
                else:
                    res += maxRight - height[j]
                j -= 1
        return res

    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        # 44. Wildcard Matching
        # sLen = len(s)
        # pLen = len(p)
        # cache = set()
        #
        # def match(i, j):
        #     # if i == sLen and j == pLen:
        #     #     return True
        #
        #     # if i >= sLen and j == pLen - 1 and p[j] == '*':
        #     #     return True
        #     # if j >= pLen:
        #     #     return False
        #
        #     if i == sLen and j == pLen:
        #         return True
        #
        #     if i > sLen or j >= pLen:
        #         return False
        #
        #     if (i, j) in cache:
        #         return False
        #     if p[j] == '*':
        #         r = match(i, j + 1) or match(i + 1, j)
        #     elif p[j] == '?':
        #         r = match(i + 1, j + 1)
        #     else:
        #         if i == sLen:
        #             r = False
        #         elif s[i] == p[j]:
        #             r = match(i + 1, j + 1)
        #         else:
        #             r = False
        #     if r:
        #         return True
        #     else:
        #         cache.add((i, j))
        #         return False
        #
        # return match(0, 0)
        si = 0
        pi = 0
        last_match = 0
        last_star = -1

        while si < len(s):
            if pi < len(p) and (s[si] == p[pi] or p[pi] == '?'):
                si += 1
                pi += 1
            elif pi < len(p) and p[pi] == '*':
                last_match = si
                last_star = pi
                pi += 1
            elif last_star != -1:
                pi = last_star + 1
                si = last_match + 1
                last_match += 1
            else:
                return False
        while pi < len(p) and p[pi] == '*':
            pi += 1

        return pi == len(p)

    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 52. N-Queens II
        path = [-1] * n

        def judge(path, n, v):
            for i in range(n):
                if path[i] == v or i + path[i] == v + n or i - path[i] == n - v:
                    return False
            return True

        def dfs(path, ni):
            if ni == n:
                return 1
            r = 0
            for i in range(n):
                if judge(path, ni, i):
                    path[ni] = i
                    r += dfs(path, ni + 1)
            return r

        r = dfs(path, 0)
        return r

    def minWindow(self, s: str, t: str) -> str:
        """
        :type s: str
        :type t: str
        :rtype: str
        # 76. Minimum Window Substring
        """
        # if len(s) < len(t):
        #     return ''
        # d = {}
        # for i in t:
        #     d[i] = d.get(i, 0) + 1
        # l = 0
        # r = 0
        # Len = len(s)
        # t = {}
        #
        # def valid(t):
        #     for k, v in d.items():
        #         if v > t.get(k, 0):
        #             return False
        #     return True
        #
        # ret = ""
        # while r < Len:
        #     t[s[r]] = t.get(s[r], 0) + 1
        #     while valid(t):
        #         m = s[l:r + 1]
        #         if not ret or len(m) < len(ret):
        #             ret = m
        #         t[s[l]] -= 1
        #         l += 1
        #     r += 1
        # return ret
        mapT = collections.defaultdict(int)
        for char in t:
            mapT[char] += 1
        missing = len(t)
        i = start = end = 0
        for j, c in enumerate(s, 1):
            missing -= mapT[c] > 0
            mapT[c] -= 1
            if not missing:
                while start < j and mapT[s[i]] < 0:
                    mapT[s[i]] += 1
                    i += 1
                if end == 0 or j - i <= end - start:
                    start, end = i, j
        return s[start:end]

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        84. Largest Rectangle in Histogram
        """
        heights.append(0)
        ret = 0
        s = []
        for i, hi in enumerate(heights):
            while s and heights[s[-1]] > hi:
                h = heights[s.pop()]
                w = i if not s else i - s[-1] - 1
                ret = max(ret, h * w)
            s.append(i)
        return ret

    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        97. Interleaving String
        """
        if len(s1) + len(s2) != len(s3):
            return False

        Len1 = len(s1)
        Len2 = len(s2)
        Len3 = len(s3)
        cache = set()

        def dfs(i, j, k):

            if i == Len1 and j == Len2 and k == Len3:
                return True

            if (i, j, k) in cache:
                return False

            if i < Len1 and k < Len3 and s1[i] == s3[k]:
                if dfs(i + 1, j, k + 1):
                    return True

            if j < Len2 and k < Len3 and s2[j] == s3[k]:
                if dfs(i, j + 1, k + 1):
                    return True
            cache.add((i, j, k))
            return False

        return dfs(0, 0, 0)

    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        87. Scramble String
        """

        if s1 == s2:
            return True

        count = [0] * 26
        for i, j in zip(s1, s2):
            count[ord(i) - ord('a')] += 1
            count[ord(j) - ord('a')] -= 1
        for i in count:
            if i != 0:
                return False
        Len = len(s1)
        for i in range(1, Len):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
                return True
            if self.isScramble(s1[:i], s2[-i:]) and self.isScramble(s1[i - Len:], s2[:Len - i]):
                return True

        return False

    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        85. Maximal Rectangle
        """
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        h = [0] * (n + 1)
        ret = 0
        s = [0]
        for row in range(m):
            for j in range(n):
                if matrix[row][j] == 1:
                    h[j] += 1
                else:
                    h[j] = 0
            s.clear()
            s.append(0)
            for i, hi in enumerate(h):
                while s and h[s[-1]] > hi:
                    H = h[s.pop()]
                    W = i - 1 - s[-1] if s else i
                    ret = max(ret, W * H)
                s.append(i)
        return ret

    def recoverTree(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        99. Recover Binary Search Tree
        """
        self.first = None
        self.second = None
        self.prev = TreeNode(-2147483648)

        def traverse(p):
            if not p:
                return
            traverse(p.left)
            if not self.first and self.prev.val >= p.val:
                self.first = self.prev
            if self.first and self.prev.val >= p.val:
                self.second = p
            self.prev = p
            traverse(p.right)

        traverse(root)
        self.first.val, self.second.val = self.second.val, self.first.val

    def postorderTraversal(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: List[int]
        145. Binary Tree Postorder Traversal
        """
        s = collections.deque([(root, 0)])
        ret = []
        while s:
            cur, f = s.pop()
            if not cur:
                continue
            if f == 0:
                s.append((cur, 1))
                if cur.left:
                    s.append((cur.right, 0))
                if cur.right:
                    s.append((cur.left, 0))
            else:
                ret.append(cur.val)
        return ret

    def deleteAndEarn(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        740. Delete and Earn
        """

        d = [0] * 10003
        for i in nums:
            d[i] += i
        dp = [0] * 10003
        dp[1] = d[1]
        dp[2] = max(d[1], d[2])

        for i in range(3, 10001):
            dp[i] = max(dp[i - 2] + d[i], dp[i - 1])
        return dp[10000]

if __name__ == '__main__':
    sol = Solution()
    print(sol.deleteAndEarn([3, 4, 2]))
    # root = TreeNode(0)
    # root.left = TreeNode(1)
    # root.right = TreeNode(2)
    # print(root.val)
    # print(root.left.val)
    # print(root.right.val)
    # sol.recoverTree(root)
    # print(root.val)
    # print(root.left.val)
    # print(root.right.val)
    # matrix = [
    #     [1, 0, 1, 0, 0],
    #     [1, 0, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, 0, 0, 1, 0]
    # ]
    # print(sol.maximalRectangle(matrix))
    # print(sol.isScramble('abc', 'bca'))
    # print(sol.isScramble("dbdac", "abcdd"))
    # print(sol.isInterleave('aabcc', 'dbbca', 'aadbbcbcac'))
    # print(sol.isInterleave('aabcc', 'dbbca', 'aadbbbaccc'))
    # print(sol.largestRectangleArea([2, 1, 5, 6, 2, 3]))
    # print(sol.largestRectangleArea([1]))
    # print(sol.minWindow('ADOBECODEBANC', 'ABC'))
    # print(sol.totalNQueens(8))
    # print(sol.isMatch("abbbbbbbaabbabaabaa", "*****a*ab"))
    # print(sol.isMatch("aa", "a"))
    # print(sol.isMatch("a", "aa"))
    # print(sol.isMatch("aa", "aa"))
    # print(sol.isMatch("aa", "*"))
    # print(sol.isMatch("aa", "a*"))
    # print(sol.isMatch("ab", "?*"))
    # print(sol.isMatch("aab", "c*a*b"))
    # print(sol.isMatch("abefcdgiescdfimde", "ab*cd?i*de"))
    # print(sol.isMatch("ho", "ho**"))

    # print(sol.areSentencesSimilarTwo(["great", "acting", "skills"],
    #                                  ["fine", "painting", "talent"],
    #                                  [["great", "fine"], ["drama", "acting"], ["skills", "talent"]]))
    # print(sol.findShortestSubArray([1, 2, 2, 3, 1]))
    # print(sol.findShortestSubArray([1, 2, 2, 3, 1, 4, 2]))
    # print(sol.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]))
    # print(sol.hasAlternatingBits(4))
    # print(sol.hasAlternatingBits(5))
    # print(sol.findLength([1, 2, 3, 2, 1], [3, 2, 1, 4, 7]))
    # print(sol.hasAlternatingBits(4))
    # print(sol.hasAlternatingBits(5))
    # print(sol.findLength([1, 2, 3, 2, 1], [3, 2, 1, 4, 7]))
    # print(sol.topKFrequent(["i", "love", "leetcode", "i", "love", "coding"], 2))
    # print(sol.topKFrequent(["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], 4))
    # print(sol.knightProbability(8, 30, 6, 4))
    # board = [[".", ".", ".", ".", ".", ".", ".", ".", "."],
    #          [".", ".", ".", ".", ".", ".", "3", ".", "."],
    #          [".", ".", ".", "1", "8", ".", ".", ".", "."],
    #          [".", ".", ".", "7", ".", ".", ".", ".", "."],
    #          [".", ".", ".", ".", "1", ".", "9", "7", "."],
    #          [".", ".", ".", ".", ".", ".", ".", ".", "."],
    #          [".", ".", ".", "3", "6", ".", "1", ".", "."],
    #          [".", ".", ".", ".", ".", ".", ".", ".", "."],
    #          [".", ".", ".", ".", ".", ".", ".", "2", "."]
    #          ]
    # print(sol.isValidSudoku(board))
    # print(sol.knightProbability(3, 2, 1, 2))
    # print(sol.convert("PAYPALISHIRING", 3))
    # board = [[".", ".", "9", "7", "4", "8", ".", ".", "."], ["7", ".", ".", ".", ".", ".", ".", ".", "."],
    #          [".", "2", ".", "1", ".", "9", ".", ".", "."], [".", ".", "7", ".", ".", ".", "2", "4", "."],
    #          [".", "6", "4", ".", "1", ".", "5", "9", "."], [".", "9", "8", ".", ".", ".", "3", ".", "."],
    #          [".", ".", ".", "8", ".", "3", ".", "2", "."], [".", ".", ".", ".", ".", ".", ".", ".", "6"],
    #          [".", ".", ".", "2", "7", "5", "9", ".", "."]]
    # sol.solveSudoku(board)
    # for line in board:
    #     print(line)
    # head = buildList([1, 2, 3, 4, 5])
    # r = sol.reverseKGroup(head, 3)
    # showList(r)
    # print(sol.firstMissingPositive([1, 2, 0]))
    # print(sol.firstMissingPositive([3, 4, -1, 1]))
    # print(sol.firstMissingPositive([1]))
    # print(sol.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
