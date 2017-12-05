import collections
import heapq
from typing import List


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
        dp = [[0] * (LenA + 1) for _ in range(LenB+1)]
        ret = 0
        for i in range(1, LenA + 1):
            for j in range(1, LenB + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    ret = max(ret, dp[i][j])
        return ret

    def knightProbability(self, N, K, r, c):
        """
        :type N: int
        :type K: int
        :type r: int
        :type c: int
        :rtype: float
        """
        # 688. Knight Probability in Chessboard
        # dirs = [(-2, -1), (-1, -2), (-2, 1), (-1, 2), (1, 2), (2, 1), (-1, 2), (1, -2)]
        dirs = [(2, 1), (-2, 1), (2, -1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2), ]
        Map = set()

        res = [0, 0]

        def dfs(x, y, K):
            if not (0 <= x < N and 0 <= y < N):
                res[0] += 1
                return
            if (x, y) in Map:
                return
            if K == 0:
                res[1] += 1
                return
            # r = 0
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                dfs(nx, ny, K - 1)
            Map.add((x, y))

        dfs(r, c, K)
        return res[1] / (res[0] + res[1])


if __name__ == '__main__':
    sol = Solution()
    # print(sol.areSentencesSimilarTwo(["great", "acting", "skills"],
    #                                  ["fine", "painting", "talent"],
    #                                  [["great", "fine"], ["drama", "acting"], ["skills", "talent"]]))
    # print(sol.findShortestSubArray([1, 2, 2, 3, 1]))
    # print(sol.findShortestSubArray([1, 2, 2, 3, 1, 4, 2]))
    # print(sol.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]))
    # print(sol.hasAlternatingBits(4))
    # print(sol.hasAlternatingBits(5))
    print(sol.findLength([1, 2, 3, 2, 1], [3, 2, 1, 4, 7]))
    # print(sol.hasAlternatingBits(4))
    # print(sol.hasAlternatingBits(5))
    print(sol.knightProbability(3, 2, 0, 0))
