import heapq
import random


class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

    def __str__(self):
        return str(self.start) + ' ' + str(self.end)


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def heapAdjust(nums, i, size):
    if i >= size // 2:
        return
    l = i * 2 + 1
    r = i * 2 + 2
    m = i
    if l < size and nums[m] < nums[l]:
        m = l
    if r < size and nums[m] < nums[r]:
        m = r
    if m != i:
        nums[m], nums[i] = nums[i], nums[m]
        heapAdjust(nums, m, size)


def buildHeap(nums):
    size = len(nums)
    for i in reversed(range(size // 2)):
        heapAdjust(nums, i, size)


def headpSort(nums):
    buildHeap(nums)
    for i in range(len(nums)):
        heapAdjust(nums, 0, len(nums) - i)
        nums[0], nums[len(nums) - i - 1] = nums[len(nums) - i - 1], nums[0]


class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def copyRandomList(slf, head: RandomListNode):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        # 138
        if not head:
            return None
        d = {}
        cur = head
        while cur:
            d[cur] = RandomListNode(cur.label)
            cur = cur.next
        for k, v in d.items():
            v.next = d.get(k.next, None)
            v.random = d.get(k.random, None)
        return d[head]

    def isSymmetric(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: bool
        """

        # 101

        def judge(left, right):
            if left is None and right is None:
                return True
            if left and right:
                if left.val != right.val:
                    return False
                return judge(left.left, right.right) and judge(left.right, right.left)

            return False

        if not root:
            return True
        return judge(root.left, root.right)

    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        # 165
        tokens1 = version1.split('.')
        tokens2 = version2.split('.')
        length = max(len(tokens1), len(tokens2))
        for i in range(length):
            c1 = tokens1[i] if i < len(tokens1) else "0"
            c2 = tokens2[i] if i < len(tokens2) else "0"
            if int(c1) != int(c2):
                return 1 if int(c1) - int(c2) > 0 else -1
        return 0

    def findRepeatedDnaSequences(self, dna):
        """
        :type s: str
        :rtype: List[str]
        """

        # 187
        ans = set()
        d = set()
        for i in range(len(dna) - 10 + 1):
            k = dna[i:i + 10]
            if k in d:
                ans.add(k)
            else:
                d.add(k)
        return list(ans)

    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # 215
        heap = nums[:k]
        heapq.heapify(heap)
        for i in nums[k:]:
            if i > heap[0]:
                heapq.heapreplace(heap, i)
        return heap[0]

    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # 242
        alphas = [0] * 26
        for i in s:
            alphas[ord(i) - ord('a')] += 1
        for i in t:
            alphas[ord(i) - ord('a')] -= 1
            if alphas[ord(i) - ord('a')] < 0:
                return False
        for i in alphas:
            if i != 0:
                return False
        return True

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        # 240
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        row = 0
        col = len(matrix[0]) - 1
        while row < len(matrix) and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        return False

    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        # 237
        node.val = node.next.val
        node.next = node.next.next

    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        # 221

        m = len(matrix)
        if m == 0:
            return 0
        n = len(matrix[0])
        v = min(m, n)

        def hasSquare(v):
            for x in range(m - v + 1):
                for y in range(n - v + 1):
                    flag = True
                    for dx in range(v):
                        if not flag:
                            break
                        for dy in range(v):
                            if matrix[x + dx][y + dy] != '1':
                                flag = False
                                break
                    if flag:
                        return True
            return False

        while v >= 0:
            if hasSquare(v):
                return v * v
            v -= 1
        return 0

    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        # 223

        c1 = (C - A) * (D - B)
        c2 = (G - E) * (H - F)
        c = c1 + c2
        dx = min(C, G) - max(A, E)
        dy = min(D, H) - max(B, F)
        if dx > 0 and dy > 0:
            c -= dx * dy
        return c

    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return n % 4 == 1

    def increasing(self, nums):
        dp = [0] * len(nums)

        dp[0] = nums[0]

        length = 1

        def binarysearch(nums, target):
            i = 0
            j = length - 1
            while i <= j:
                mid = (i + j) // 2
                if nums[mid] >= target:
                    j = mid - 1
                else:
                    i = mid + 1
            return i

        for i in range(1, len(nums)):
            if nums[i] > dp[length - 1]:
                dp[length] = nums[i]
                length += 1
            else:
                index = binarysearch(dp, nums[i])
                dp[index] = nums[i]
                # print(dp)
        return length

    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        # 350
        d = {}
        for i in nums1:
            d[i] = d.get(i, 0) + 1
        ans = []
        for i in nums2:
            if i in d and d[i] > 0:
                ans.append(i)
                d[i] -= 1
        return ans

    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 387
        d = {}
        for i in s:
            d[i] = d.get(i, 0) + 1
        for i, v in enumerate(s):
            if d[v] == 1:
                return i
        return -1

    def getMoneyAmount(self, n):
        """
        :type n: int
        :rtype: int
        """
        table = [[-1] * (n + 1) for _ in range(n + 1)]

        def dp(s, e):
            if s >= e:
                return 0
            if table[s][e] != -1:
                return table[s][e]
            res = 2147483647
            for x in range(s, e + 1):
                res = min(res, max(x + dp(s, x - 1), x + dp(x + 1, e)))
            table[s][e] = res
            return res

        return dp(1, n)

    # def __init__(self, head):
    #     """
    #     @param head The linked list's head.
    #     Note that the head is guaranteed to be not null, so it contains at least one node.
    #     :type head: ListNode
    #     """
    #     self.head = head

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        c = self.head
        r = c.val
        i = 1
        while c.next:
            c = c.next
            if random.randint(1, i + 1) == i:
                r = c.val
            i += 1
        return r

    # def __init__(self, nums):
    #     """
    #     :type nums: List[int]
    #     """
    #     self.nums = nums[:]

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        return self.nums

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        nums = self.nums[:]
        n = len(nums)
        for i in range(n):
            j = random.randint(0, i)
            nums[i], nums[j] = nums[j], nums[i]

        return nums

    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """

        # 401
        def bitCount(n):
            r = 0
            while n:
                if n & 1:
                    r += 1
                n >>= 1
            return r

        ret = []
        for h in range(0, 12):
            for m in range(0, 60):
                if bitCount(64 * h + m) == num:
                    ret.append("{:d}:{:02d}".format(h, m))
        return ret

    def originalDigits(self, s):
        """
        :type s: str
        :rtype: str
        """
        # 423
        words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

        m = {}
        for i, v in enumerate(words):
            m[v] = str(i)

        d = {}
        for i in s:
            if i not in d:
                d[i] = 1
            else:
                d[i] += 1

        def dfs(d: dict, ret):

            flag = True
            for v in d.values():
                if v != 0:
                    flag = False
                    break
            if flag:
                return True

            for word in words:
                flag = True
                for wi in word:
                    if wi not in d or d[wi] <= 0:
                        flag = False
                        break
                if flag:
                    for wi in word:
                        d[wi] -= 1
                    if dfs(d, ret):
                        ret.append(m[word])
                    else:
                        for wi in word:
                            d[wi] += 1
            flag = True
            for v in d.values():
                if v != 0:
                    flag = False
                    break
            if flag:
                return True
            return False

        ret = []
        dfs(d, ret)
        ret.sort()
        return ''.join(ret)

    def findRightInterval(self, intervals: list):
        """
        :type intervals: List[Interval]
        :rtype: List[int]
        """
        # 436


        if not intervals:
            return intervals
        d = {}
        for i, interval in enumerate(intervals):
            d[interval.start] = i
        ret = []
        for interval in intervals:
            if interval.end in d:
                ret.append(d[interval.end])
            else:
                ret.append(-1)
        return ret

    def findMinArrowShots(self, points: list):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        # 452
        if not points:
            return 0
        points.sort(key=lambda x: (x[1], x[0]))
        res = 1
        cur = points[0][1]
        for i in range(len(points)):
            s = points[i][0]
            e = points[i][1]
            if s <= cur:
                res -= 1
            else:
                cur = e
        return res


def binarysearch(nums, target):
    i = 0
    j = len(nums) - 1
    while i <= j:
        mid = (i + j) // 2
        if nums[mid] > target:
            j = mid - 1
        else:
            i = mid + 1
    return i, j


if __name__ == '__main__':
    sol = Solution()
    print(sol.findMinArrowShots([[10, 16], [2, 8], [1, 6], [7, 12]]))
    # intervals = [Interval(1, 4), Interval(2, 3), Interval(3, 4)]
    # print(sol.findRightInterval(intervals))
    # nums = [1, 2, 3, 4, 5, 6, 7]
    # dummy = RandomListNode(-1)
    # pd = dummy
    # for i in nums:
    #     pd.next = RandomListNode(i)
    #     pd = pd.next
    # pd = dummy
    # pd.next.next.next.next.next.random = pd.next.next.next
    # t = sol.copyRandomList(dummy.next)
    # print(sol.compareVersion('1.2', '1.1'))
    # print(sol.findRepeatedDnaSequences('AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT'))
    # print(sol.findRepeatedDnaSequences("AAAAAAAAAAA"))
    # print(sol.findKthLargest([3, 2, 1, 5, 6, 4], 2))
    # print(sol.findKthLargest([-1, -1], 2))
    # nums = [12, 321, 312, 312, 31, 23, 123, 12, 312, 3, 123, 123, 12, 312, 3]
    # headpSort(nums)
    # print(nums)
    # print(sol.isAnagram('ab', 'a'))
    # matrix = [
    #     [1, 4, 7, 11, 15],
    #     [2, 5, 8, 12, 19],
    #     [3, 6, 9, 16, 22],
    #     [10, 13, 14, 17, 24],
    #     [18, 21, 23, 26, 30]
    # ]
    # print(sol.searchMatrix(matrix, 5))
    # matrix = [
    #     ['1', '0', '1', '0', '0'],
    #     ['1', '0', '1', '1', '1'],
    #     ['1', '1', '1', '1', '1'],
    #     ['1', '0', '0', '1', '0'],
    # ]
    # print(sol.maximalSquare(matrix))
    # print(sol.computeArea(-3, 0, 3, 4, 0, -1, 9, 2))
    # print(sol.computeArea(0, 0, 0, 0, -1, -1, 1, 1))
    # print(sol.computeArea(-2,
    #                       -2,
    #                       2,
    #                       2,
    #                       -3,
    #                       -3,
    #                       3,
    #                       -1, ))
    # print(sol.increasing([1, 3, 54, 3, 2, 54, 7]))
    # print(sol.intersect([1, 2, 2, 1], [2, 2]))
    # print(sol.getMoneyAmount(4))
    # print(sol.readBinaryWatch(2))
    # print(sol.originalDigits('owoztneoer'))
    # print(sol.originalDigits('fviefurofour'))
