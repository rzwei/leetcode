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


class Node:
    def __init__(self, c):
        self.next = {}
        self.c = c
        self.v = 0

    def __str__(self):
        return str(self.v) + ' ' + str(self.c)


class MapSum:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node('-')

    def insert(self, key, val):
        """
        :type key: str
        :type val: int
        :rtype: void
        """
        cur = self.root
        for ki in key:
            if ki not in cur.next:
                cur.next[ki] = Node(ki)
            cur = cur.next[ki]
        cur.v = val

    def _sum(self, p):
        if not p:
            return 0
        r = p.v
        for s in p.next.values():
            r += self._sum(s)
        return r

    def sum(self, prefix):
        """
        :type prefix: str
        :rtype: int
        """
        cur = self.root
        for pi in prefix:
            if pi in cur.next:
                cur = cur.next[pi]
            else:
                return 0
        return self._sum(cur)


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


class Codec:
    def __init__(self):
        self.cache = {}
        self.n = 0

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.

        :type longUrl: str
        :rtype: str
        """
        self.cache[self.n] = longUrl
        self.n += 1
        return str(self.n - 1)

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.

        :type shortUrl: str
        :rtype: str
        """
        return self.cache[int(shortUrl)]


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

    def makesquare(self, nums: list):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 475
        if not nums or len(nums) < 4:
            return False
        nums.sort(reverse=True)
        sums = sum(nums)
        if sums % 4 != 0:
            return False
        L = sums / 4
        for i in nums:
            if i > L:
                return False

        def dfs(sums, n, target):
            if n == len(nums):
                return sums[0] == target and sums[1] == target and sums[2] == target
            # print(sums, n)
            for i in range(4):
                if sums[i] + nums[n] > target:
                    continue
                sums[i] += nums[n]
                if dfs(sums, n + 1, target):
                    return True
                sums[i] -= nums[n]
            return False

        return dfs([0, 0, 0, 0], 0, L)

    def licenseKeyFormatting(self, S: str, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        ret = ''
        k = K
        for i in reversed(S):
            if i == '-':
                continue
            if k == 0:
                ret = '-' + ret
                k = K
            ret = i.upper() + ret
            k -= 1
        return ret

    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        # 486
        cache = {}

        def dp(s, e):
            if s == e:
                return nums[s]
            if (s, e) in cache:
                return cache[(s, e)]
            r = max(nums[s] - dp(s + 1, e), nums[e] - dp(s, e - 1))
            cache[(s, e)] = r
            return r

        return dp(0, len(nums) - 1) >= 0

    def findPoisonedDuration(self, timeSeries: list, duration):
        """
        :type timeSeries: List[int]
        :type duration: int
        :rtype: int
        """
        # 495
        # last = -1
        # ret = 0
        # for i in range(len(timeSeries)):
        #     if timeSeries[i] > last:
        #         last = timeSeries[i] + duration - 1
        #         ret += duration
        #     else:
        #         ret += timeSeries[i] + duration - 1 - last
        #         last = timeSeries[i] + duration - 1
        # return ret
        if len(timeSeries) == 0:
            return 0

        res = 0
        for i in range(len(timeSeries) - 1):
            if timeSeries[i] + duration > timeSeries[i + 1]:
                res += timeSeries[i + 1] - timeSeries[i]
            else:
                res += duration
        res += duration
        return res

    def nextGreaterElement(self, findNums, nums):
        """
        :type findNums: List[int]
        :type nums: List[int]
        :rtype: List[int]
        """
        # ret = [-1] * len(findNums)
        # d = {}
        # for i, v in enumerate(nums):
        #     d[v] = i
        # for i in range(len(findNums)):
        #     for j in range(d[findNums[i]] + 1, len(nums)):
        #         if nums[j] > findNums[i]:
        #             ret[i] = nums[j]
        #             break
        # return ret
        stack = []
        d = {}
        for n in nums:
            while stack and stack[-1] < n:
                d[stack.pop()] = n
            stack.append(n)
        res = [-1] * len(findNums)
        for i in range(len(findNums)):
            res[i] = d.get(findNums[i], -1)
        return res

    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 516
        cache = [[-1] * (len(s) + 1) for _ in range(len(s))]

        def dp(i, j):
            if i > j:
                return 0
            if i == j:
                return 1
            if cache[i][j] != -1:
                return cache[i][j]
            if s[i] == s[j]:
                return dp(i + 1, j - 1) + 2
            r = max(dp(i + 1, j), dp(i, j - 1))
            cache[i][j] = r
            return r

        return dp(0, len(s) - 1)

    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        # 498
        if not matrix: return []
        m, n = len(matrix), len(matrix[0])
        ans = []
        for i in range(m + n - 1):
            if i % 2 == 0:
                row = i if i <= m - 1 else m - 1
                col = 0 if i <= m - 1 else i - (m - 1)
                while col < n and row >= 0:
                    ans.append(matrix[row][col])
                    row -= 1
                    col += 1
            else:
                row = 0 if i <= n - 1 else i - (n - 1)
                col = i if i <= n - 1 else n - 1
                while col >= 0 and row < m:
                    ans.append(matrix[row][col])
                    row += 1
                    col -= 1
        return ans

    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        d = {'a': 1, 'b': 2, 'c': 2, 'd': 1, 'e': 0, 'f': 1, 'g': 1, 'h': 1, 'i': 0, 'j': 1, 'k': 1, 'l': 1, 'm': 2,
             'n': 2, 'o': 0, 'p': 0, 'q': 0, 'r': 0, 's': 1, 't': 0, 'u': 0, 'v': 2, 'w': 0, 'x': 2, 'y': 0, 'z': 2, }
        ret = []
        for word in words:
            s = -1
            f = True
            for i in word:
                i = i.lower()
                if s == -1:
                    s = d[i]
                if d[i] != s:
                    f = False
                    break
            if f:
                ret.append(word)
        return ret

    def nextGreaterElements_2(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 503
        stack = []
        Len = len(nums)
        ret = [-1] * Len
        for i in range(Len * 2):
            n = nums[i % Len]
            while stack and nums[stack[-1]] < n:
                ret[stack.pop()] = n
            if i < Len:
                stack.append(i)
        return ret

    def findLUSlength(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: int
        """
        return -1 if a == b else max(len(a), len(b))

    def complexNumberMultiply(self, n1, n2):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        tokens = n1.split('+')
        a = int(tokens[0])
        b = int(tokens[1][:-1])
        tokens = n2.split('+')
        c = int(tokens[0])
        d = int(tokens[1][:-1])
        A = a * c - b * d
        B = (a * d + c * b)
        return "{}+{}i".format(A, B)

    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 525
        Len = len(nums)
        for i in range(Len):
            if nums[i] == 0:
                nums[i] = -1
        d = {0: -1}
        s = 0
        m = 0
        for i in range(Len):
            s += nums[i]
            if s in d:
                m = max(m, i - d[s])
            else:
                d[s] = i
        return m

    def findMinDifference(self, timePoints):
        """
        :type timePoints: List[str]
        :rtype: int
        """
        # 539
        timeStamp = []
        for timepoint in timePoints:
            tokens = timepoint.split(':')
            h = int(tokens[0])
            m = int(tokens[1])
            timeStamp.append(h * 60 + m)
        timeStamp.sort()
        timeStamp.append(timeStamp[0] + 24 * 60)
        print(timeStamp)
        res = 24 * 60
        for i in range(len(timeStamp) - 1):
            res = min(res, timeStamp[i + 1] - timeStamp[i])
        return res

    def optimalDivision(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        # 553
        Len = len(nums)
        if Len == 1:
            return str(nums[0])
        ans = ''
        ans = str(nums[0])
        if Len == 2:
            return ans + '/' + str(nums[1])

        ans += '/(' + str(nums[1])
        for i in range(2, Len):
            ans += '/' + str(nums[i])
        ans += ')'
        return ans

    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 561
        nums.sort()
        Len = len(nums)
        res = 0
        for i in range(0, Len, 2):
            res += nums[i]
        return res

    def validSquare(self, p1, p2, p3, p4):
        """
        :type p1: List[int]
        :type p2: List[int]
        :type p3: List[int]
        :type p4: List[int]
        :rtype: bool
        """
        # 593
        # [1,1]
        # [5,3]
        # [3,5]
        # [7,7]
        distence = lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
        nums = [p1, p2, p3, p4]
        nums.sort()
        p0 = nums[0]
        p1 = nums[1]
        p2 = nums[2]
        p3 = nums[3]
        L = distence(p0, p1)
        if L == 0:
            return False
        v0 = [p1[0] - p0[0], p1[1] - p0[1]]
        v1 = [p2[0] - p0[0], p2[1] - p0[1]]
        if v0[0] * v1[0] + v0[1] * v1[1] != 0:
            return False
        if distence(p0, p2) != L or distence(p3, p2) != L or distence(p3, p1) != L:
            return False
        return True

    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """

        # 728
        def isValid(n):
            if n == 0:
                return False
            t = n
            while n:
                i = n % 10
                if i == 0:
                    return False
                if t % i != 0:
                    return False
                n //= 10
            return True

        ret = []
        for i in range(left, right + 1):
            if isValid(i):
                ret.append(i)
        return ret

    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 724
        Len = len(nums)
        sums = [0] * (Len + 1)
        s = 0
        for i in range(Len):
            s += nums[+i]
            sums[i + 1] = s
        ret = -1
        for i in range(Len):
            if sums[i] == sums[Len] - sums[i + 1]:
                ret = i
                break
        return ret

    def trimBST(self, root: TreeNode, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        # 669
        if not root:
            return None

        if L <= root.val <= R:
            root.left = self.trimBST(root.left, L, R)
            root.right = self.trimBST(root.right, L, R)
            return root
        else:
            left = self.trimBST(root.left, L, R)
            if left:
                return left
            right = self.trimBST(root.right, L, R)
            if right:
                return right

        return None

    def distributeCandies(self, candies: list):
        """
        :type candies: List[int]
        :rtype: int
        """
        # 575
        Len = len(set(candies))
        if Len >= len(candies) // 2:
            return len(candies) // 2
        else:
            return Len

    def isOneBitCharacter(self, bits):
        """
        :type bits: List[int]
        :rtype: bool
        """
        s = 0
        Len = len(bits)
        while s < Len:
            if bits[s] == 0:
                s += 1
                if s == Len:
                    return True
            else:
                s += 2
                if s == Len:
                    return False


if __name__ == '__main__':
    sol = Solution()
    print(sol.isOneBitCharacter([1, 0, 1,0]))
    # print(sol.pivotIndex([1, 7, 3, 6, 5, 6]))
    # print(sol.pivotIndex([-1, -7, -3, -6, -5, -6]))
    # m = MapSum()
    # m.insert('apple', 3)
    # print(m.sum('ap'))
    # m.insert('app', 2)
    # print(m.sum('ap'))

    # print(sol.selfDividingNumbers(1, 22))
    # print(sol.validSquare([1, 1], [5, 3], [3, 5], [7, 7]))
    # print(sol.validSquare([0, 0], [1, 1], [1, 0], [0, 1]))
    # print(sol.optimalDivision([1, 2, 3]))
    # print(sol.findMinDifference(["23:59", "00:00"]))
    # print(sol.findMaxLength([1, 0, 0, 1]))
    # print(sol.complexNumberMultiply('1+1i', '1+1i'))
    # print(sol.findLUSlength("aba", "cdc"))
    # print(sol.nextGreaterElements_2([100, 1, 11, 1, 120, 111, 123, 1, -1, -100]))
    # print(sol.findWords(["Hello", "Alaska", "Dad", "Peace"]))
    # print(sol.findDiagonalOrder([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ])
    # )

    # print(sol.longestPalindromeSubseq("bbbab"))
    # print(sol.nextGreaterElement([4, 1, 2], [1, 3, 4, 2]))
    # print(sol.nextGreaterElement([2, 4], [1, 2, 3, 4]))
    # print(sol.findPoisonedDuration([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1))
    # print(sol.findPoisonedDuration([1, 2], 2))
    # print(sol.findPoisonedDuration([1, 4], 2))
    # print(sol.findPoisonedDuration([1, 2, 3, 4, 5], 5))
    # print(sol.PredictTheWinner([1, 5, 233, 7]))
    # print(sol.licenseKeyFormatting('2-4A0r7-4k', 3))
    # print(sol.makesquare([3, 1, 3, 3, 10, 7, 10, 3, 6, 9, 10, 3, 7, 6, 7]))
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
