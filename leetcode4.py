import collections
import functools
import heapq


import itchat

class SegNode:
    def __init__(self):
        self.left = 0
        self.right = 0

        self.leftNode = None
        self.rightNode = None
        self.val = 0


class NumArray(object):
    def buildTree(self, l, r):
        if l > r:
            return None
        if l == r:
            node = SegNode()
            node.left = l
            node.right = r
            node.val = self.nums[l]
            return node
        mid = (l + r) // 2
        node = SegNode()
        node.left = l
        node.right = r
        node.leftNode = self.buildTree(l, mid)
        node.rightNode = self.buildTree(mid + 1, r)
        v = 0
        if node.leftNode:
            v += node.leftNode.val
        if node.rightNode:
            v += node.rightNode.val
        node.val = v
        return node

    def query(self, p: SegNode, l, r):
        if not p or p.right < l or p.left > r:
            return 0
        if p.left >= l and p.right <= r:
            return p.val
        # mid = (l + r) // 2
        return self.query(p.leftNode, l, r) + self.query(p.rightNode, l, r)

    def updateSegTree(self, p: SegNode, i, val):
        if not p or p.left > i or p.right < i:
            return 0
        if p.left == p.right == i:
            r = val - p.val
            p.val = val
            return r
        r = self.updateSegTree(p.leftNode, i, val) + self.updateSegTree(p.rightNode, i, val)
        p.val += r
        return r

    def __init__(self, nums):
        """
        :type nums: List[int]
        """

        self.nums = nums
        self.root = self.buildTree(0, len(nums) - 1)

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: void
        """
        self.updateSegTree(self.root, i, val)

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.query(self.root, i, j)


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution_382(object):
    def __init__(self, head):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        :type head: ListNode
        """
        self.root = head
        self.cur = head

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        r = self.cur.val
        self.cur = self.cur.next
        if not self.cur:
            self.cur = self.root
        return r


class TrieNode:
    def __init__(self):
        self.is_word = False
        self.next = {}


class WordDictionary_my(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        self.cache = {}

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """

        if '.' in word and word in self.cache:
            del self.cache[word]

        cur = self.root
        for i in word:
            if i not in cur.next:
                cur.next[i] = TrieNode()
            cur = cur.next[i]
        cur.is_word = True

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """

        def dfs(p: TrieNode, word):
            if not p:
                return False
            if not word:
                return p.is_word
            i = word[0]
            if i != '.':
                if i not in p.next:
                    return False
                else:
                    return dfs(p.next[i], word[1:])
            for v in p.next.values():
                if dfs(v, word[1:]):
                    return True
            return False

        if '.' in word:
            if word not in self.cache:
                self.cache[word] = dfs(self.root, word)
            return self.cache[word]
        # if word not in self.cache:
        #     self.cache[word] = dfs(self.root, word)
        #
        # return self.cache[word]
        return dfs(self.root, word)


class WordDictionary(object):
    def __init__(self):
        self.word_dict = collections.defaultdict(set)

    def addWord(self, word):
        if word:
            self.word_dict[word[0]].add(word)

    def search(self, word):
        if not word:
            return False
        if '.' not in word:
            return word in self.word_dict[word[0]]
        for v in self.word_dict[word[0]]:
            # match xx.xx.x with yyyyyyy
            for i, ch in enumerate(word):
                if ch != v[i] and ch != '.':
                    break
            else:
                return True
        return False


class MinStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.q = collections.deque()
        self.min = collections.deque()

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        curMin = self.getMin()

        if x < curMin:
            curMin = x
        self.q.append(x)
        self.min.append(curMin)

    def pop(self):
        """
        :rtype: void
        """
        self.q.pop()
        self.min.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.q[-1]

    def getMin(self):
        """
        :rtype: int
        """
        if len(self.q) == 0:
            return 2147483647
        return self.min[-1]

    # def averageOfLevels(self, root):
    #     """
    #     :type root: TreeNode
    #     :rtype: List[float]
    #     """
    #     layers = collections.OrderedDict()
    #
    #     def travel(p, l):
    #         if not p:
    #             return
    #         if l not in layers:
    #             layers[l] = []
    #         layers[l].append(p.val)
    #         travel(p.left, l + 1)
    #         travel(p.right, l + 1)
    #
    #     travel(root, 0)
    #     ans = []
    #     for i, v in layers:
    #         ans.append(sum(v) / len(v))
    #     return ans
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        queue = [root]
        ans = []
        while queue:
            t = []
            s = 0
            n = 0
            while queue:
                node = queue.pop()
                s += node.val
                n += 1
                if node.left:
                    t.append(node.left)
                if node.right:
                    t.append(node.right)
            queue.extend(t)
            ans.append(s / (n + 0.0))
        return ans


def binarySearch(i, j, nums, v):
    while i <= j:
        mid = (i + j) // 2
        vmid = nums[mid]
        if vmid == v:
            return mid
        if vmid < v:
            i = mid + 1
        else:
            j = mid - 1
    return i - 1


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class MedianFinder(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        # self.q = collections.deque()
        # self.q = []
        self.max = []
        self.min = []

    def addNum(self, num):
        heapq.heappush(self.max, num)
        v = heapq.heappop(self.max)

        # v = heapq.heappushpop(self.max, num)

        heapq.heappush(self.min, -v)
        if len(self.max) < len(self.min):
            heapq.heappush(self.max, -heapq.heappop(self.min))
        print(self.max, self.min)

    # def addNum(self, num):
    #     """
    #     :type num: int
    #     :rtype: void
    #     """
    #     i = 0
    #     j = len(self.q) - 1
    #     while i <= j:
    #         mid = (i + j) // 2
    #         v = self.q[mid]
    #         if v <= num:
    #             i = mid + 1
    #         else:
    #             j = mid - 1
    #     self.q.insert(i, num)
    #     # print(self.q)


    # def findMedian(self):
    #     """
    #     :rtype: float
    #     """
    #     ln = len(self.q)
    #     if ln % 2 == 1:
    #         return self.q[ln // 2]
    #     else:
    #         return (self.q[ln // 2 - 1] + self.q[ln // 2]) / 2

    def findMedian(self):
        if len(self.max) > len(self.min):
            return self.max[0]
        return (self.max[0] + self.min[0]) / 2.0


# class NumArray(object):
#     def __init__(self, nums):
#         """
#         :type nums: List[int]
#         """
#         # 303
#         self.sums = [0 for i in range(len(nums) + 1)]
#         s = 0
#         for i, v in enumerate(nums):
#             s += v
#             self.sums[i + 1] = s
#
#     def sumRange(self, i, j):
#         """
#         :type i: int
#         :type j: int
#         :rtype: int
#         """
#         return self.sums[j + 1] - self.sums[i]
#
#     def isMatch(self, s, p):
#         """
#         :type s: str
#         :type p: str
#         :rtype: bool
#         """
#         # 44
#         pass


class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def travel(p: TreeNode):
            if not p:
                return 0
            return max(travel(p.left), travel(p.right)) + 1

        return travel(root)

    def buildTree(self, inorder: list, postorder: list):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        # inorder and postorder

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

    # def buildTree(self, preorder, inorder: list):
    #     """
    #     :type preorder: List[int]
    #     :type inorder: List[int]
    #     :rtype: TreeNode
    #     """
    #     # 105
    #     # print(preorder, inorder)
    #     if not preorder or not inorder:
    #         return None
    #
    #     v = preorder.pop(0)
    #     node = TreeNode(v)
    #
    #     vi = inorder.index(v)
    #
    #     node.left = self.buildTree(preorder, inorder[:vi])
    #     node.right = self.buildTree(preorder, inorder[vi + 1:])
    #
    #     return node

    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 91
        if not s:
            return 0
        cache = {}

        cache[''] = 1

        def dfs(s):
            if s in cache:
                return cache[s]

            if len(s) <= 1:
                if 1 <= int(s) <= 9:
                    cache[s] = 1
                    return 1
                else:
                    cache[s] = 0
                    return 0

            r1 = int(s[0])
            r2 = int(s[:2])

            # if s[0] == '0':
            #     return 0

            if 1 <= r1 <= 9 and 10 <= r2 <= 26:
                cache[s] = dfs(s[1:]) + dfs(s[2:])
                return dfs(s[1:]) + dfs(s[2:])

            if 1 <= r1 <= 9:
                cache[s] = dfs(s[1:])
                return cache[s]

            if 10 <= r2 <= 26:
                cache[s] = dfs(s[2:])
                return cache[s]

            cache[s] = 0
            return 0

        return dfs(s)

    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        # 557
        tokens = s.split(' '
                         )

        for i in range(len(tokens)):
            tokens[i] = tokens[i][::-1]

        return ' '.join(tokens)

    def minSteps(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 650
        pass

    # def minSubArrayLen(self, s, nums):
    #     """
    #     :type s: int
    #     :type nums: List[int]
    #     :rtype: int
    #     """
    #     # 209
    #     sums = [0 for i in range(len(nums) + 1)]
    #
    #     c = 0
    #     for i in range(len(nums)):
    #         c += nums[i]
    #         sums[i + 1] = c
    #
    #     if c < s:
    #         return 0
    #
    #     def search(i, j, v):
    #         while i <= j:
    #             # print(i, j)
    #             mid = (i + j) // 2
    #             vi = sums[mid]
    #             if vi == v:
    #                 return mid
    #             if vi < v:
    #                 i = mid + 1
    #             else:
    #                 j = mid - 1
    #         return i - 1
    #
    #     ans = len(nums)
    #
    #     bi = search(0, len(sums) - 1, s)
    #
    #     for i in range(len(sums) - 1, bi - 1, -1):
    #         diff = sums[i] - s
    #         j = search(0, i - 1, diff)
    #         # print(j, i)
    #         if i - j < ans:
    #             ans = i - j
    #     return ans


    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        # 209
        if sum(nums) < s:
            return 0

        minn = len(nums)

        window = 0
        i = 0
        j = 0

        while i < len(nums):
            window += nums[i]
            i += 1

            while window >= s:
                if i - j < minn:
                    minn = i - j
                window -= nums[j]
                j += 1

        return minn

    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 171
        t = 1
        ans = 0
        for i in range(len(s) - 1, -1, -1):
            v = ord(s[i]) - ord('A') + 1
            ans += v * t
            t *= 26
        return ans

    def trapRainWater(self, heightMap):
        """
        :type heightMap: List[List[int]]
        :rtype: int
        """
        # 407
        pass

    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        tokens = s.strip().split(' ')
        if len(tokens) == 0:
            return 0
        return len(tokens[-1])

    # def lexicalOrder(self, n):
    #     """
    #     :type n: int
    #     :rtype: List[int]
    #     """
    #     # 386
    #     cur = 1
    #     ans = []
    #     for i in range(n):
    #         ans.append(cur)
    #
    #         if cur * 10 <= n:
    #             cur = cur * 10
    #         elif cur % 10 != 9 and cur + 1 <= n:
    #             cur += 1
    #         else:
    #             while (cur // 10) % 10 == 9:
    #                 cur //= 10
    #             cur = cur // 10 + 1
    #     return ans
    def lexicalOrder(self, n):
        ans = []

        def dfs(cur):
            if cur > n:
                return
            ans.append(cur)
            for i in range(10):
                if cur * 10 + i > n:
                    return
                dfs(cur * 10 + i)

        for i in range(1, n + 1):
            dfs(i)
        return ans

    # def findTarget(self, root, k):
    #     """
    #     :type root: TreeNode
    #     :type k: int
    #     :rtype: bool
    #     """
    #     nums = []
    #
    #     def dfs(p: TreeNode):
    #         if not p:
    #             return
    #
    #         dfs(p.left)
    #         nums.append(p.val)
    #         dfs(p.right)
    #     dfs(root)
    #     i = 0
    #     j = len(nums) - 1
    #     while i < j:
    #         v = nums[j] + nums[i]
    #         if v == k:
    #             return True
    #         if v > k:
    #             j -= 1
    #         else:
    #             i += 1
    #     return False

    def findTarget(self, root, k):
        nums = set()

        def dfs(p):
            if not p:
                return False
            if k - p.val in nums:
                return True
            nums.add(p.val)
            return dfs(p.left) or dfs(p.right)

        return dfs(root)

    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        # 89
        ans = [0]
        # ans_s = {0}
        # cur = 0
        # flag = True
        # while flag:
        #     flag = False
        #     mask = 1
        #     for i in range(n):
        #         cur ^= mask
        #         print(cur)
        #         if cur in ans_s:
        #             cur ^= mask
        #         else:
        #             flag = True
        #             ans.append(cur)
        #             ans_s.add(cur)
        #
        #         mask *= 2

        return ans

    # def singleNumber(self, nums):
    #     """
    #     :type nums: List[int]
    #     :rtype: List[int]
    #     """
    #     d = {}
    #     for i in nums:
    #         d[i] = d.get(i, 0) + 1
    #     ans = []
    #     for i, v in d.items():
    #         if v != 2:
    #             ans.append(i)
    #     return ans

    # def singleNumber(self, nums):
    #     """
    #     :type nums: List[int]
    #     :rtype: List[int]
    #     """
    #     diff = 0
    #     for i in nums:
    #         diff ^= i
    #     diff &= -diff
    #
    #     g0 = 0
    #     g1 = 0
    #
    #     for i in nums:
    #         if i & diff:
    #             g0 ^= i
    #         else:
    #             g1 ^= i
    #     return [g0, g1]
    def singleNumber(self, nums):
        if len(nums) == 1:
            return nums[0]
        s = set()
        c = 0
        for i in nums:
            if i in s:
                c ^= i
            else:
                s.add(i)
        return c

    def largestNumber(self, nums):

        nums = list(map(str, nums))

        def cmp(x, y):
            return int(y + x) - int(x + y)

        nums.sort(key=functools.cmp_to_key(cmp))
        s = ''.join(nums)
        if s[0] == '0':
            return '0'
        return s

    # def rightSideView(self, root):
    #     """
    #     :type root: TreeNode
    #     :rtype: List[int]
    #     """
    #     ans = []
    #     if not root:
    #         return []
    #     queue = [root]
    #     while queue:
    #         ans.append(queue[-1].val)
    #         t = []
    #         while queue:
    #             v = queue.pop(0)
    #             if v.left:
    #                 t.append(v.left)
    #             if v.right:
    #                 t.append(v.right)
    #         queue.extend(t)
    #     return ans
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ans = []

        def dfs(p, depth):
            if not p:
                return
            if depth == len(ans):
                ans.append(p.val)
            dfs(p.left, depth + 1)
            dfs(p.right, depth + 1)

        dfs(root, 0)
        return ans

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 55
        # if len(nums) == 1:
        #     return True
        #
        # dp = [False for i in range(len(nums))]
        # ln = len(nums)
        # dp[ln - 1] = True
        # dp[ln - 2] = True if nums[ln - 2] >= 1 else False
        #
        # for i in range(len(nums) - 3, -1, -1):
        #     r = False
        #     for j in range(1, nums[i]+1):
        #         if i + j < ln:
        #             r = r or dp[i + j]
        #     dp[i] = r
        # return dp[0]
        i = 0
        search = 0
        n = len(nums)
        while i < n and i <= search:
            search = max(i + nums[i], search)
            i += 1
        return i == n

    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        pass

    def reverseBits(self, n):
        j = 1 << 31
        i = 1
        while i < j:
            t0 = n & i
            t1 = n & j
            print(bin(n))
            if t0 != t1:
                if t0:
                    n = n | j
                else:
                    n = n & ~j

                if t1:
                    n = n | i
                else:
                    n = n & ~i
            i <<= 1
            j >>= 1
        return n

    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        a = list(map(int, list(a)))
        b = list(map(int, list(b)))

        i = len(a) - 1
        j = len(b) - 1

        c = 0
        ans = ''
        while i >= 0 or j >= 0:
            t = c
            if i >= 0:
                t += a[i]
                i -= 1
            if j >= 0:
                t += b[j]
                j -= 1
            c = t // 2
            ans = str(t % 2) + ans

        if c:
            ans = str(c) + ans
        return ans

    # def findPaths(self, m, n, N, i, j):
    #     """
    #     :type m: int
    #     :type n: int
    #     :type N: int
    #     :type i: int
    #     :type j: int
    #     :rtype: int
    #     """
    #     # 576
    #     # def onBounder(i,j):
    #     #      return i==0 and j
    #     dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    #     queue=[]
    # def findPoisonedDuration(self, timeSeries: list, duration):
    #     """
    #     :type timeSeries: List[int]
    #     :type duration: int
    #     :rtype: int
    #     """
    #     # 495
    #     if not timeSeries:
    #         return 0
    #
    #     ans = 0
    #     i = 0
    #
    #     # tn = timeSeries[0] + duration - 1
    #
    #     # ans += duration
    #
    #     pre = 0
    #     tn = 0
    #
    #     while i < len(timeSeries):
    #         j = i + duration - 1
    #
    #         if j > tn:
    #             ans += tn - pre
    #
    #             pre = i
    #             tn = j
    #
    #         i += 1
    #
    #     return ans
    def quickSort(self, A: list, s, t):
        i = s
        j = t
        if i >= j:
            return
        v = A[i]
        while i < j:
            while i < j and A[j] >= v:
                j -= 1
            A[i] = A[j]
            while i < j and A[i] <= v:
                i += 1
            A[j] = A[i]
        A[i] = v
        self.quickSort(A, s, i - 1)
        self.quickSort(A, i + 1, t)
        return A

    def diameterOfBinaryTree(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.r = 0

        def dfs(p: TreeNode):
            if not p:
                return 0

            if not p.left and not p.right:
                return 1
            left = dfs(p.left)
            right = dfs(p.right)
            self.r = max(self.r, left + right)
            return max(left, right) + 1

        dfs(root)
        return self.r

    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        if word.isupper() or word.islower():
            return True
        if word[0].isupper():
            return word[1:].islower()
        return False

    def isPalindrome_234(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        nums = []
        p = head
        while p:
            nums.append(p.val)
            p = p.next
        i = 0
        j = len(nums) - 1
        while i < j:
            if nums[i] != nums[j]:
                return False
            i += 1
            j -= 1
        return True

    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        # 406

    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # 326
        # if n<=0:
        #     return False
        # while n != 1:
        #     if n % 3 != 0:
        #         return False
        #     n //= 3
        # return True
        #
        return n > 0 and 2 ** 19 % n == 0

    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums2 = nums[:]
        nums2.sort()
        s = 0
        t = len(nums) - 1
        while s <= t:
            if nums[s] == nums2[s]:
                s += 1
            elif nums[t] == nums2[t]:
                t -= 1
        return t - s + 1

    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        # 456
        def isSorted(nums, k):
            lp = -1
            for i in range(0, len(nums)):
                if nums[i] <= k:
                    continue
                if lp == -1:
                    lp = i
                else:
                    if nums[i] < nums[lp]:
                        return False
                    lp = i
            return True

        flag = False
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                flag = True
                break

        if not flag:
            return False

        for i in range(0, len(nums)):
            if not isSorted(nums[i + 1:], nums[i]):
                return True
        return False

    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        pv = nums[0]

        i = 0
        j = len(nums) - 1
        while i <= j:
            mid = (i + j) // 2
            v = nums[mid]
            if v <= pv:
                i = mid + 1
            else:
                j = mid - 1


if __name__ == '__main__':
    # a = MedianFinder()
    # for i in range(10):
    #     a.addNum(random.randint(1, 100))
    # print(a.findMedian())
    sol = Solution()
    # print(sol.numDecodings('0121'))
    # print(sol.numDecodings(
    #     "4757562545844617494555774581341211511296816786586787755257741178599337186486723247528324612117156948"))
    # print(sol.reverseWords('ni hao ma !'))
    # print(sol.minSubArrayLen(8, [2, 3, 1, 2, 4, 3]))
    # print(sol.titleToNumber('AB'))
    # print(sol.lexicalOrder(20))
    # print(sol.lexicalOrder(10))
    # a = MinStack()
    # a.push(-2)
    # a.push(0)
    # a.push(-3)
    # print(a.getMin())
    # a.pop()
    # print(a.top(), a.getMin())
    # root = sol.buildTree([2, 3, 4, 5, 6, 7], [2, 4, 3, 7, 6, 5])
    root = sol.buildTree([4, 2, 5, 1, 7, 3], [4, 5, 2, 7, 3, 1])

    # print(sol.findTarget(root, 9))
    # print(sol.grayCode(3))
    # print(sol.singleNumber([2, 2, 3, 2]))
    # print(sol.largestNumber([3, 30, 34, 5, 9]))
    # print(sol.rightSideView(root))
    # print(sol.canJump([2, 3, 1, 1, 4]))
    # print(sol.canJump([3, 2, 1, 0, 4]))
    # print(sol.canJump([2, 0, 0, ]))
    # r = sol.buildTree([1, 2, 3], [1, 3, 2])
    # r = sol.buildTree([1, 2, 3], [2, 1, 3])
    # print(r)
    # print(sol.reverseBits(1))
    # print(sol.addBinary('1', '1111101'))
    # print(sol.findPoisonedDuration([1, 4], 2))
    # print(sol.findPoisonedDuration([1, 2], 2))
    # '''
    #

    # print(sol.quickSort([5, 2, 4, 5432, 2, 1], 0, 5))
    # w = WordDictionary()
    # w.addWord('abc')
    # w.addWord('bca')
    # w.addWord('abd')
    # print(w.search('abc'))
    # print(w.search('ab.'))
    # print(w.search('ad.'))
    # print(sol.diameterOfBinaryTree(root))
    # print(sol.detectCapitalUse('USA'))
    # print(sol.detectCapitalUse('Flag'))
    # print(sol.detectCapitalUse('FlAg'))
    # print(sol.detectCapitalUse('leetcode'))
    # n = NumArray([1, 2, 4, 6, 7, 3, 8, 34, 25, 3243, 34, 5, 6, 7])
    # print(n.sumRange(2, 7))
    # n.update(3, 9)
    # print((n.sumRange(2, 7)))
    # print(sol.isPowerOfThree(3))
    # print(sol.isPowerOfThree(4))
    # print(sol.isPowerOfThree(243))
    # print(sol.isPowerOfThree(177146))
    # print(sol.findUnsortedSubarray([1, 2, 3, 4]))
    print(sol.find132pattern([1, 2, 3, 4, 4]))
    print(sol.find132pattern([3, 1, 4, 2]))
    print(sol.find132pattern([-1, 3, 2, 0]))
    print(sol.find132pattern([3, 5, 0, 3, 4]))
