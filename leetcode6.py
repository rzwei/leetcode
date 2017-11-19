class TrieNode:
    def __init__(self, w):
        self.w = w
        self.isWord = False
        self.next = {}


class Trie:
    def __init__(self):
        self.root = TrieNode(None)

    def addWord(self, word):
        cur = self.root
        for i in word:
            if i not in cur.next:
                cur.next[i] = TrieNode(i)
            cur = cur.next[i]
        cur.isWord = True

    def isword(self, s):
        cur = self.root
        for i in s:
            if i not in cur.next:
                return False
            else:
                cur = cur.next[i]
        return cur.isWord


class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # 560
        prefixSums = [0 for i in range(len(nums) + 1)]
        sumsPrefix = {}
        sumsPrefix[0] = [0]

        sums = 0
        for i in range(len(nums)):
            sums += nums[i]
            prefixSums[i + 1] = sums
            if sums not in sumsPrefix:
                sumsPrefix[sums] = []
            sumsPrefix[sums].append(i + 1)
        ans = 0
        for i in range(len(prefixSums)):
            target = prefixSums[i] + k
            if target in sumsPrefix:
                for j in sumsPrefix[target]:
                    if j > i:
                        ans += 1
        return ans

    def kSmallestPairs(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        # 373
        # ret = []
        # for i in range(len(nums1)):
        #     for j in range(len(nums2)):
        #         if k <= 0:
        #             return ret
        #         ret.append([nums1[i], nums2[j]])
        #         k -= 1
        # return ret

    def largestDivisibleSubset(self, nums: list):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # nums.sort()
        # sets={}
        # for i in nums:
        pass

    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 540
        # for i in range(0,len(nums),2):
        #     if nums[i]!=nums[i+1]:
        pass

        # def calculate(self, s: str):
        """
        :type s: str
        :rtype: int
        """
        # 224
        # self.si = 0
        # s = list(s)
        #
        # def calc(op1, op, op2):
        #     if op == '+':
        #         return int(op1) + int(op2)
        #     if op == '-':
        #         return int(op1) - int(op2)
        #
        # def guiyue(operators: list):
        #     print(operators)
        #     v = operators[-1]
        #     if v == '+' or v == '-' or v == '(':
        #         return 0
        #     if v == ')':
        #
        #         operators.pop()
        #         temp=[]
        #         while operators[-1]!='(':
        #             temp.append(operators.pop())
        #         operators.pop()
        #
        #         while temp:
        #             operators.append(temp.pop())
        #         # op = operators.pop()
        #         # operators.pop()
        #         # operators.append(op)
        #         return guiyue(operators) or 1
        #     if type(v) is int:
        #         if operators[-2] == '+' or operators[-2] == '-':
        #             op2 = operators.pop()
        #             op = operators.pop()
        #             op1 = operators.pop()
        #             operators.append(calc(op1, op, op2))
        #             return guiyue(operators) or 1
        #         else:
        #             return 0
        #     return 0
        #
        # def getsym():
        #     if self.si >= len(s):
        #         return None
        #     v = s[self.si]
        #     if v.isdigit():
        #         ret = 0
        #         while self.si < len(s) and s[self.si].isdigit():
        #             ret = ret * 10 + int(s[self.si])
        #             self.si += 1
        #         return ret
        #     elif v.isspace():
        #         while self.si < len(s) and s[self.si].isspace():
        #             self.si += 1
        #         return ''
        #     elif (v == '-' and self.si == 0) or (v == '-' and self.si > 0 and s[self.si - 1] == '('):
        #         self.si += 1
        #         ret = 0
        #         while self.si < len(s) and s[self.si].isdigit():
        #             ret = ret * 10 + int(s[self.si])
        #             self.si += 1
        #         return -ret
        #     elif v == '+':
        #         self.si += 1
        #         return '+'
        #     elif v == '-':
        #         self.si += 1
        #         return '-'
        #     elif v == '(':
        #         self.si += 1
        #         return '('
        #     elif v == ')':
        #         self.si += 1
        #         return ')'
        #
        # operators = ['#']
        # v = getsym()
        # while v is not None:
        #     if v is not '':
        #         operators.append(v)
        #         guiyue(operators)
        #     v = getsym()
        # while guiyue(operators):
        #     pass
        # return operators[1]

    def calculate(self, s):
        total = 0
        i, signs = 0, [1, 1]
        while i < len(s):
            c = s[i]
            if c.isdigit():
                start = i
                while i < len(s) and s[i].isdigit():
                    i += 1
                total += signs.pop() * int(s[start:i])
                continue
            if c in '+-(':
                signs += signs[-1] * (1, -1)[c == '-'],
            elif c == ')':
                signs.pop()
            i += 1
        return total

    def __init__(self):
        self.cache = dict()
        self.cache[0] = 1
        self.cache[1] = 0
        self.cache[2] = 1
        self.cache[3] = 2

    def integerReplacement(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 397
        if n in self.cache:
            return self.cache[n]
        if n & 1:
            self.cache[n] = min(self.integerReplacement(n - 1), self.integerReplacement(n + 1)) + 1
            return self.cache[n]
        else:
            return self.integerReplacement(n // 2) + 1

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # 39
        ans = []

        def dfs(path, remaining, start):
            if remaining < 0:
                return
            if remaining == 0:
                ans.append(path[:])
                return
            for i in range(start, len(candidates)):
                path.append(candidates[i])
                dfs(path, remaining - candidates[i], i)
                path.pop()

        dfs([], target, 0)
        return ans

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # 40
        ans = []
        candidates.sort()

        def dfs(path, remaining, start):
            if remaining < 0:
                return
            if remaining == 0:
                ans.append(path[:])
                return
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                dfs(path, remaining - candidates[i], i + 1)
                path.pop()

        dfs([], target, 0)
        return ans

    # def __init__(self):
    #     self.cache = {}

    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # 377
        if not nums:
            return 0
        nums.sort()
        self.ans = 0

        cache = {}

        def dfs(remaining):
            if remaining in cache:
                return cache[remaining]
            if remaining < 0:
                return 0
            if remaining == 0:
                # self.ans += 1
                return 1
            ret = 0
            for i in nums:
                ret += dfs(remaining - i)
            cache[remaining] = ret
            return ret

        return dfs(target)

    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 147
        nums = []
        p = head
        while p:
            nums.append(p.val)
            p = p.next
        nums.sort()
        i = 0
        p = head
        while p:
            p.val = nums[i]
            i += 1
            p = p.next
        return head

    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 647
        self.ans = 0

        def count(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                self.ans += 1
                left -= 1
                right += 1

        for i in range(0, len(s)):
            count(i, i)
            count(i, i + 1)
        return self.ans

    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 645
        ans = [0, 0]
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            if nums[index] < 0:
                ans[0] = i
            else:
                nums[index] = -nums[index]

        for i in range(len(nums)):
            if nums[i] > 0:
                ans[1] = i + 1

        return ans

    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        # 415
        i = len(num1) - 1
        j = len(num2) - 1
        c = 0
        ans = ''
        while i >= 0 or j >= 0:
            a = 0
            if i >= 0:
                a = num1[i]
            b = 0
            if j >= 0:
                b = num2[j]

            v = int(a) + int(b) + c
            c = v // 10

            ans = chr(ord('0') + v % 10) + ans

            i -= 1
            j -= 1
        if c:
            ans = chr(ord('0') + c) + ans
        return ans

    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """

        # 378
        # k -= 1
        def upper_bound(nums, target):
            if target > nums[-1]:
                return len(nums)
            i = 0
            j = len(nums)
            while i < j:
                mid = (i + j) // 2
                if nums[mid] <= target:
                    i = mid + 1
                else:
                    j = mid
            return j

        n = len(matrix)
        le = matrix[0][0]
        ri = matrix[-1][-1]
        while le < ri:
            mid = (le + ri) // 2
            num = 0
            for i in range(n):
                num += upper_bound(matrix[i], mid)
            if num < k:
                le = mid + 1
            else:
                ri = mid
        return le

    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # 347
        d = {}
        for i in nums:
            if i not in d:
                d[i] = 0
            d[i] += 1
        values = [(j, i) for i, j in d.items()]
        values.sort()
        ans = []
        for i in range(len(values) - 1, len(values) - 1 - k, -1):
            ans.append(values[i][1])
        return ans

    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 191
        k = 1
        ans = 0
        for i in range(32):
            ans += n & k
            # n >>= 1
            k <<= 1
            print(k)
        return ans

    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n == 0:
            return False
        if n < 0:
            return False
        while n != 1:
            if n % 2 != 0:
                return False
            n >>= 1
        return True

    def minSteps(self, i):
        """
        :type n: int
        :rtype: int
        """
        # 650
        cache = dict()
        cache[1] = 0
        cache[2] = 2

        def dp(n):
            if n in cache:
                return cache[n]
            ret = n
            for j in range(n - 1, 0, -1):
                if n % j == 0:
                    ret = dp(j) + n // j
                    break
            cache[n] = ret
            return ret

        return dp(i)

    def swapPairs(self, head: ListNode):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 24
        p = head
        while p:
            if p.next:
                p.val, p.next.val = p.next.val, p.val
                p = p.next.next
            else:
                p = p.next
        return head

    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # 46
        ans = []
        length = len(nums)
        nums.sort()
        visited = set()

        def dfs(path):
            print(path)
            if len(path) >= length:
                ans.append(path[:])
                return
            for i in range(len(nums)):
                if i in visited or i > 0 and nums[i] == nums[i - 1] and i - 1 in visited:
                    continue
                visited.add(i)
                path.append(nums[i])
                dfs(path)
                path.pop()
                visited.remove(i)

        dfs([])
        return ans

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # 78
        ans = []
        nums.sort()

        def dfs(path, i):
            # if i >= len(nums):
            #     ans.append(path[:])
            #     return
            ans.append(path[:])
            # dfs(path, i + 1)
            #
            # if i > 0 and nums[i] == nums[i - 1] and nums[i - 1] in path:
            #     return
            # path.append(nums[i])
            # dfs(path, i + 1)
            # path.pop()
            for j in range(i, len(nums)):
                if j > i and nums[j] == nums[j - 1]:
                    continue
                path.append(nums[j])
                dfs(path, j + 1)
                path.pop()

        dfs([], 0)
        return ans

    def partition(self, s: str):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        # 131
        ans = []

        def isPalindrome(s, i, j):
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        def dfs(path, n, s):
            if n >= len(s):
                ans.append(path[:])
                return
            start = n
            end = n
            while end < len(s):
                if isPalindrome(s, start, end):
                    path.append(s[start:end + 1])
                    dfs(path, end + 1, s)
                    path.pop()
                end += 1

        dfs([], 0, s)
        return ans

    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        i = 0
        j = len(nums) - 1
        while i < j:
            mid = (i + j) // 2
            # if nums[mid] == target:
            #     return mid
            if nums[mid] < target:
                i = mid + 1
            else:
                j = mid
        return i

    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # 392

    def removeElements(self, head: ListNode, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        # 203
        thead = ListNode(0)
        p = thead
        p.next = head
        q = head
        while q:
            if q.val == val:
                p.next = q.next
                q = p.next
            else:
                p = p.next
                q = q.next
        return thead.next

    def zigzagLevelOrder(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        # 103
        if not root:
            return []
        ans = []
        q = [root]
        flag = 1
        while q:
            p = []
            for i in q:
                if i.left:
                    p.append(i.left)
                if i.right:
                    p.append(i.right)
            if flag == 1:
                t = []
                while q:
                    v = q.pop(0)
                    t.append(v.val)
                ans.append(t)
                flag = 0
            else:
                t = []
                while q:
                    v = q.pop()
                    t.append(v.val)
                ans.append(t)
                flag = 1
            q = p
        return ans

    def zigzagLevelOrder(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        # 103
        if not root:
            return []
        ans = []
        q = [root]
        while q:
            p = []
            for i in q:
                if i.left:
                    p.append(i.left)
                if i.right:
                    p.append(i.right)
            t = []
            while q:
                v = q.pop(0)
                t.append(v.val)
            ans.append(t)
            q = p
        return ans[::-1]

    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 111
        if not root:
            return 0

        def dfs(p):
            if not p.left and not p.right:
                return 1
            if p.left and p.right:
                return min(dfs(p.left), dfs(p.right)) + 1
            if p.left:
                return dfs(p.left) + 1
            if p.right:
                return dfs(p.right) + 1

        return dfs(root)

    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        # 606
        if not t:
            return ''
        left = ''
        right = ''
        if t.left:
            left = self.tree2str(t.left)
        if t.right:
            right = self.tree2str(t.right)
        if right == '':
            if left == '':
                return str(t.val)
            else:
                return str(t.val) + '(' + left + ')'
        return str(t.val) + '(' + left + ')' + '(' + right + ')'

    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        # 100
        if not p and not q:
            return True
        if p and q:
            if p.val != q.val:
                return False
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False

    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 414
        maxs = []
        for i in range(3):
            max_v = -2147483649
            for j in range(len(nums)):
                if nums[j] not in maxs and nums[j] > max_v:
                    max_v = nums[j]
            if max_v != -2147483648:
                maxs.append(max_v)
        if len(maxs) != 3:
            return maxs[0]
        return maxs[-1]

    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # 64
        m = len(grid)
        n = len(grid[0])
        cache = [[-1 for i in range(n)] for j in range(m)]

        def dp(i, j):
            if i < 0 or i >= m or j < 0 or j >= n:
                return -1
            if cache[i][j] != -1:
                return cache[i][j]
            down = dp(i + 1, j)
            right = dp(i, j + 1)

            if down == -1 and right == -1:
                return grid[i][j]
            ret = 3147483647
            if down != -1:
                ret = min(down, ret)
            if right != -1:
                ret = min(right, ret)
            cache[i][j] = ret + grid[i][j]
            return cache[i][j]

        return dp(0, 0)

    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        # 283
        i = 0
        for j in range(len(nums)):
            if nums[j] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1

    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        # 228
        s = 0
        ans = []
        while s < len(nums):
            t = s + 1
            while t < len(nums):
                if nums[t] - nums[t - 1] == 1:
                    t += 1
                else:
                    t -= 1
                    break
            if t == len(nums):
                t -= 1
            if t == s:
                st = str(nums[s])
            else:
                st = str(nums[s]) + '->' + str(nums[t])
            ans.append(st)
            if s == t:
                s += 1
            else:
                s = t + 1

        return ans

    def eraseOverlapIntervals(self, intervals: list):
        """
        :type intervals: List[Interval]
        :rtype: int
        """
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x.end)
        count = 1
        end = intervals[0].end
        for i in range(1, len(intervals)):
            if intervals[i].start >= end:
                end = intervals[i].end
                count += 1
        return len(intervals) - count

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """

        # 139
        # 140

        class TrieNode:
            def __init__(self, w):
                self.w = w
                self.isWord = False
                self.next = {}

        tree = TrieNode(None)

        for word in wordDict:
            cur = tree
            for i in word:
                if i not in cur.next:
                    cur.next[i] = TrieNode(i)
                cur = cur.next[i]
            cur.isWord = True

        def isWord(s):
            cur = tree
            for i in s:
                if i not in cur.next:
                    return False
                cur = cur.next[i]
            return cur.isWord

        def getTries(p: TrieNode, i, s, paths):
            if i > len(s) or not p:
                return
            if p.isWord:
                paths.append(i)
            if i + 1 < len(s):
                if s[i + 1] in p.next:
                    getTries(p.next[s[i + 1]], i + 1, s, paths)

        def dfs(s):
            if not s:
                return True
            if s[0] in tree.next:
                paths = []
                getTries(tree.next[s[0]], 0, s, paths)
                for t in paths:
                    if dfs(s[t + 1:]):
                        return True
            return False

        return dfs(s)

    # def wordBreak(self, s, wordDict):
    #     """
    #     :type s: str
    #     :type wordDict: List[str]
    #     :rtype: List[str]
    #     """
    #     # 140
    #     max_len = 0
    #     for i in wordDict:
    #         max_len = max(max_len, len(i))
    #
    #     class TrieNode:
    #         def __init__(self, w):
    #             self.w = w
    #             self.isWord = False
    #             self.next = {}
    #
    #     tree = TrieNode(None)
    #
    #     for word in wordDict:
    #         cur = tree
    #         for i in word:
    #             if i not in cur.next:
    #                 cur.next[i] = TrieNode(i)
    #             cur = cur.next[i]
    #         cur.isWord = True
    #
    #     def isWord(s):
    #         cur = tree
    #         for i in s:
    #             if i not in cur.next:
    #                 return False
    #             cur = cur.next[i]
    #         return cur.isWord
    #
    #     ans = []
    #
    #     def dfs(s, string):
    #         if not s:
    #             ans.append(string[1:])
    #             return
    #         for i in range(1, max_len + 1):
    #             if i <= len(s) and isWord(s[:i]):
    #                 print(s[:i])
    #                 dfs(s[i:], string + ' ' + s[:i])
    #
    #     dfs(s, '')
    #     return ans
    def kthSmallest_230(self, root: TreeNode, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        # 230
        self.count = 0
        self.ret = None

        def dfs(p):
            if self.ret:
                return
            if not p:
                return None
            dfs(p.left)

            self.count += 1
            if self.count == k:
                self.ret = p.val
                return
            return dfs(p.right)

        dfs(root)
        return self.ret

    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        # 73
        if not matrix:
            return
        mm = [1 for i in range(len(matrix) + len(matrix[0]))]
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    mm[i] = 0
                    mm[m + j] = 0
        for i in range(m):
            if mm[i] == 0:
                for j in range(n):
                    matrix[i][j] = 0
        for i in range(n):
            if mm[m + i] == 0:
                for j in range(m):
                    matrix[j][i] = 0

    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        # 95
        if n == 0:
            return []

        def dfs(start, end):
            if start > end:
                return [None]
            if start == end:
                return [TreeNode(start)]
            ret = []
            i = start
            while i <= end:
                lefts = dfs(start, i - 1)
                rights = dfs(i + 1, end)
                for l in lefts:
                    for r in rights:
                        root = TreeNode(i)
                        root.left = l
                        root.right = r
                        ret.append(root)
                i += 1
            return ret

        return dfs(1, n)

    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 152
        r = nums[0]
        rmax = r
        rmin = r

        for i in range(1, len(nums)):
            if nums[i] < 0:
                rmax, rmin = rmin, rmax
            rmax = max(nums[i], rmax * nums[i])
            rmin = min(nums[i], rmin * nums[i])
            r = max(r, rmax)
        return r

    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 238
        res = [0 for i in nums]
        res[0] = 1
        for i in range(1, len(nums)):
            res[i] = res[i - 1] * nums[i - 1]
        right = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= right
            right *= nums[i]
        return res

    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # 205
        d1 = {}
        d2 = {}
        for i in range(len(s)):
            if d1.get(s[i], 0) != d2.get(t[i], 0):
                return False
            d1[s[i]] = i + 1
            d2[t[i]] = i + 1
        return True

        # d1 = {}
        # for i in s:
        #     d1[i] = d1.get(i, 0) + 1
        # v1 = list(d1.items())
        # v1.sort(key=lambda x: x[1])
        # for i, v in enumerate(v1):
        #     d1[v[0]] = str(i)
        #
        # d2 = {}
        # for i in t:
        #     d2[i] = d2.get(i, 0) + 1
        # v2 = list(d2.items())
        # v2.sort(key=lambda x: x[1])
        # for i, v in enumerate(v2):
        #     d2[v[0]] = str(i)
        # for i in range(len(s)):
        #     if d1[s[i]] != d2[t[i]]:
        #         return False
        # return True

    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0:
            return 0
        i = 0
        j = x
        ret = 0
        while i <= j:
            # mid = i + (j - i) >> 1
            mid = (i + j) >> 1
            if mid * mid <= x:
                i = mid + 1
                ret = mid
            else:
                j = mid - 1
        return ret

    def wordPattern(self, pattern, s: str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        # 290
        tokens = s.split(' ')
        if len(tokens) != len(pattern):
            return False
        d1 = {}
        d2 = {}
        for i in range(len(pattern)):
            if d1.get(pattern[i], 0) != d2.get(tokens[i], 0):
                return False
            d1[pattern[i]] = i + 1
            d2[tokens[i]] = i + 1
        return True


def buildTree(nodes: list):
    s = [TreeNode(nodes.pop(0))]
    head = s[0]
    while nodes:
        r = s.pop()
        left = nodes.pop(0)
        if left:
            r.left = TreeNode(left)

        right = nodes.pop(0)
        if right:
            r.right = TreeNode(right)

        if r.right:
            s.append(r.right)

        if r.left:
            s.append(r.left)
    return head


def buildLinkList(nums: list):
    head = ListNode(0)
    p = head
    for i in nums:
        p.next = ListNode(i)
        p = p.next
    return head.next


def binarySearch(nums, target):
    nums.sort()
    print(nums)
    i = 0
    j = len(nums) - 1
    while i < j:
        mid = (i + j) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            i = mid + 1
        else:
            j = mid
    return i


def upper_bound(nums, target):
    i = 0
    j = len(nums) - 1
    while i < j:
        mid = i + (j - i) // 2
        if nums[mid] <= target:
            i = mid + 1
        else:
            j = mid
    return j


if __name__ == '__main__':
    sol = Solution()

    # print(sol.subarraySum([1, 1], 1))
    # print(sol.subarraySum([-1, -1, 1], 0))
    # print(sol.kSmallestPairs([1, 7, 11], [2, 4, 6], 3))
    # print(sol.kSmallestPairs([1, 1, 2], [1, 2, 3], 2))
    # print(sol.singleNonDuplicate([1, 1, 2, 3, 3, 4, 4, 8, 8]))
    # print(sol.calculate('1+2+8-8-(-2+8+1)'))
    # print(sol.calculate('0'))
    # print(sol.calculate('   30'))
    # print(sol.calculate('(1-(3-4))'))
    # print(sol.calculate("(7)-(0)+(4)"))
    # print(sol.integerReplacement(8))
    # print(sol.combinationSum4([1, 2, 3], 4))
    # print(sol.combinationSum4([1, 2, 3], 4))
    # print(sol.combinationSum([2, 3, 6, 7], 7))
    # print(sol.combinationSum2([10, 1, 2, 7, 6, 1, 5], 8))
    # head = buildLinkList([2, 4, 5, 1, 2, 5])
    # head = buildLinkList([3, 2])
    # head = sol.insertionSortList(head)
    # while head:
    #     print(head.val)
    #     head = head.next
    # print(sol.countSubstrings('aaa'))
    # print(sol.findErrorNums([1, 2, 3, 3]))
    # print(sol.findErrorNums([1, 2, 2, 4]))
    # print(sol.findErrorNums([2, 3, 2]))
    # print(sol.addStrings('123123122', '1231231231'))
    # print(sol.addStrings('98', '9'))
    matrix = [
        [1, 5, 9],
        [10, 11, 13],
        [12, 13, 15]
    ]
    print(sol.kthSmallest(matrix, 1))
    print(sol.kthSmallest(matrix, 2))
    print(sol.kthSmallest(matrix, 3))
    print(sol.kthSmallest(matrix, 4))
    print(sol.kthSmallest(matrix, 5))
    print(sol.kthSmallest(matrix, 6))
    print(sol.kthSmallest(matrix, 7))
    print(sol.kthSmallest(matrix, 8))
    print(sol.kthSmallest(matrix, 9))
    # print(binarySearch([1, 3, 4, 4, 7, 32, 56], 2))
    # print(upper_bound([12, 13, 15], 16))
    # print(sol.topKFrequent([1, 1, 1, 2, 2, 3], 2))
    # print(sol.hammingWeight(11))
    # print(sol.isPowerOfTwo(-1 << 5))

    # 1 2 3
    # print(sol.minSteps(5))
    # print(sol.permute([1, 1, 3]))
    # print(sol.permute([1, 3, 3]))
    # print(sol.subsets([1, 2, 3]))
    # print(sol.subsets([1, 2, 2]))
    # print(sol.partition('aab'))
    # print(sol.searchInsert([1, 3, 5, 6], 1))
    # print(sol.searchInsert([1, 3, 5], 2))
    # root = buildTree([3, 9, 20, None, None, 15, 7])
    # print(sol.zigzagLevelOrder(root))
    # root = TreeNode(0)
    # root.left = TreeNode(1)
    # print(sol.minDepth(root))
    # print(sol.tree2str(root))
    # print(sol.thirdMax([3, 2, 2, 1]))
    # print(sol.thirdMax([2, 1]))
    # nums = [0, 1, 0, 3, 12]
    # nums = [0, 1, 0, 3, 12]
    # sol.moveZeroes(nums)
    # print(nums)
    #
    # prims = []
    #
    #
    # def isPrime(n):
    #     v = math.sqrt(n)
    #     for i in prims:
    #         if i > v:
    #             break
    #         if n % i == 0:
    #             return False
    #     prims.append(n)
    #     return True
    #
    #
    # sums = 0
    # n = 2
    # count = 1
    # while n < 1e8:
    #     if isPrime(n):
    #         count += 1
    #         if count % 2 == 0:
    #             print(n)
    #             sums += n
    #         else:
    #             print(-n)
    #             sums -= n
    #     n += 1
    # print(sums)
    # print(sol.summaryRanges([0, 1, 2, 4, 5, 7]))
    # print(sol.eraseOverlapIntervals([[1, 2], [2, 3], [3, 4], [1, 3]]))
    # print(sol.eraseOverlapIntervals([[1, 2], [1, 2], [1, 2]]))
    # print(sol.eraseOverlapIntervals([[1, 2], [2, 3]]))
    # def buildInterval(nums):
    #     ret = []
    #     for i in nums:
    #         ret.append(Interval(i[0], i[1]))
    #     return ret


    # nums = buildInterval([[1, 100], [11, 22], [1, 11], [2, 12]])
    # print(sol.eraseOverlapIntervals(nums))
    # print(sol.wordBreak('catsanddog', ["cat", "cats", "and", "sand", "dog"]))
    # print(sol.wordBreak('leetcodea', ['leet', 'code']))
    # print(sol.wordBreak(
    #     "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    #     "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    #     ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaa", "aaaaaaaaaa"]))
    # tree = buildTree(
    #     [31, 30, 48, 3, None, 38, 49, 0, 16, 35, 47, None, None, None, 2, 15, 27, 33, 37, 39, None, 1, None, 5, None,
    #      22, 28, 32, 34, 36, None, None, 43, None, None, 4, 11, 19, 23, None, 29, None, None, None, None, None, None,
    #      40, 46, None, None, 7, 14, 17, 21, None, 26, None, None, None, 41, 44, None, 6, 10, 13, None, None, 18, 20,
    #      None, 25, None, None, 42, None, 45, None, None, 8, None, 12, None, None, None, None, None, 24, None, None,
    #      None, None, None, None, 9])
    # print(sol.kthSmallest(tree, 1))
    # v = [[1, 2, 3], [0, 4, 5], [6, 0, 8]]
    # sol.setZeroes(v)
    # print(v)
    # trees = sol.generateTrees(1)
    # print(trees)
    # print(sol.productExceptSelf([1, 2, 3, 4]))
    # print(sol.isIsomorphic('paper', 'title'))
    # print(sol.isIsomorphic('aba', 'baa'))
    # print(sol.isIsomorphic('egg', 'baa'))
    # print(sol.wordPattern('abba', 'dog cat cat dog'))
    # print(sol.wordPattern('abba', 'dog dog dog dog'))
    # print(sol.wordPattern('jquery', 'jquery'))
    # print(sol.wordPattern("abcdefghijklmnnmlkjihgfedcba",
    #                       "aa bb cc dd ee ff gg hh ii jj kk ll mm nn nn mm ll kk jj ii hh gg ff ee dd cc bb aa"))
    # print(sol.wordPattern('aa','ww aa'))
    # print(sol.wordPattern("abba",
    #                       "dog cat cat dog"))
