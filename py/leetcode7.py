import collections
import heapq
import queue


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        sums = sum(nums)
        if sums & 1:
            return False
        cache = {}
        length = len(nums)

        def find(target, start):
            if target in cache:
                return cache[target]
            cache[target] = False
            if target < 0:
                return False
            if target == 0:
                cache[target] = True
                return True
            for i in range(start, length):
                if find(target - nums[i], i + 1):
                    cache[target] = True
                    return True
            return False

        return find(sums >> 1, 0)

    def canPartition_dp(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 416
        sums = sum(nums)
        if sums & 1 != 0:
            return False
        sums = sums >> 1
        n = len(nums)
        dp = [[False for i in range(sums + 1)] for j in range(n + 1)]
        dp[0][0] = True
        for i in range(1, sums + 1):
            dp[0][i] = False
        for i in range(1, n + 1):
            dp[i][0] = True
        for i in range(1, n + 1):
            for j in range(1, sums + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= nums[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j - nums[i - 1]]
        return dp[n][sums]

    def findTarget(self, nums, target):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 416
        n = len(nums)
        dp = [[False for i in range(target + 1)] for j in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, target + 1):
            dp[0][i] = -1
        for i in range(1, n + 1):
            dp[i][0] = 0
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= nums[i - 1] and dp[i - 1][j - nums[i - 1]] != -1:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - nums[i - 1]]) + 1
        return dp[n][target]

    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        sums = sum(nums)
        if abs(S) > sums:
            return 0
        sums += S
        if sums & 1:
            return 0
        target = sums >> 1

        def dfs(start, target):
            if start == len(nums) and target == 0:
                return 1
            ret = 0
            for i in range(start, len(nums)):
                ret += dfs(i + 1, target - nums[i])
            return ret

        cache = {}

        def dfs2(i, s):
            key = (i, s)
            if key in cache:
                return cache[key]

            if i == len(nums):
                return 1 if s == S else 0
            ret = 0
            ret += dfs2(i + 1, s - nums[i])
            ret += dfs2(i + 1, s + nums[i])
            cache[key] = ret
            return ret

        return dfs2(0, 0)

    def findTargetSumWays_dp(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        sums = sum(nums)
        if abs(S) > sums:
            return 0

        if (sums + S) & 1:
            return 0

        def findTarget(nums, target):
            dp = [0] * (target + 1)
            dp[0] = 1
            for n in nums:
                for i in range(target, n - 1, -1):
                    dp[i] += dp[i - n]
            return dp[target]

        return findTarget(nums, (sums + S) >> 1)

    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        # 115
        # dp = [[0] * len(s)] * len(t)
        cache = {}
        if s[0] == t[0]:
            cache[(0, 0)] = 1
        else:
            cache[(0, 0)] = 0

        def dp(i, j):
            if i < 0 or j < 0:
                return 0
            key = (i, j)
            if key in cache:
                return cache[key]
            ret = 0
            if s[i] == t[j]:
                ret += dp(i - 1, j - 1)
            ret += dp(i - 1, j)
            ret += dp(i, j - 1)
            cache[key] = ret
            return ret

        return dp(len(s) - 1, len(t) - 1)

    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # 21
        head = ListNode(0)
        p = l1
        q = l2
        th = head
        while p or q:
            if p and q:
                if p.val < q.val:
                    t = p
                    p = p.next
                else:
                    t = q
                    q = q.next
            elif p:
                t = p
                p = p.next
            else:
                t = q
                q = q.next
            th.next = t
            th = t
        return head.next

    def kSmallestPairs(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        # 373
        if k > len(nums1) * len(nums2):
            k = len(nums1) * len(nums2)
        if not nums1 or not nums2:
            return []
        heap = []
        if nums2:
            for i in range(min(k, len(nums1))):
                heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))
        ret = []
        while len(ret) < k and heap:
            _, i, j = heapq.heappop(heap)
            ret.append([nums1[i], nums2[j]])
            j += 1
            if j < len(nums2):
                heapq.heappush(heap, (nums1[i] + nums2[j], i, j))
        return ret

    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 70
        if n == 1 or n == 2:
            return n

        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def pathSum(self, root: TreeNode, target):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """

        # 437
        def hasSum(p, v):
            if not p:
                return 0
            r = 0
            if v == p.val:
                r = 1
            r += hasSum(p.left, v - p.val)
            r += hasSum(p.right, v - p.val)
            return r

        self.ans = 0

        def preOrder(p):
            if not p:
                return
            self.ans += hasSum(p, target)
            preOrder(p.left)
            preOrder(p.right)

        preOrder(root)
        return self.ans

    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 671
        minimum = [-1]

        def travel(p):
            if not p:
                return
            if minimum[0] == -1 or p.val < minimum[0]:
                minimum[0] = p.val
            travel(p.left)
            travel(p.right)

        minimum += [-1]

        def travel2(p):
            if not p:
                return
            if minimum[1] == -1 and p.val != minimum[0] or p.val < minimum[1] and p.val != minimum[0]:
                minimum[1] = p.val
            travel2(p.left)
            travel2(p.right)

        travel(root)
        travel2(root)
        return minimum[1]

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        # 98
        self.val = -0x80000001

        def travel(p):
            if not p:
                return True
            if not travel(p.left):
                return False
            if self.val >= p.val:
                return False
            self.val = p.val
            return travel(p.right)

        return travel(root)

    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        # 23
        head = ListNode(0)
        p = head
        while True:
            min_i = -1
            for i in range(len(lists)):
                if lists[i] and (min_i == -1 or lists[i].val < lists[min_i].val):
                    min_i = i
            if min_i == -1:
                break
            p.next = lists[min_i]
            p = p.next
            lists[min_i] = lists[min_i].next
        return head.next

    # def mergeKLists(self, lists):
    #     """
    #     :type lists: List[ListNode]
    #     :rtype: ListNode
    #     """
    #     # 23
    #     nums = []
    #     for ilist in lists:
    #         p = ilist
    #         while p:
    #             nums.append(p.val)
    #             p = p.next
    #     nums.sort()
    #     head = ListNode(0)
    #     p = head
    #     for i in nums:
    #         p.next = ListNode(i)
    #         p = p.next
    #     return head.next

    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        # 23
        nums = queue.PriorityQueue()
        for ilist in lists:
            if ilist:
                nums.put((ilist.val, ilist))
        dummy = ListNode(0)
        tail = dummy
        while nums.qsize() != 0:
            _, ilist = nums.get()
            tail.next = ilist
            tail = tail.next
            if ilist.next:
                ilist = ilist.next
                nums.put((ilist.val, ilist))
        return dummy.next

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 80

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # 322
        cache = dict()
        cache[0] = 0
        for i in coins:
            cache[i] = 1

        def dp(n):
            if n < 0:
                return -1
            if n in cache:
                return cache[n]
            ret = -1
            for i in coins:
                r = dp(n - i)
                if r != -1:
                    if ret == -1:
                        ret = r
                    else:
                        ret = min(ret, r)
            if ret != -1:
                ret += 1
            cache[n] = ret
            return ret

        return dp(amount)

    def sumOfLeftLeaves(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 404
        if not root:
            return 0
        ret = 0
        if root.left:
            left = root.left
            if not left.left and not left.right:
                ret += left.val
            else:
                ret += self.sumOfLeftLeaves(root.left)
        ret += self.sumOfLeftLeaves(root.right)
        return ret

    def wordBreak_dfs(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        # 139
        wordDict = set(wordDict)
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        for i in range(1, len(s) + 1):
            for j in reversed(range(i)):
                if dp[j]:
                    ts = s[j:i]
                    if ts in wordDict:
                        dp[i] = 1
                        break
        return dp[len(s)]

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        # 139
        wordDict = set(wordDict)
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        for i in range(1, len(s) + 1):
            for j in reversed(range(i)):
                if dp[j]:
                    ts = s[j:i]
                    if ts in wordDict:
                        dp[i] = 1
                        break
        return dp[len(s)]

    def wordBreak_2(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """

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

        def getTries(p, i, s, paths):
            if i > len(s) or not p:
                return
            if p.isWord:
                paths.append(s[:i + 1])
            if i + 1 < len(s):
                if s[i + 1] in p.next:
                    getTries(p.next[s[i + 1]], i + 1, s, paths)

        ans = []

        def dfs(s, string):
            if not s:
                ans.append(string[1:])
                return
            if s[0] in tree.next:
                paths = []
                getTries(tree.next[s[0]], 0, s, paths)
                for t in paths:
                    dfs(s[len(t):], string + ' ' + t)

        dfs(s, '')
        return ans

    def wordBreak_2_(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        # 140 hard
        wordDict = set(wordDict)
        wordsLength = set()
        cache = {}
        for i in wordDict:
            wordsLength.add(len(i))
            # cache[i] = [i]

        def dfs(s):
            if s in cache:
                return cache[s]
            ret = []
            for i in wordsLength:
                if i <= len(s) and s[:i] in wordDict:
                    temp = dfs(s[i:])
                    if temp:
                        for ts in temp:
                            ret.append(s[:i] + ' ' + ts)
                    elif i == len(s):
                        ret.append(s[:i])
            cache[s] = ret
            return ret

        return dfs(s)

    def totalHammingDistance(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 477
        ans = 0
        for i in range(33):
            bitCount = 0
            for j in nums:
                bitCount += (j >> i) & 1
            ans += bitCount * (len(nums) - bitCount)
        return ans

    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        # 77
        nums = [i for i in range(1, n + 1)]

        ans = []

        flag = False
        if k > len(nums) // 2:
            flag = True
            k = n - k

        def dfs(n, path):
            if len(path) == k:
                if flag:
                    t = []
                    for i in nums:
                        if i not in path:
                            t.append(i)
                    ans.append(t)
                else:
                    ans.append(list(path))
                return
            for i in range(n, len(nums)):
                path.append(nums[i])
                dfs(i + 1, path)
                path.pop()

        dfs(0, [])
        return ans

    def findKthNumber(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        # 440

    def __init__(self):
        self.count = 0

    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        # 212
        self.count += 1
        if self.count == 21:
            return ['aa']

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        m = len(board)

        if m == 1:
            ans = []
            for word in words:
                for j in board:
                    if word in j and word not in ans:
                        ans.append(word)
                        break
            return ans

        n = len(board[0])

        startDict = {}

        for i in range(m):
            for j in range(n):
                if board[i][j] not in startDict:
                    startDict[board[i][j]] = []
                startDict[board[i][j]].append((i, j))

        cache = set()

        def dfs(i, j, word, k):
            if (i, j) in cache:
                return False
            if i < 0 or i >= m or j < 0 or j >= n:
                return False

            if k >= len(word):
                return True

            if board[i][j] != word[k]:
                return False

            cache.add((i, j))
            for d in directions:
                x = i + d[0]
                y = j + d[1]
                if (x, y) not in cache and dfs(x, y, word, k + 1):
                    cache.remove((i, j))

                    return True
            cache.remove((i, j))
            return False

        ans = []
        for word in words:
            # cache.clear()
            for i, j in startDict.get(word[0], []):
                if dfs(i, j, word, 0) and word not in ans:
                    ans.append(word)
        return ans

    def findLongestChain(self, pairs: list):
        """
        :type pairs: List[List[int]]
        :rtype: int
        """
        # 646
        if not pairs:
            return 0
        pairs.sort(key=lambda x: x[1])
        last = pairs[0][0] - 1
        ans = 0
        for p in pairs:
            if p[0] > last:
                ans += 1
                last = p[1]
        return ans

    def lowestCommonAncestor_bst(self, root: TreeNode, p: TreeNode, q: TreeNode):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # 235
        while root:
            if p.val < root.val > q.val:
                root = root.left
            elif p.val > root.val < q.val:
                root = root.right
            else:
                return root

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # 236
        # inorder successor in bst   path sum iv redundant connection
        if not root:
            return None
        if root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root
        return left or right

    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """

        # 680

        # def dfs(s, i, j, flag):
        #     if i >= j:
        #         return True
        #     if s[i] == s[j]:
        #         return dfs(s, i + 1, j - 1, flag)
        #     if s[i] != s[j]:
        #         return dfs(s, i, j - 1, False) or dfs(s, i + 1, j, False)
        #
        # return dfs(s, 0, len(s) - 1, True)
        def isValid(s):
            return s == s[::-1]

        i, j = 0, len(s) - 1
        while i < j:
            if s[i] != s[j]:
                if s[i] == s[j - 1] or s[i + 1] == s[j]:
                    return isValid(s[:j] + s[j + 1:]) or isValid(s[:i] + s[i + 1:])
                else:
                    return False
            i += 1
            j -= 1
        return True

    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 530
        nums = []

        def inorder(p):
            if not p:
                return
            if p.left:
                inorder(p.left)
            nums.append(p.val)
            if p.right:
                inorder(p.right)

        inorder(root)
        # nums.sort()

        ans = nums[-1] - nums[0]
        for i in range(1, len(nums)):
            if ans > nums[i] - nums[i - 1]:
                ans = nums[i] - nums[i - 1]
        return ans

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 125
        i = 0
        j = len(s) - 1

        while i < j:
            while i < j and not (s[i].isalpha() or s[i].isdigit()):
                i += 1
            while i < j and not (s[j].isalpha() or s[j].isdigit()):
                j -= 1
            if i == j:
                return True
            if i > j or (s[i].isalpha() and s[i].lower() != s[j].lower()) or (s[i].isdigit() and s[i] != s[j]):
                return False
            i += 1
            j -= 1
        return True

    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        # 450

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """

        # 240
        def lower_bound(nums, target):
            i = 0
            j = len(nums) - 1
            while i <= j:
                mid = (i + j) // 2
                if nums[mid] <= target:
                    i = mid
                else:
                    j = mid - 1
            return i

    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        # 63

    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # using bfs
        # 45
        n = len(nums)
        if n < 2:
            return 0
        i = 0
        level = 0
        curMax = 0
        nextMat = 0
        while curMax - i + 1 >= 0:
            level += 1
            while i <= curMax:
                nextMat = max(nextMat, nums[i] + i)
                if nextMat >= n - 1:
                    return level
                i += 1
            curMax = nextMat
        return 0

    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        # 72
        dp = [[0] * (len(word2) + 1) for j in range(len(word1) + 1)]
        for i in range(1, len(word1) + 1):
            dp[i][0] = i
        for i in range(1, len(word2) + 1):
            dp[0][i] = i
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return dp[-1][-1]

    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 128
        # if not nums:
        #     return 0
        # index = []
        # d = set()
        # for i in nums:
        #     if i not in d:
        #         heapq.heappush(index, i)
        #         d.add(i)
        # ret = 1
        # curi = heapq.heappop(index)
        # cur = 1
        # while index:
        #     t = heapq.heappop(index)
        #     if t - curi == 1:
        #         cur += 1
        #         curi = t
        #         if cur > ret:
        #             ret = cur
        #     else:
        #         curi = t
        #         cur = 1
        # return ret
        nums = set(nums)
        ret = 0
        for i in nums:
            if i - 1 not in nums:
                cur = 1
                j = i + 1
                while j in nums:
                    cur += 1
                    j += 1
                if cur > ret:
                    ret = cur
        return ret

    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        # 114
        # q = []
        #
        # def preorder(p):
        #     if not p:
        #         return
        #     q.append(p)
        #     preorder(p.left)
        #     preorder(p.right)
        #
        # preorder(root)
        # if not q:
        #     return root
        # q[0].left = None
        # for i in range(1, len(q)):
        #     q[i - 1].right = q[i]
        #     q[i].left = None
        # q[-1].right = None
        # return root
        if not root:
            return
        left = root.left
        right = root.right
        root.left = None
        self.flatten(left)
        self.flatten(right)
        root.right = left
        cur = root
        while cur.right:
            cur = cur.right
        cur.right = right

    def findNumberOfLIS(self, nums: list):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 673
        if not nums:
            return 0
        max_v = 1
        ans = 1
        # nums.insert(0, -2147483649)
        dp = [1] * len(nums)
        dp2 = [1] * len(nums)
        for i in range(1, len(nums)):

            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[i] == dp[j] + 1:
                        dp2[i] += dp2[j]
                    elif dp[i] < dp[j] + 1:
                        dp[i] = dp[j] + 1
                        dp2[i] = dp2[j]
            if max_v == dp[i]:
                ans += dp2[i]
            elif max_v < dp[i]:
                max_v = dp[i]
                ans = dp2[i]
        return ans

    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # 16
        nums.sort()
        ans = nums[0] + nums[1] + nums[2]

        for i in range(len(nums) - 2):
            j = i + 1
            k = len(nums) - 1
            # print(i, j, k)
            while j < k:
                s = nums[i] + nums[j] + nums[k]
                if s == target:
                    return target
                if abs(target - s) < abs(target - ans):
                    ans = s
                if s > target:
                    k -= 1
                else:
                    j += 1
        return ans

    def sortList(self, head: ListNode):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return head
        pre = None
        slow = head
        fast = head

        p = head
        while p:
            print(p.val, end=',')
            p = p.next
        print()
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
        pre.next = None
        l1 = self.sortList(head)
        l2 = self.sortList(slow)
        dump = ListNode(0)
        p = dump
        while l1 or l2:
            if l1 and l2:
                if l1.val < l2.val:
                    p.next = l1
                    p = p.next
                    l1 = l1.next
                else:
                    p.next = l2
                    l2 = l2.next
                    p = p.next
            elif l1:
                p.next = l1
                l1 = l1.next
                p = p.next
            else:
                p.next = l2
                l2 = l2.next
                p = p.next
        return dump.next

    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        # num_0 = 0
        # num_1 = 0
        # num_2 = 0
        # for i in nums:
        #     if i == 0:
        #         num_0 += 1
        #     elif i == 1:
        #         num_1 += 1
        #     else:
        #         num_2 += 1
        # for i in range(num_0):
        #     nums[i] = 0
        # for i in range(num_0, num_0 + num_1):
        #     nums[i] = 1
        # for i in range(num_0 + num_1, len(nums)):
        #     nums[i] = 2
        n0 = -1
        n1 = -1
        n2 = -1
        for i in nums:
            if i == 0:
                n2 += 1
                nums[n2] = 2
                n1 += 1
                nums[n1] = 1
                n0 += 1
                nums[n0] = 0
            if i == 1:
                n2 += 1
                nums[n2] = 2
                n1 += 1
                nums[n1] = 1
            if i == 2:
                n2 += 1
                nums[n2] = 2

    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 300
        # if not nums:
        #     return 0
        # dp = [1] * len(nums)
        # ret = 1
        # for i in range(1, len(nums)):
        #     for j in range(i):
        #         if nums[i] > nums[j]:
        #             dp[i] = max(dp[i], dp[j] + 1)
        #             if dp[i] > ret:
        #                 ret = dp[i]
        # return ret
        # tails = [0] * len(nums)
        # size = 0
        # for x in nums:
        #     i, j = 0, size
        #     while i != j:
        #         m = (i + j) / 2
        #         if tails[m] < x:
        #             i = m + 1
        #         else:
        #             j = m
        #     tails[i] = x
        #     size = max(i + 1, size)
        # return size
        tails = [0] * len(nums)
        size = 0
        for n in nums:
            i = 0
            j = size
            while i < j:
                m = (i + j) // 2
                if tails[m] < n:
                    i = m + 1
                else:
                    j = m
            tails[i] = n
            if i == size:
                size += 1
        return size

    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        min_j = i + 1
        if i != -1:
            for j in range(i + 1, len(nums)):
                if nums[min_j] >= nums[j] > nums[i]:
                    min_j = j
            nums[i], nums[min_j] = nums[min_j], nums[i]
        ti = i + 1
        tj = len(nums) - 1
        while ti < tj:
            nums[ti], nums[tj] = nums[tj], nums[ti]
            ti += 1
            tj -= 1

    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        # 474
        ret = 0
        for s in strs:
            if len(s) > m + n:
                continue
            num = 0
            for j in s:
                if j == '0':
                    num += 1
            if num <= m and len(s) - num <= n:
                ret = max(ret, len(s))
        return ret

    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 459
        if not s:
            return False

        ss = (s + s)[1:-1]
        return ss.find(s) != -1

    def __init__(self):
        self.operators = ('+', '-', '*')

    def diffWaysToCompute(self, string):
        """
        :type input: str
        :rtype: List[int]
        """
        # 241
        ret = []
        for i, v in enumerate(string):
            if v in ('+', '-', "*"):
                part1 = string[:i]
                part2 = string[i + 1:]
                ret_1 = self.diffWaysToCompute(part1)
                ret_2 = self.diffWaysToCompute(part2)
                for p1 in ret_1:
                    for p2 in ret_2:
                        ans = 0
                        if v == '+':
                            ans = p1 + p2
                        elif v == '-':
                            ans = p1 - p2
                        elif v == '*':
                            ans = p1 * p2
                        ret.append(ans)
        if not ret:
            ret.append(int(string))
        return ret

    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 312

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        # 84

    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        # 143

    def change(self, amount: int, coins: [int]):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        # 518

    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # 42


def buildTree(nums: list):
    head = TreeNode(nums.pop(0))
    queue = [head]
    while nums:
        v = queue.pop(0)
        l = nums.pop(0)
        r = nums.pop(0)
        if l:
            v.left = TreeNode(l)
            queue.append(v.left)
        if r:
            v.right = TreeNode(r)
            queue.append(v.right)
    return head


def travel(p):
    if not p:
        print('null', end=',')
        return
    print(p.val, end=',')
    travel(p.left)
    travel(p.right)


class Codec:
    def serialize(self, root):
        """
        Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """

        def preorder(node, vals):
            if node:
                vals.append(node.val)
                preorder(node.left, vals)
                preorder(node.right, vals)

        vals = []
        preorder(root, vals)
        return ' '.join(map(str, vals))

    def deserialize(self, data):
        """
        Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        """

        def helper(minVal, maxVal, vals):
            if not vals:
                return None
            if minVal < vals[0] < maxVal:
                val = vals.popleft()
                node = TreeNode(val)
                node.left = helper(minVal, val, vals)
                node.right = helper(val, maxVal, vals)
                return node
            else:
                return None

        vals = collections.deque([int(val) for val in data.split()])
        return helper(float('-inf'), float('inf'), vals)


class Codec:
    # 449
    def serialize(self, root):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """
        ret = []

        def preorder(p):
            if p:
                ret.append(p.val)
                preorder(p.left)
                preorder(p.right)
            else:
                ret.append(None)

        preorder(root)
        return ','.join(map(str, ret))

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        """
        tokens = data.split(',')

        def build(tokens):
            if not tokens:
                return None
            v = tokens.pop(0)
            if v == 'None':
                return None
            r = TreeNode(int(v))
            r.left = build(tokens)
            r.right = build(tokens)
            return r

        return build(tokens)


def buildLinkList(nums: list):
    head = ListNode(0)
    p = head
    for i in nums:
        p.next = ListNode(i)
        p = p.next
    return head.next


if __name__ == '__main__':
    sol = Solution()
    # print(sol.canPartition([1, 5, 11, 5]))
    # print(sol.findTargetSumWays([1, 1, 1, 1, 1], 3))
    # print(sol.findTargetSumWays([1, 0, 0, 0], 1))
    # print(sol.findTargetSumWays([1, 0, 0, 0, 0], 1))
    # print(sol.findTargetSumWays([1000], -1000))
    # print(sol.findTargetSumWays([0, 0, 0, 0, 0, 0, 0, 0, 1], 1))
    # print(sol.numDistinct('abc', 'bc'))
    # print(sol.kSmallestPairs([1, 1, 3], [2, 3, 4], 9))
    # print(sol.kSmallestPairs([1, 7, 11], [2, 4, 6], 9))
    # print(sol.kSmallestPairs([1, 1, 2], [1, 2, 3], 9))
    # print(sol.kSmallestPairs([1, 2], [3], 2))
    # tree = [1, 2, 3, None, None, 4, None]
    # head = buildTree(tree)
    # codec = Codec()
    # r = codec.deserialize(codec.serialize(head))
    # travel(r)
    # head2 = buildTree([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1])
    # travel(head2)
    # print(sol.pathSum(head2, 8))
    # print(sol.coinChange([1, 2, 7], 11))
    # print(sol.coinChange([2, ], 3))
    # print(sol.coinChange([1], 0))
    # print(sol.sumOfLeftLeaves(head2))
    # print(sol.wordBreak_2_('leetcode', ['leet', 'code']))
    # print(sol.wordBreak_2_(
    #     "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    #     , ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaa", "aaaaaaaaaa"]))
    #
    # print(sol.wordBreak_2_('catsanddog', ["cat", "cats", "and", "sand", "dog"]))
    # print(sol.wordBreak_2_('abcd', ['a', 'abc', 'b', 'cd']))
    # print(sol.wordBreak_2_("aaaaaaa", ["aaaa", "aa", "a"]))
    # print(sol.totalHammingDistance([4, 14, 2]))
    # a = time.time()
    # r = sol.combine(20, 4)
    # print(time.time() - a)
    # a = time.time()
    # r = sol.combine(20, 16)
    # print(time.time() - a)
    # print(sol.findWords(
    #     [
    #         ['o', 'a', 'a', 'n'],
    #         ['e', 't', 'a', 'e'],
    #         ['i', 'h', 'k', 'r'],
    #         ['i', 'f', 'l', 'v']
    #     ],
    #     ["oath", "pea", "eat", "rain"]
    # )
    # )
    # print(sol.findWords(
    #     [['aa']], ['aa']
    # )
    # )
    # print(sol.findLongestChain([[1, 2], [2, 3], [3, 4]]))
    # print(sol.validPalindrome('abac'))
    # print(sol.validPalindrome(s))
    # print(sol.isPalindrome('A man, a plan, a canal: Panama'))
    # print(sol.isPalindrome('ab2a'))
    # print(sol.jump([2, 3, 1, 1, 4]))
    # print(sol.change(5, [1, 2, 5]))
    # print(sol.minDistance('abc', 'abb'))
    # print(sol.longestConsecutive([100, 4, 200, 1, 3, 2]))
    # print(sol.longestConsecutive([]))
    # root = buildTree([0, None, 1])
    # print(id(root))
    # sol.flatten(root)
    # print(root, id(root))
    # print(sol.findNumberOfLIS([1, 3, 5, 4, 7]))
    # print(sol.findNumberOfLIS([1, 2, 4, 3, 5, 4, 7, 2]))
    # print(sol.findNumberOfLIS([2, 2, 2, 2]))
    # print(sol.findNumberOfLIS([]))
    # print(sol.lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]))
    # print(sol.lengthOfLIS([-2, -1]))
    # print(sol.threeSumClosest([-1, 2, 1, -4], 1))
    # print(sol.threeSumClosest([1, 1, 1, 1], 0))
    # head = buildLinkList([1, 2, 3, 1, 2, 4, 5, 6, 3])
    # head = buildLinkList([2, 1])
    # r = sol.sortList(head)
    # nums = [1, 2, 1, 2, 2, 1, 2, 1, 0, 0, 2]
    # sol.sortColors(nums)
    # print(nums)
    # nums = [2, 3, 1, 3, 3]
    # [2, 3, 3, 3, 1]

    # nums = [2, 3, 3, 1, 3]
    # nums = [3, 2, 1]
    # # nums = [1, 2, 3]
    # sol.nextPermutation(nums)
    # print(nums)
    # print(sol.findMaxForm(["10", "0001", "111001", "1", "0"], 5, 3))
    # print(sol.findMaxForm(["10", "0", "1"], 5, 3))
    # print(sol.repeatedSubstringPattern('abababc'))
    # print(sol.repeatedSubstringPattern('aba'))
    # print(sol.repeatedSubstringPattern("abaababaab"))
    print(sol.diffWaysToCompute('2*3-4*5'))
    # print(sol.diffWaysToCompute('45465'))
