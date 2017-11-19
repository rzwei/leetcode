import math


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """
        self.nodes = ''

        def preorder(p):
            if not p:
                self.nodes += ',#'
                return
            self.nodes += ',' + str(p.val)
            preorder(p.left)
            preorder(p.right)

        preorder(root)
        return self.nodes[1:]

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        tokens = data.split(',')
        nodes = []
        for i in tokens:
            if i == '#':
                nodes.append('#')
            else:
                nodes.append(int(i))

        def build(nodes):
            cur = nodes.pop(0)
            if cur == '#':
                return None
            p = TreeNode(cur)
            p.left = build(nodes)
            p.right = build(nodes)
            return p

        return build(nodes)


class Solution:
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if len(nums) <= 3:
            return min(nums)
        pv = nums[0]
        i = 0
        j = len(nums) - 1
        while i <= j:
            mid = (i + j) // 2
            v = nums[mid]
            # print(i, j)
            if v < pv and mid - 1 >= 0 and nums[mid - 1] >= pv:
                return nums[mid]
            if v > pv and mid + 1 < len(nums) and nums[mid + 1] <= pv:
                return nums[mid + 1]
            if v >= pv:
                i = mid + 1
            else:
                j = mid - 1
        return nums[0]

    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        if not word1 or not word2:
            return len(word2) + len(word1)

        dp = [[0 for i in range(len(word2) + 1)] for j in range(len(word1) + 1)]

        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                vi = word1[i - 1]
                vj = word2[j - 1]
                if vi == vj:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        r = dp[-1][-1]
        return len(word1) + len(word2) - r - r

    def findTilt(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.r = 0

        def travel(p):
            if not p:
                return 0
            if not p.left and not p.right:
                return p.val
            lv = travel(p.left)
            rv = travel(p.right)
            self.r += abs(rv - lv)
            return lv + rv + p.val

        travel(root)
        return self.r

    def imageSmoother(self, M):
        """
        :type M: List[List[int]]
        :rtype: List[List[int]]
        """
        m = len(M)
        n = len(M[0])
        ret = [[0 for i in range(n)] for j in range(m)]

        dirs = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

        def inScalar(i, j):
            return 0 <= i < m and 0 <= j < n

        for i in range(m):
            for j in range(n):
                r = M[i][j]
                nn = 1
                for d in dirs:
                    x = i + d[0]
                    y = j + d[1]
                    if inScalar(x, y):
                        r += M[x][y]
                        nn += 1
                ret[i][j] = r // nn

        return ret

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

    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 222
        if not root:
            return 0

        def height(p):
            if not p:
                return 0
            return height(p.left) + 1

        hl = height(root.left)
        hr = height(root.right)
        if hl == hr:
            return (1 << hl) + self.countNodes(root.right)
        else:
            return (1 << hr) + self.countNodes(root.left)

    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        if len(nums) == 1:
            return TreeNode(nums[0])
        if len(nums) == 2:
            r = TreeNode(nums[1])
            r.left = TreeNode(nums[0])
            return r
        if len(nums) == 3:
            r = TreeNode(nums[1])
            r.left = TreeNode(nums[0])
            r.right = TreeNode(nums[2])
            return r
        i = 0
        j = len(nums) - 1
        mid = (i + j) // 2
        r = TreeNode(nums[mid])
        r.left = self.sortedArrayToBST(nums[i:mid])
        r.right = self.sortedArrayToBST(nums[mid + 1:j + 1])
        return r

    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # 144
        stack = [root]
        ret = []
        while stack:
            p = stack.pop()
            if not p:
                continue
            else:
                ret.append(p.val)
                stack.append(p.right)
                stack.append(p.left)
        return ret

    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        d = {}
        for i in nums:
            d[i] = d.get(i, 0) + 1
        maxv = 0
        for k in d:
            if k + 1 in d:
                maxv = max(maxv, d[k] + d[k + 1])
        return maxv

    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if k < 0:
            return 0
        if k == 0:
            d = {}
            ans = set()
            for i in nums:
                d[i] = d.get(i, 0) + 1
                if d[i] >= 2:
                    ans.add((i, i))
            return len(ans)

        ans = set()
        nums_s = set(nums)
        for i in nums_s:
            if i + k in nums_s:
                ans.add((i, i + k))
        return len(ans)

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        # 200
        m = len(grid)
        n = len(grid[0])
        M = [[-1 for i in range(n)] for j in range(m)]

        dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]

        def inGrid(i, j):
            return 0 <= i < m and 0 <= j < n

        def dfs(i, j):
            if not inGrid(i, j):
                return
            if M[i][j] != -1:
                return

            M[i][j] = 0
            for d in dirs:
                x = i + d[0]
                y = j + d[1]
                if inGrid(x, y) and M[x][y] == -1 and grid[x][y] == '1':
                    dfs(x, y)

        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and M[i][j] == -1:
                    dfs(i, j)
                    ans += 1
        return ans

    def findFrequentTreeSum(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # 508
        d = {}

        def postorder(p):
            if not p:
                return 0
            r = postorder(p.left) + postorder(p.right)
            s = r + p.val
            d[s] = d.get(s, 0) + 1
            return r + p.val

        postorder(root)
        maxs = []
        maxv = -1
        for k, v in d.items():
            if v > maxv:
                maxs = [k]
                maxv = v
            elif v == maxv:
                maxs.append(k)
        return maxs

    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        i = 0
        j = 0
        s = 0
        maxs = -2147483648

        while j < len(nums):
            if j - i < k - 1:
                s += nums[j]
                j += 1
            elif j - i == k - 1:
                s += nums[j]
                if s > maxs:
                    maxs = s
                s -= nums[i]
                i += 1
                j += 1
        return maxs / k

    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # 239
        ans = []
        i = 0
        j = k - 1
        while j < len(nums):
            ans.append(max(nums[i:j + 1]))
            j += 1
            i += 1
        return ans

    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """

        if n <= 26:
            return chr(ord('A') + n - 1)
        t = n % 26
        return self.convertToTitle((n - 1) // 26) + chr(ord('A') + t)

    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 665
        t = 0
        i = 1
        while i < len(nums):
            if nums[i - 1] > nums[i]:
                t += 1
                if i - 2 < 0 or nums[i - 2] <= nums[i]:
                    nums[i - 1] = nums[i]
                else:
                    nums[i] = nums[i - 1]
            i += 1
        return t <= 1

    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        # 406
        ans = [people.pop()]
        while people:
            v = people.pop()

    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 647

    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 442
        ans = []
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            if nums[index] < 0:
                ans.append(index + 1)
            else:
                nums[index] = -nums[index]

        return ans

    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """

        def guess(n):
            pass

        i = 1
        j = n

        while i <= j:
            mid = (i + j) // 2
            v = guess(mid)
            if v == 0:
                return mid
            elif v == 1:
                i = mid + 1
            else:
                j = mid - 1
        return i

    def countArrangement(self, N):
        """
        :type N: int
        :rtype: int
        """
        # 526
        pass

    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 287
        # for i in range(len(nums)):
        #     index = abs(nums[i]) - 1
        #     if nums[index] < 0:
        #         return index + 1
        #     else:
        #         nums[index] = -nums[index]
        slow = nums[0]
        fast = nums[nums[0]]
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]
        fast = 0
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        return nums[fast]

    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        notPrimes = [False for i in range(n)]
        ans = 0
        for i in range(2, n):
            if not notPrimes[i]:
                ans += 1
                j = 2
                while j * i < n:
                    notPrimes[j * i] = True
                    j += 1
        return ans

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 26
        if len(nums) == 1:
            return 1

        i = 1
        li = 0
        ans = len(nums)

        while i < len(nums):
            if nums[i] == nums[li]:
                ans -= 1
            else:
                li = i
            i += 1
        return ans

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        ans = []
        self.callTimes = 0

        # 93
        def dfs(cur, s, ni):
            self.callTimes += 1

            if not s and ni < 0:
                ans.append(cur[1:])
                return

            if ni < 0:
                return
            i = 1
            ret = 0
            while i <= 3 and i <= len(s):
                n = s[:i]
                if i >= 2 and n[0] == '0':
                    i += 1
                    continue
                # if 0 <= int(n) <= 255 and len(s[i:]) >= ni - 1:
                if 0 <= int(n) <= 255:
                    dfs(cur + '.' + n, s[i:], ni - 1)
                i += 1
            return ret

        dfs('', s, 3)
        # print(self.callTimes)
        return ans

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # 33

        if not nums:
            return -1

        pv = 0
        i = 0
        j = len(nums) - 1

        s = nums[i]
        e = nums[j]

        while i < j:
            mid = (i + j) // 2
            if nums[mid] <= e:
                j = mid
            else:
                i = mid + 1
        pv = i
        i = 0
        j = len(nums) - 1
        ln = len(nums)

        while i <= j:
            mid = (i + j) // 2
            realmid = (mid + pv) % ln
            if nums[realmid] == target:
                return realmid
            if nums[realmid] < target:
                i = mid + 1
            else:
                j = mid - 1
        return -1

    def checkPerfectNumber(self, num):
        """
        :type num: int
        :rtype: bool
        """
        # 507
        if num == 1:
            return False
        ln = int(math.ceil(math.sqrt(num)))
        s = 1
        for i in range(2, ln):
            if num % i == 0:
                s += i
                print(i)
                if i != num // i:
                    print(num // i)
                    s += num // i
        return s == num

    def buildList(self, nums):
        head = ListNode(0)
        p = head
        for i in nums:
            p.next = ListNode(i)
            p = p.next
        return head.next

    def showList(self, head: ListNode):
        p = head
        while p:
            print(p.val, end=' ')
            p = p.next
        print()

    def removeNthFromEnd(self, head: ListNode, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        # 19
        start = ListNode(0)
        slow = start
        fast = start
        slow.next = head
        for i in range(n + 1):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return start.next

    def arrangeCoins(self, n):
        """
        :type n: int
        :rtype: int
        """
        ans = 0
        i = 1
        while True:
            if n < i:
                break
            ans += 1
            n -= i
            i += 1
        return ans

    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 162
        i = 0
        j = len(nums) - 1
        while i < j:
            mid = (i + j) // 2
            mid2 = mid + 1
            if nums[mid] < nums[mid2]:
                i = mid2
            else:
                j = mid
        return i

    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 172
        ans = 0
        i = 5
        while i <= n and n // i > 0:
            ans += n // i
            i *= 5
        return ans

    def convertBST(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        # 538
        self.sums = 0

        def travel(p):
            if not p:
                return
            travel(p.right)
            v = p.val + self.sums
            p.val = v
            self.sums = v
            travel(p.left)

        travel(root)
        return root

    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        # 141
        if not head:
            return False

        slow = head
        fast = head
        fast = fast.next
        while fast and slow != fast:
            if fast.next:
                fast = fast.next.next
            else:
                return False
            slow = slow.next
        return True if slow == fast else False

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        # 101
        def fun(p, q):
            if not p or not q:
                return p == q
            if p.val != q.val:
                return False
            return fun(p.left, q.right) and fun(p.right, q.left)

        if not root:
            return True
        return fun(root.left, root.right)

    def isSubtree(self, s: TreeNode, t: TreeNode):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """

        # 572
        def fun(now, t):
            if not now and not t:
                return True
            if not now or not t:
                return False
            if now.val != t.val:
                return False
            return fun(now.left, t.left) and fun(now.right, t.right)

        def travel(p):
            if not p:
                return False
            if p.val == t.val and fun(p, t):
                return True
            return travel(p.left) or travel(p.right)

        return travel(s)

    def countSegments(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 434
        s = s.strip()
        tokens = s.split(' ')
        ans = 0
        for i in tokens:
            if i and not i.isspace():
                ans += 1
        return ans

    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        k = len(nums) - k
        temp = []
        offset = len(nums) - k
        for i in nums[:k]:
            temp.append(i)
        for i in range(0, offset):
            nums[i] = nums[i + k]
        for i in range(offset, len(nums)):
            nums[i] = temp[i - offset]

            # k = k % len(nums)
            #
            # k = len(nums) - k
            #
            # while k:
            #     temp = nums[0]
            #     for i in range(0, len(nums) - 1):
            #         nums[i] = nums[i + 1]
            #     nums[-1] = temp
            #     k -= 1

    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        # 476
        mask = 1
        while mask < num:
            mask = mask << 1 | 1
        return (num ^ 0x7fffffff) & mask

    def __init__(self):
        self.cache = {}
        self.cache[0] = 1
        self.cache[1] = 1
        self.cache[2] = 2

    def numTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        # 96
        if n in self.cache:
            return self.cache[n]
        # if n == 0 or n == 1:
        #     return 1
        # if n == 2:
        #     return 2
        ret = 0
        for i in range(n):
            ret += self.numTrees(i) * self.numTrees(n - i - 1)
        self.cache[n] = ret
        return ret

    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        # 95
        nums = [i for i in range(1, n + 1)]

        # def buildTree(nums):
        #     if len(nums)==1:
        #         return TreeNode(0)
        #     if len(nums)==2:
        #         maxv=0 if nums[0]>nums[1] else 1
        #
        pass

    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        # 338
        # def get1(n):
        #     ans = 0
        #     while n:
        #         ans += 1
        #         n &= n - 1
        #     return ans
        #
        # ret = []
        # for i in range(0, num + 1):
        #     ret.append(get1(i))
        # return ret
        ret = [0 for i in range(num + 1)]
        for i in range(1, num + 1):
            ret[i] = ret[i >> 1] + (i & 1)
        return ret

    def getMoneyAmount(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 375
        i = 1
        j = n
        target = n
        cost = 0
        while i <= j:
            mid = (i + j) // 2
            print(i, j, mid)
            if mid == target:
                return cost
            cost += mid
            if mid < target:
                i = mid + 1
            else:
                j = mid - 1
        return cost

    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        i = m - 1
        j = n - 1
        k = m + n - 1
        while j >= 0:
            if i >= 0 and nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1

    def sumNumbers(self, root: TreeNode):
        """
        :type root: TreeNode
        :rtype: int
        """

        # 129

        def dfs(p, s):
            if not p:
                return 0
            if not p.left and not p.right:
                return s * 10 + p.val
            return dfs(p.left, s * 10 + p.val) + dfs(p.right, s * 10 + p.val)

        return dfs(root, 0)




if __name__ == '__main__':
    sol = Solution()
    # print(sol.findMin([7, 7, 7, 7, 7, 1, 1]))
    # print(sol.findMin([7, 1, 7, 7, 7, 7, 7]))
    # print(sol.findMin([7, 7, 7, 7, 1, 7, 7]))
    # print(sol.findMin([1, 1, 1, 1, 1, 1, 7]))
    # print(sol.findMin([4, 5, 6, 7, 8, 8, 0]))
    # print(sol.findMin([4, 4, 6, 7, 8, 9, 10]))
    # print(sol.minDistance('sea', 'eat'))
    # print(sol.minDistance('a', 'ab'))
    # print(sol.imageSmoother([[1, 1, 1],
    #                          [1, 0, 1],
    #                          [1, 1, 1]]))
    # root = sol.buildTree([4, 2, 5, 1, 7, 3], [4, 5, 2, 7, 3, 1])
    # print(sol.countNodes(root))
    # root = sol.sortedArrayToBST([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(sol.preorderTraversal(root))
    # c = Codec()
    # r = c.serialize(None)
    # print(r)
    # r2 = c.deserialize(r)
    # print(sol.preorderTraversal(r2))
    # print(sol.findLHS([1, 3, 2, 2, 5, 2, 3, 7]))
    # print(sol.findPairs([1, 2, 3, 1, 5], 0))
    # print(sol.numIslands(['11000',
    #                       '11000',
    #                       '00100',
    #                       '00011']))
    # print(sol.findMaxAverage([1, 12, -5, -6, 50, 3], 1))
    # print(sol.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
    # print(sol.convertToTitle(1))
    # print(sol.convertToTitle(25))
    # print(sol.convertToTitle(26))
    # print(sol.convertToTitle(27))
    # print(sol.convertToTitle(28))
    # print(sol.convertToTitle(26 * 2))
    # print(sol.checkPossibility([3, 4, 2, 3]))
    # print(sol.checkPossibility([-1, 4, 2, 3]))
    # print(sol.findDuplicates([4, 3, 2, 7, 8, 2, 3, 1]))
    # print(sol.findDuplicate([1, 2, 3, 4, 5, 6, 5, 6, 6, 7]))
    # print(sol.countPrimes(12))
    # print(sol.removeDuplicates([1, 1, 2, 2]))
    # print(sol.restoreIpAddresses('25525511135'))
    # print(sol.restoreIpAddresses('2552551135'))
    # print(sol.restoreIpAddresses("010010"))
    # print(sol.search([4, 5, 6, 7, 0, 1, 2], 5))
    # print(sol.search([6, 5], 5))

    # head = sol.buildList([1, 2, 3, 4, 5])
    # sol.showList(head)
    # head = sol.removeNthFromEnd(head, 5)
    # sol.showList(head)

    # print(sol.checkPerfectNumber(28))
    # print(sol.arrangeCoins(1))
    # print(sol.arrangeCoins(2))
    # print(sol.arrangeCoins(3))
    # print(sol.arrangeCoins(4))
    # print(sol.arrangeCoins(5))
    # print(sol.arrangeCoins(8))
    # print(sol.findPeakElement([1, 2, 3]))
    # print(sol.findPeakElement([1, ]))
    # print(sol.findPeakElement([1, 2, 3, 4]))
    # print(sol.findPeakElement([4, 3, 2, 1]))
    # print(sol.findPeakElement([3, 4, 3, 2, 1]))
    # print(sol.findPeakElement([4, 1, 5]))
    # print(sol.trailingZeroes(10))
    # root = TreeNode(5)
    # root.left = TreeNode(2)
    # root.right = TreeNode(13)
    # root2 = sol.convertBST(root)
    # print(root2.val)
    # print(root2.left.val)
    # print(root2.right.val)
    # print(sol.countSegments(''))
    # v = [1, 2, 3]
    # v = [1, 2, 3, 4, 5, 6, 7]
    # sol.rotate(v, 1)
    # print(v)
    # print(bin(sol.findComplement(5)))
    # print(sol.numTrees(3))
    # print(sol.countBits(5))
    # print(sol.getMoneyAmount(4))
    # nums1 = [1, 3, 5, 7, 9, 0, 0, 0, 0, 0]
    # nums2 = [2, 4, 6, 8, 10]
    # sol.merge(nums1, 5, nums2, 5)
    # print(nums1)
    # root = TreeNode(1)
    # root.left = TreeNode(2)
    # root.right = TreeNode(3)
    # root.left.left = TreeNode(4
    #
    #                           )
    # print(sol.sumNumbers(root))
    # print(sol.subarraySum([1, 1, 3, 4, 5, 6, 7, 3, 4], 1))
    # print(sol.subarraySum([-1, -1, 1], 0))
