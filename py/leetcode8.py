import collections


class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []


class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None


class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class TrieNode:
    def __init__(self, val=None):
        self.val = val
        self.word = False
        self.next = collections.defaultdict(TrieNode)


class Trie:
    def __init__(self):
        self.root = TrieNode('null')

    def add(self, word):
        cur = self.root
        for i in word:
            cur = cur.next[i]
        cur.word = True

    def judge(self, word):
        cur = self.root
        ret = ''
        for i in word:
            ret += i
            cur = cur.next[i]
            if not cur:
                return word
            if cur.word:
                return ret
        return None


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        # 654
        if not nums:
            return None
        if len(nums) == 1:
            return TreeNode(nums[0])
        max_i = 0
        for i, v in enumerate(nums):
            if v > nums[max_i]:
                max_i = i
        ret = TreeNode(nums[max_i])
        ret.left = self.constructMaximumBinaryTree(nums[:max_i])
        ret.right = self.constructMaximumBinaryTree(nums[max_i + 1:])
        return ret

    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # 392
        if not s:
            return True
        if len(s) > len(t):
            return False
        idx = 0
        for i, c in enumerate(t):
            if c == s[idx]:
                idx += 1
            if idx >= len(s):
                return True
        return idx == len(s)

    def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # 201
        if n == 0:
            return 0
        count = 1
        while m != n:
            m >>= 1
            n >>= 1
            count <<= 1
        return count * n

    def getIntersectionNode(self, headA: ListNode, headB: ListNode):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        # 160
        if not headA or not headB:
            return None
        pa = headA
        pb = headB
        while pa != pb:
            pa = headB if not pa else pa.next
            pb = headA if not pb else pb.next
        return pa

    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 696
        preLen = 0
        curLen = 1
        res = 0
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                curLen += 1
            else:
                preLen = curLen
                curLen = 1
            if preLen >= curLen:
                res += 1
        return res

    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        # 289
        m = len(board)
        n = len(board[0])

        def neighbors(x, y):
            # if x < 0 or x >= m or y < 0 or y >= n:
            #     return 0
            res = 0
            for i in range(-1, 1 + 1):
                for j in range(-1, 1 + 1):
                    if i == 0 and j == 0:
                        continue
                    nx = x + i
                    ny = y + j
                    if nx < 0:
                        nx = nx + m
                    elif nx >= m:
                        nx -= m
                    if ny < 0:
                        ny += n
                    elif ny >= n:
                        ny -= n
                    if board[nx][ny] & 1:
                        res += 1
            return res

        for x in range(m):
            for y in range(n):
                count = neighbors(x, y)
                if board[x][y] & 1:
                    if count < 2:
                        board[x][y] &= 1
                    elif count < 4:
                        board[x][y] |= 2
                    else:
                        board[x][y] &= 1
                elif count == 3:
                    board[x][y] |= 2
        for x in range(m):
            for y in range(n):
                board[x][y] >>= 1

        return board

    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ret = []
        for i in range(1, n + 1):
            t = ""
            if i % 3 == 0:
                t += "Fizz"
            if i % 5 == 0:
                t += "Buzz"
            if not t:
                t = str(i)
            ret.append(t)
        return ret

    def replaceWords(self, words, sentence):
        """
        :type dict: List[str]
        :type sentence: str
        :rtype: str
        """
        # 648
        trie = Trie()
        for w in words:
            trie.add(w)
        tokens = sentence.split(' ')
        for i in range(len(tokens)):
            r = trie.judge(tokens[i])
            if r:
                tokens[i] = r
        return ' '.join(tokens)

    def isMatch(self, text, pattern):
        if not pattern:
            return not text

        first_match = bool(text) and pattern[0] in {text[0], '.'}

        if len(pattern) >= 2 and pattern[1] == '*':
            return (self.isMatch(text, pattern[2:]) or
                    first_match and self.isMatch(text[1:], pattern))
        else:
            return first_match and self.isMatch(text[1:], pattern[1:])

    def isMatch_(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """

        # 10
        def dfs(curs, curp):
            if curs >= len(s) or curp >= len(p):
                return False
            rep = ''
            if curp + 1 < len(p):
                rep = p[curp + 1]

            if p[curp] != '.' and rep == '*':

                if s[curs] == p[curp] and curs == len(s) - 1 and curp == len(p) - 2:
                    return True
                if s[curs] == p[curp] and dfs(curs + 1, curp):
                    return True
                return dfs(curs, curp + 2)
            elif p[curp] != '.' and rep != '*':
                if s[curs] == p[curp]:

                    if s[curs] == p[curp] and curs == len(s) - 1 and curp == len(p) - 1:
                        return True

                    return dfs(curs + 1, curp + 1)
                else:
                    return False
            elif p[curp] == '.' and rep == '*':

                if curs == len(s) - 1 and curp == len(p) - 2:
                    return True

                return dfs(curs, curp + 2) or dfs(curs + 1, curp)
            elif p[curp] == '.' and rep != '*':
                if curs == len(s) - 1 and curp == len(p) - 1:
                    return True
                return dfs(curs + 1, curp + 1)

        return dfs(0, 0)

    def findKthNumber(self, m, n, k):
        """
        :type m: int
        :type n: int
        :type k: int
        :rtype: int
        """

        # 410
        def binarysearch(num, m, n):
            ret = 0
            for row in range(1, m + 1):
                i = 1
                j = n + 1
                while i < j:
                    mid = (i + j) // 2
                    if row * mid >= num:
                        j = mid
                    else:
                        i = mid + 1
                ret += i - 1
            return ret

        start = 1
        end = m * n + 1
        while start < end:
            mid = (start + end) // 2
            count = binarysearch(mid, m, n)
            if count >= k:
                end = mid
            else:
                start = mid + 1
        return start

    def splitArray(self, nums, m):
        """
        :type nums: List[int]
        :type m: int
        :rtype: int
        """

        def vaild(target, nums, m):
            total = 0
            count = 1
            for n in nums:
                total += n
                if total > target:
                    total = n
                    count += 1
                    if count > m:
                        return False
            return True

        minn = nums[0]
        sums = 0
        for n in nums:
            minn = max(n, minn)
            sums += n
        sums += 1
        while minn < sums:
            mid = (minn + sums) // 2
            if vaild(mid, nums, m):
                sums = mid
            else:
                minn = mid + 1
        return minn

    def checkRecord(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 551
        A = 0
        L = 0
        for i in s:
            if i == 'A':
                A += 1
                if A > 1:
                    return False
            if i == 'L':
                L += 1
                if L >= 3:
                    return False
            else:
                L = 0

        return True

    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        # 51
        path = [-1] * n
        ret = []

        def judge(path, n, v):
            for i in range(n):
                if path[i] == v or i + path[i] == v + n or i - path[i] == n - v:
                    return False
            return True

        def dfs(path, ni):
            if ni == n:
                # print(path)
                temp = []
                for i, v in enumerate(path):
                    ts = ''
                    for j in range(n):
                        if j == v:
                            ts += 'Q'
                        else:
                            ts += '.'
                    temp.append(ts)
                ret.append(temp)
                return
            for i in range(n):
                if judge(path, ni, i):
                    path[ni] = i
                    dfs(path, ni + 1)

        dfs(path, 0)
        return ret

    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        # 518
        dp = [[0 for _ in range(amount + 1)] for __ in range(len(coins) + 1)]
        dp[0][0] = 1
        for i in range(1, len(coins) + 1):
            dp[i][0] = 1
            for j in range(1, amount + 1):
                if j >= coins[i - 1]:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[len(coins)][amount]

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 34
        if not nums:
            return [-1, -1]

        def binary_l(v):
            i = 0
            j = len(nums)
            while i < j:
                mid = (i + j) // 2
                # if nums[mid] == v:
                #     return mid
                if nums[mid] < v:
                    i = mid + 1
                else:
                    j = mid
            return i

        def binary_r(v):
            i = 0
            j = len(nums)
            while i < j:
                mid = (i + j) // 2
                # if nums[mid] == v:
                #     return mid
                if nums[mid] <= v:
                    i = mid + 1
                else:
                    j = mid
            return i

        index = binary_l(target)
        if index >= len(nums) or nums[index] != target:
            return [-1, -1]
        return [index, binary_r(target) - 1]

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        # 79
        visited = set()
        dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))
        m = len(board)
        n = len(board[0])

        def dfs(x, y, wi):
            if not (0 <= x < m and 0 <= y < n and (x, y) not in visited and board[x][y] == word[wi]):
                return False

            if wi == len(word) - 1:
                return True

            visited.add((x, y))
            for dx, dy in dirs:
                nx = dx + x
                ny = dy + y
                if dfs(nx, ny, wi + 1):
                    return True
            visited.remove((x, y))
            return False

        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    visited.clear()
                    if dfs(i, j, 0):
                        return True
        return False

    def findWords_(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """

        # 212
        def hasStr(ts, word):
            last = 0
            for i in range(len(ts)):
                if ts[i] == word[last]:
                    last += 1
                    if last == len(word):
                        return True
                else:
                    last = 0
            last = 0
            for i in reversed(range(len(ts))):
                if ts[i] == word[last]:
                    last += 1
                    if last == len(word):
                        return True
                else:
                    last = 0
            return False

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        m = len(board)

        if m == 1:
            ans = []
            for word in words:
                if hasStr(board[0], word) and word not in ans:
                    ans.append(word)

            return ans

        n = len(board[0])
        startDict = {}
        for i in range(m):
            for j in range(n):
                if board[i][j] not in startDict:
                    startDict[board[i][j]] = []
                startDict[board[i][j]].append((i, j))

        visited = set()

        def dfs(i, j, word, k):
            if (i, j) in visited:
                return False

            if i < 0 or i >= m or j < 0 or j >= n:
                return False

            # print(i, j, word[k])

            if board[i][j] != word[k]:
                return False

            if k == len(word) - 1:
                return True

            visited.add((i, j))
            for d in directions:
                x = i + d[0]
                y = j + d[1]
                if dfs(x, y, word, k + 1):
                    # cache.remove((i, j))
                    return True
            visited.remove((i, j))
            return False

        ans = []
        for word in words:
            visited.clear()
            for i, j in startDict.get(word[0], []):
                if dfs(i, j, word, 0) and word not in ans:
                    ans.append(word)
        return ans

    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # 695
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        m = len(grid)
        n = len(grid[0])
        visited = set()

        def dfs(x, y):
            if x < 0 or x >= m or y < 0 or y >= n or grid[x][y] == 0 or (x, y) in visited:
                return 0
            r = 1
            visited.add((x, y))
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy
                r += dfs(nx, ny)
            return r

        ret = 0
        for i in range(m):
            for j in range(n):
                if (i, j) not in visited:
                    ret = max(ret, dfs(i, j))
        return ret

    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        # 43
        pos = [0] * (len(num1) + len(num2))
        for i in reversed(range(len(num1))):
            for j in reversed(range(len(num2))):
                p1 = i + j
                p2 = p1 + 1
                v = pos[p2] + int(num1[i]) * int(num2[j])
                pos[p1] += v // 10
                pos[p2] = v % 10
        ret = ''
        for i in pos:
            if not ret and i == 0:
                continue
            ret += str(i)
        return ret if ret else '0'

    def triangleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 611
        nums.sort()
        ans = 0
        for i in reversed(range(2, len(nums))):
            l = 0
            r = i - 1
            while l < r:
                if nums[l] + nums[r] > nums[i]:
                    ans += r - l
                    r -= 1
                else:
                    l += 1
        return ans

    def __init__(self):
        self.cache = {}

    def find24(self, nums):
        def getValues(a, b):
            ret = [a + b, a - b, b - a, a * b]
            if a:
                ret.append(b / a)
            if b:
                ret.append(a / b)

            return ret

        cache = self.cache

        def dfs(nums):

            tkey = tuple(sorted(nums))

            if tkey in cache:
                return cache[tkey]

            if len(nums) == 1:
                return abs(nums[0] - 24) < 1e-6

            for pv in range(len(nums) - 1):
                cur = []
                for i in range(len(nums)):
                    if i == pv:
                        cur.append(0)
                    elif i == pv + 1:
                        continue
                    else:
                        cur.append(nums[i])
                for v in getValues(nums[pv], nums[pv + 1]):
                    cur[pv] = v
                    tkey = tuple(sorted(cur))
                    cache[tkey] = dfs(cur)
                    if cache[tkey]:
                        return True
            return False

        return dfs(nums)

    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        # 59
        matrix = [[-1] * n for _ in range(n)]
        dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
        x = 0
        y = -1
        di = 0
        i = 1
        while i <= n * n:
            nx = x + dirs[di][0]
            ny = y + dirs[di][1]
            if nx < n and ny < n and matrix[nx][ny] == -1:
                x = nx
                y = ny
                matrix[x][y] = i
                i += 1
            else:
                di += 1
                if di >= 4:
                    di -= 4
        return matrix

    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        # 54
        m = len(matrix)
        if not m:
            return []
        n = len(matrix[0])
        flags = [[-1] * n for _ in range(m)]
        dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
        x = 0
        y = -1
        di = 0
        i = 0
        ret = [0] * (m * n)
        while i < m * n:
            nx = x + dirs[di][0]
            ny = y + dirs[di][1]
            if nx < m and ny < n and flags[nx][ny] == -1:
                x = nx
                y = ny
                ret[i] = matrix[x][y]
                flags[x][y] = 0
                i += 1
            else:
                di += 1
                if di >= 4:
                    di -= 4
        return ret

    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.m = 0

        def dfs(p):
            if not p:
                return 0

            l = dfs(p.left)
            r = dfs(p.right)
            ret = 1

            ll = 1
            rr = 1

            if p.left and p.left.val == p.val:
                ret += l
                ll += l

            if p.right and p.right.val == p.val:
                ret += r
                rr += r

            self.m = max(self.m, ret)

            return max(ll, rr)

        dfs(root)
        return self.m - 1

    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 32
        stack = []
        for i, v in enumerate(s):
            if v == '(':
                stack.append(i)
            else:
                if stack and s[stack[-1]] == '(':
                    stack.pop()
                else:
                    stack.append(i)
        if not stack:
            return len(s)
        a = len(s)
        b = 0
        ans = 0
        while stack:
            b = stack.pop()
            ans = max(ans, a - b - 1)
            a = b
        ans = max(ans, a)
        return ans

    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        i = 0
        j = len(numbers)
        while i < j:
            n = numbers[i] + numbers[j]
            if n == target:
                return [i + 1, j + 1]
            elif n > target:
                j -= 1
            else:
                i += 1

    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 416
        sums = sum(nums)
        if sums & 1:
            return False
        vis = [0] * len(nums)
        cache = {}

        def dfs(start, cur):
            if cur in cache:
                return cache[cur]
            if cur == 0:
                return True
            for i in range(start, len(nums)):
                if vis[i] == 0 and nums[i] <= cur:
                    vis[i] = 1
                    r = dfs(i + 1, cur - nums[i])
                    cache[cur - nums[i]] = r
                    if r:
                        return True
                    vis[i] = 0
            cache[cur] = False
            return False

        return dfs(0, sums >> 1)

    def canPartitionKSubsets(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        # 698
        if k > len(nums):
            return False
        if k == 1:
            return True
        sums = 0
        for n in nums:
            sums += n
        if sums % k != 0:
            return False
        target = sums // k
        visited = set()

        def dfs(nums, k, start, count, cur):
            if k == 1:
                return True

            if cur == target and count > 0:
                # print(visited)
                return dfs(nums, k - 1, 0, 0, 0)

            for i in range(start, len(nums)):
                if cur + nums[i] <= target and i not in visited:
                    visited.add(i)
                    if dfs(nums, k, i + 1, count + 1, cur + nums[i]):
                        return True
                    visited.remove(i)
            return False

        # dfs(nums, [], set(), target)
        # def dfs_(start, cur_sum, cur_num, k, target):
        #     if k == 1:
        #         return True
        #     if cur_sum == target and cur_num > 0:
        #         return dfs_(0, 0, 0, k - 1, target)
        #     for i in range(start, len(nums)):
        #         if visited[i] == 0 and cur_sum + nums[i] <= target:
        #
        #             visited[i] = 1
        #             if dfs_(i + 1, cur_sum + nums[i], cur_num + 1, k, target):
        #                 return True
        #             visited[i] = 0
        #     return False

        # return len(visited) == len(nums)
        return dfs(nums, k, 0, 0, 0)

    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        # 29
        r = dividend // divisor
        if r >= 2147483648 or r < -2147483648:
            r = 2147483647
        return r

    def rotateRight(self, head: ListNode, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head:
            return head
        tail = head
        l = 1
        while tail.next:
            tail = tail.next
            l += 1
        tail.next = head
        k %= l
        if k:
            for i in range(l - k):
                tail = tail.next
        newH = tail.next
        tail.next = None
        return newH

    def uniquePathsWithObstacles(self, grid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        # 63
        m = len(grid)
        n = len(grid[0])

        if grid[-1][-1] == 1:
            return 0
        cache = {}

        def dp(i, j):
            if i < 0 or i >= m or j < 0 or j >= n:
                return False
            if (i, j) in cache:
                return cache[(i, j)]
            ret = 0
            if 0 <= i - 1 < m and not grid[i - 1][j]:
                ret += dp(i - 1, j)
            if 0 <= j - 1 < n and not grid[i][j - 1]:
                ret += dp(i, j - 1)
            cache[(i, j)] = ret
            return ret

        if grid[0][0] == 0:
            cache[(0, 0)] = 1

        return dp(m - 1, n - 1)

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        # 81
        lo = 0
        hi = len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] == target:
                return True
            if nums[mid] > nums[hi]:
                if nums[mid] > target and nums[lo] <= target:
                    hi = mid
                else:
                    lo = mid + 1
            elif nums[mid] < nums[hi]:
                if nums[mid] < target and nums[hi] >= target:
                    lo = mid + 1
                else:
                    hi = mid
            else:
                hi -= 1
        return nums[lo] == target

    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 82
        if not head:
            return None
        if not head.next:
            return head
        val = head.val
        p = head.next

        if p.val != val:
            head.next = self.deleteDuplicates(p)
            return head
        else:
            while p and p.val == val:
                p = p.next
            return self.deleteDuplicates(p)

    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """

        # 119
        def cnr(m, n):
            v = 1
            i = m
            while i > m - n:
                v *= i
                i -= 1

            v2 = 1
            for i in range(1, n + 1):
                v2 *= i
            return v // v2

        ret = [0] * (rowIndex + 1)
        lenght = rowIndex + 1

        for i in range((lenght + 1) // 2):
            ret[i] = cnr(rowIndex, i)
            ret[rowIndex - i] = ret[i]

        return ret

    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        # 109
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        if not head.next.next:
            v = head.next
            head.next = None
            r = TreeNode(v.val)
            r.left = self.sortedListToBST(head)
            return r

        low = head
        fast = head
        pre = head
        while fast and fast.next:
            fast = fast.next.next
            pre = low
            low = low.next

        pre.next = None
        r = TreeNode(low.val)
        r.left = self.sortedListToBST(head)
        r.right = self.sortedListToBST(low.next)
        return r

    def connect(self, root: TreeLinkNode):
        levels = []

        def preorder(p, dep):
            if not p:
                return

            if len(levels) < dep + 1:
                levels.append(p)
            else:
                levels[dep].next = p
                levels[dep] = p
            preorder(p.left, dep + 1)
            preorder(p.right, dep + 1)

        preorder(root, 0)
        # return root

    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        # 130

        if not board:
            return
        m = len(board)
        n = len(board[0])

        dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))

        bounds = []
        for i in range(m):
            if board[i][0] == 'O':
                bounds.append((i, 0))
            if board[i][n - 1] == 'O':
                bounds.append((i, n - 1))

        for j in range(n):
            if board[0][j] == 'O':
                bounds.append((0, j))
            if board[m - 1][j] == 'O':
                bounds.append((m - 1, j))

        for (x, y) in bounds:
            if board[x][y] != 'O':
                continue
            q = [(x, y)]
            while q:
                x, y = q.pop()
                if not board[x][y] == 'O':
                    continue
                board[x][y] = 'S'
                for dx, dy in dirs:
                    x2 = x + dx
                    y2 = y + dy
                    if 0 <= x2 < m and 0 <= y2 < n and board[x2][y2] == 'O':
                        q.append((x2, y2))
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'S':
                    board[i][j] = 'O'
                else:
                    board[i][j] = 'X'

    # def __init__(self):
    #     self.d = {}

    def cloneGraph(self, node: UndirectedGraphNode):
        if not node:
            return None
        d = {}
        q = []
        ret = UndirectedGraphNode(node.label)
        d[ret.label] = ret
        q.append(ret)
        while q:
            cur = q.pop(0)
            if cur.label in d:
                continue
            new = UndirectedGraphNode(cur.label)
            for ni in cur.neighbors:
                if ni.label in d:
                    new.neighbors.append(d[ni.label])
                else:
                    r = UndirectedGraphNode(ni.label)
                    d[ni.label] = r
                    new.neighbors.append(r)
                    q.append(r)
        return ret

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 142
        # if not head:
        #     return None
        # dummy = ListNode(-1)
        # dummy.next = head
        # fast = head.next
        # slow = head
        # while fast and slow != fast:
        #     if fast.next:
        #         fast = fast.next.next
        #     else:
        #         return None
        #     slow = slow.next
        #
        # if fast != slow:
        #     return None
        #
        # d = set()
        # d.add(slow)
        # slow = slow.next
        # while slow != fast:
        #     d.add(slow)
        #     slow = slow.next
        #
        # cur = dummy
        # while cur.next:
        #     if cur.next in d:
        #         return cur.next
        #     else:
        #         cur = cur.next
        if not head or not head.next:
            return None
        entry, slow, fast = head, head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                while entry != slow:
                    print(slow.val, fast.val)
                    slow = slow.next
                    entry = entry.next
                return entry
        return None

    def serverseList(self, head: ListNode):
        pre = None
        cur = head
        while cur:
            t = cur.next
            cur.next = pre
            pre = cur
            cur = t
        return pre

    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        # 143
        if not head:
            return
        slow = head
        fast = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        middle = slow
        p2 = middle.next
        middle.next = None
        while p2:
            n = p2.next
            p2.next = middle.next
            middle.next = p2
            p2 = n
        l2 = middle.next
        middle.next = None

        l1 = head.next
        p = head
        p.next = None
        flag = 2
        while l1 and l2:
            if flag == 2:
                p.next = l2
                p = p.next
                l2 = l2.next
                flag = 1
            else:
                flag = 2
                p.next = l1
                p = p.next
                l1 = l1.next
        if l1:
            p.next = l1
        elif l2:
            p.next = l2


def preorder(p: TreeNode):
    if not p:
        print("#", end=' ')
        return
    print(p.val, end=' ')
    preorder(p.left)
    preorder(p.right)


def buildList(nums):
    head = ListNode(0)
    p = head
    for i in nums:
        p.next = ListNode(i)
        p = p.next
    return head.next


# def displayGameOfLife(board):
#     m = len(board)
#     n = len(board[0])
#     for i in range(m):
#         for j in range(n):
#             if board[i][j] == 0:
#                 print('-', end=' ')
#             else:
#                 print('+', end=' ')
#         print()
#     print()

if __name__ == '__main__':
    sol = Solution()
    # print(sol.isSubsequence("abd", "ahbgdc"))
    # print(sol.isSubsequence("abx", "ahbgdc"))
    # print(sol.rangeBitwiseAnd(5, 7))
    # print(sol.rangeBitwiseAnd(4, 6))
    # print(sol.countBinarySubstrings('00110011'))
    # board = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ]
    # print(sol.fizzBuzz(15))
    # print(sol.replaceWords(["cat", "bat", "rat"], "the cattle was rattled by the battery"))
    # print(sol.isMatch('a', 'ab*'))
    # print(sol.findKthNumber(3, 3, 5))
    # print(sol.findKthNumber(2, 3, 6))
    # print(sol.splitArray([7, 2, 5, 10, 8], 2))
    # print(sol.splitArray([1, 2147483647], 2))
    # print(sol.checkRecord('LALL'))
    # print(sol.solveNQueens(4))
    # print(sol.change(5, [1, 2, 5]))
    # print(sol.searchRange([5, 7, 7, 8, 8, 10], 10))
    # print(sol.searchRange([5], 5))
    # board = [
    #     ['A', 'B', 'C', 'E'],
    #     ['S', 'F', 'C', 'S'],
    #     ['A', 'D', 'E', 'E']
    # ]
    # board = [
    #     ["A", "B", "C", "E"],
    #     ["S", "F", "E", "S"],
    #     ["A", "D", "E", "E"]
    # ]

    # word = "ABCESEEEFS"
    # board = [
    #     ['o', 'a', 'a', 'n'],
    #     ['e', 't', 'a', 'e'],
    #     ['i', 'h', 'k', 'r'],
    #     ['i', 'f', 'l', 'v']
    # ]
    # words = ["eat", "oath"]
    # print(sol.findWords(board, words))
    # print(sol.findWords([['a', 'b']], ['ba']))
    # print(sol.findWords(
    #     [
    #         ["a", "b"],
    #         ["c", "d"]
    #     ],
    #     ["acdb"]))
    # grid = [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    #         [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]
    # # print(sol.maxAreaOfIsland(grid))
    # print(sol.maxAreaOfIsland([[0, 0, 0, 0, 0, 0, 1]]))
    # print(sol.multiply('999999', '444'))
    # print(sol.multiply('214646456478987983213131654', '6546546546456464645'))
    # print(sol.triangleNumber([2, 2, 3, 4]))
    # print(sol.find24([3, 3, 8, 8]))
    #
    # n = 10
    #
    # r = sol.generateMatrix(n)
    # for i in range(n):
    #     for j in range(n):
    #         print('{:3d} '.format(r[i][j]), end='')
    #     print()
    # print(sol.spiralOrder([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ]))
    # root = TreeNode(1)
    # root.left = TreeNode(4)
    # root.left.left = TreeNode(4)
    # root.left.right = TreeNode(4)
    #
    # root.right = TreeNode(5)
    # root.right.right = TreeNode(5)
    # print(sol.longestUnivaluePath(root))
    # print(sol.longestValidParentheses("())"))
    # print(sol.canPartitionKSubsets([4, 3, 2, 3, 5, 2, 1], 5))
    # print(sol.canPartitionKSubsets([4, 3, 2, 3, 5, 2, 1], 4))
    # print(sol.canPartition(
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100]))
    # print(sol.canPartition([1, 5, 11, 5]))
    # print(sol.divide(1004958205, -2137325331))
    # print(sol.uniquePathsWithObstacles([
    #     [0, 0, 0],
    #     [0, 1, 0],
    #     [0, 1, 0],
    #     [0, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 0],
    # ]))
    # print(sol.uniquePathsWithObstacles([[0, 1]]))
    # print(sol.search([]))
    # l = buildList([1, 2, 3, 3, 4, 4, 5])
    # l = buildList([1, 1, 1, 2, 3, 3])
    # # l = buildList([1, 2, 2])
    # l2 = sol.deleteDuplicates(l)
    # p = l2
    # while p:
    #     print(p.val, end=',')
    #     p = p.next
    # print()
    # for i in range(10):
    #     print(sol.getRow(i))
    # l = buildList([1, 2, 3, 4, 5, 6])
    # r = sol.sortedListToBST(l)
    # preorder(r)
    # print()
    # board = [
    #     ['X', 'X', 'X', 'X'],
    #     ['X', 'O', 'O', 'X'],
    #     ['X', 'X', 'O', 'X'],
    #     ['X', 'O', 'X', 'X'],
    # ]
    # board = [
    #     ["X", "O", "X", "X"],
    #     ["O", "X", "O", "X"],
    #     ["X", "O", "X", "O"],
    #     ["O", "X", "O", "X"],
    #     ["X", "O", "X", "O"],
    #     ["O", "X", "O", "X"]
    # ]
    # board = [
    #     ["X", "X", "X", "X"],
    #     ["X", "O", "O", "X"],
    #     ["X", "X", "O", "X"],
    #     ["X", "O", "X", "X"]
    # ]
    # sol.solve(board)
    # for i in board:
    #     print(i)
    # print()
    # head = buildList([1, 2, 3, 4, 5, 6])
    # p = head
    # while p.val != 6:
    #     p = p.next
    # q = head
    # while q.val != 3:
    #     q = q.next
    #
    # p.next = q
    #
    # r = sol.detectCycle(head)
    # print(r and r.val)
    # head = buildList([1, 2, 3, 4, 5, 6, 7, 8])
    # head = buildList([1, 2, 3, 4, 5])
    head = buildList([1, 2, 3])
    sol.reorderList(head)
    p = head
    while p:
        print(p.val, end=' ')
        p = p.next
    print()
