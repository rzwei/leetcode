import math
import random


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# Below is the interface for Iterator, which is already defined for you.
#
class Iterator(object):
    def __init__(self, nums):
        """
        Initializes an iterator object to the beginning of a list.
        :type nums: List[int]
        """

    def hasNext(self):
        """
        Returns true if the iteration has more elements.
        :rtype: bool
        """

    def next(self):
        """
        Returns the next element in the iteration.
        :rtype: int
        """


class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.n = None
        self.p = iterator.next()

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.p

    def next(self):
        """
        :rtype: int
        """
        self.n = self.p
        self.p = self.iterator.next()
        return self.n

    def hasNext(self):
        """
        :rtype: bool
        """


# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].

class NestedInteger(object):
    def isInteger(self):
        """
        @return True if this NestedInteger holds a single integer, rather than a nested list.
        :rtype bool
        """

    def getInteger(self):
        """
        @return the single integer that this NestedInteger holds, if it holds a single integer
        Return None if this NestedInteger holds a nested list
        :rtype int
        """

    def getList(self):
        """
        @return the nested list that this NestedInteger holds, if it holds a nested list
        Return None if this NestedInteger holds a single integer
        :rtype List[NestedInteger]
        """


class NestedIterator(object):
    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        q = []

        def travel(nlist):
            if type(nlist) is list:
                for x in nlist:
                    if x.isInteger():
                        q.append(x.getInteger())
                    else:
                        travel(x.getList())
            else:
                if nlist.isInteger():
                    q.append(nlist.getInteger())
                else:
                    travel(nlist.getList())

        travel(nestedList)
        self.q = q
        self.i = 0

    def next(self):
        """
        :rtype: int
        """
        ret = self.q[self.i]
        self.i += 1
        return ret

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.i < len(self.q)


class MyStack(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = []

    def push(self, x):
        """
        Push element x onto stac.k
        :type x: int
        :rtype: void
        """
        self.queue.append(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.queue.pop()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.queue[-1]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return self.queue == []


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution2(object):
    # def __init__(self, nums):
    #     """
    #
    #     :type nums: List[int]
    #     :type numsSize: int
    #     """
    #     self.d = {}
    #     for i, v in enumerate(nums):
    #         if v not in self.d:
    #             self.d[v] = []
    #         self.d[v].append(i)



    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        pass

    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        nums = []
        for s in strs:
            v = [0, 0]
            for i in s:
                if i == '0':
                    v[0] += 1
                else:
                    v[1] += 1
            nums.append(v)

        dp = [[0 for i in range(n)] for j in range(m)]

    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)

        nums.sort()
        j = 0

        for i in nums:
            if i != j:
                return j
            j += 1
        return j

    def maximumProduct(self, nums: list):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        n = len(nums)

        nums = nums * 2

        ans = nums[0] * nums[1] * nums[2]

        for i in range(n):
            t = nums[i] * nums[i + 1] * nums[i + 2]
            ans = max(ans, t)
        return ans

    def reverseList(self, head: ListNode):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        newhead = ListNode(-1)
        p = head
        while p is not None:
            qn = newhead.next
            qq = p.next

            newhead.next = p

            p.next = qn

            p = qq
        return newhead.next

    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        q = []
        for i in s:
            if len(q) == 0:
                q.append(i)
                continue
            l = q[-1]
            r = i
            if l == '(' and r == ')' or l == '[' and r == ']' or l == '{' and r == '}':
                q.pop()
            else:
                q.append(i)
        return len(q) == 0


def solveSudoku(self, board):
    """
    :type board: List[List[str]]
    :rtype: void Do not return anything, modify board in-place instead.
    """
    pass


def minimumTotal(self, triangle):
    """
    :type triangle: List[List[int]]
    :rtype: int
    """
    n = len(triangle)

    triangle.append([0 for i in range(n + 1)])

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            triangle[i][j] = triangle[i][j] + min(triangle[i + 1][j], triangle[i + 1][j + 1])
    return triangle[0][0]


def subarraySum(self, nums, k):
    length = len(nums)

    sums = [0 for i in range(length + 1)]

    indexes = {}

    c = 0

    indexes[0] = []

    j = 1

    for i in nums:
        c += i
        sums[j] = c
        j += 1
        if c not in indexes:
            indexes[c] = []
        indexes[c].append(j)

    sums[length] = c

    ret = 0
    for i in range(length):
        v = sums[i] + k
        for j in indexes.get(v, []):
            if j > i:
                ret += 1
    return ret


def canConstruct(self, ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """
    d = {}
    for i in magazine:
        d[i] = d.get(i, 0) + 1
    for i in ransomNote:
        if d.get(i, 0) <= 0:
            return False
        d[i] -= 1
    return True


def pick(self, target):
    """
    :type target: int
    :rtype: int
    """
    lv = len(self.d[target])
    vi = random.randint(0, lv - 1)
    return self.d[target][vi]


def maxProfit1(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if prices == []:
        return 0
    minv = prices[0]
    ret = 0
    for i, vi in enumerate(prices[1:]):
        minv = minv if minv <= vi else vi
        ret = ret if ret > vi - minv else vi - minv

    return ret

    # todo modify function ---------------------------------------


def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if prices == []:
        return 0

    preProfit = []
    postProfit = []

    ret = 0

    for i in range(len(prices)):
        pre = self.maxProfit1(preProfit[:i])
        post = self.maxProfit1(preProfit[i:])
        preProfit.append(pre)
        postProfit.append(post)
        ret = max((pre, post, ret))
    return ret


def dfs(self, path: set, k):
    self.count += 1

    s = 0
    t = []
    for v in path:
        # print(v, end=',')
        t.append(v)
        s += v
    # print()

    if k == self.maxk:
        if s == self.n:
            t.sort()
            self.ans.add(tuple(t))
        return

    offset = self.n - s
    if offset <= 0:
        return

    if k == self.maxk - 1 and (offset <= 0 or offset in path):
        return

    for i in range(1, 10):
        if i not in path:
            path.add(i)
            self.dfs(path, k + 1)
            path.remove(i)


def dfs_sum3(self, path: list):
    self.count += 1

    s = 0
    paths = []
    for v in path:
        s += v
        paths.append(v)

    if s == self.n:
        paths.sort()
        self.ans.add(tuple(paths))
        return

    if s > self.n:
        return

    for i, v in enumerate(path):
        if v + 1 in path:
            continue
        path[i] += 1
        if path[i] <= 9:
            self.dfs_sum3(path)
        path[i] -= 1


def combinationSum3_2(self, k, n):
    """
    :type k: int
    :type n: int
    :rtype: List[List[int]]
    """
    s = 0
    for i in range(1, k + 1):
        s += i
    if s > n:
        return []
    if s == n:
        return [[i for i in range(1, k + 1)]]

    self.ans = set()
    self.count = 0

    self.maxk = k
    self.n = n

    path = []
    for i in range(1, k + 1):
        path.append(i)

    self.dfs_sum3(path)

    print(self.count)

    ret = []
    for one in self.ans:
        ret.append(list(one))
    return ret


def dfs_combinationSum(self, path: list):
    s = sum(path)
    # print(s)

    if s > self.target:
        return
    for v in self.candidates:
        if s + v == self.target:
            path.append(v)
            tpath = path[:]
            tpath.sort()
            self.ans.add(tuple(tpath))
            path.pop()
        elif s + v < self.target:
            path.append(v)
            self.dfs_combinationSum(path)
            path.pop()


def dfs_combinationSum2(self, path: set):
    s = 0
    for i in path:
        s += self.candidates[i]
        # print(i, end=',')
    # print()
    if s > self.target:
        return

    for j in self.candidates_index:

        v = self.candidates[j]
        if s + v == self.target:
            tpath = [v]
            for i in path:
                tpath.append(self.candidates[i])
            tpath.sort()
            self.ans.add(tuple(tpath))

        elif s + v < self.target:
            path.add(j)
            self.candidates_index.remove(j)

            self.dfs_combinationSum2(path)

            path.remove(j)
            self.candidates_index.add(j)


def combinationSum2(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    self.candidates = candidates
    self.candidates_index = set(range(len(candidates)))
    self.n = len(candidates)

    self.ans = set()
    path = set()
    self.target = target
    self.dfs_combinationSum2(path)
    ret = []
    for one in self.ans:
        ret.append(list(one))
    return ret


def combinationSum(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    self.candidates = candidates
    self.ans = set()
    path = []
    self.target = target
    self.dfs_combinationSum(path)
    ret = []
    for one in self.ans:
        ret.append(list(one))
    return ret


def combinationSum3(self, k, n):
    """
    :type k: int
    :type n: int
    :rtype: List[List[int]]
    """
    s = 0
    for i in range(1, k + 1):
        s += i
    if s > n:
        return [[]]
    if s == n:
        return [[i for i in range(1, k + 1)]]

    self.ans = set()
    self.count = 0

    self.maxk = k
    self.n = n

    path = set()

    self.dfs(path, 0)
    print(self.count)

    ret = []
    for one in self.ans:
        ret.append(list(one))
    return ret


def fourDirection(self, i, j):
    ret = []

    if 0 <= i - 1 < self.n:
        ret.append((i - 1, j))
    if 0 <= i + 1 < self.n:
        ret.append((i + 1, j))
    if 0 <= j + 1 < self.m:
        ret.append((i, j + 1))
    if 0 <= j - 1 < self.m:
        ret.append((i, j - 1))

    return tuple(ret)


def reverseWords(self, s: str):
    """
    :type s: str
    :rtype: str
    """
    s = 0


def nthUglyNumber(self, n):
    """
    :type n: int
    :rtype: int
    """
    dp = [0 for i in range(n)]
    dp[0] = 1
    t2 = 0
    t3 = 0
    t5 = 0
    for i in range(1, n):
        # print(dp[t2], dp[t3], dp[t5])
        dp[i] = min((dp[t2] * 2, dp[t3] * 3, dp[t5] * 5))
        if dp[i] == dp[t2] * 2:
            t2 += 1
        if dp[i] == dp[t3] * 3:
            t3 += 1
        if dp[i] == dp[t5] * 5:
            t5 += 1
    return dp[n - 1]


def isUgly(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 0:
        return False
    while num > 1:
        tv = num
        if num % 2 == 0:
            num //= 2
        if num % 3 == 0:
            num //= 3
        if num % 5 == 0:
            num //= 5
        if tv == num:
            return False
    return True


def findTheDifference(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    n = {}
    for i in t:
        if i not in n:
            n[i] = 0
        n[i] += 1
    for i in s:
        n[i] -= 1
        if n[i] == 0:
            del n[i]
    for k in n.keys():
        return k


def travel_loop(self, path, u):
    while u >= self.length:
        u -= self.length
    while u < 0:
        u += self.length

    if u in self.visit:
        return self.visit[u]

    if len(path) > 0 and u == path[-1]:
        self.visit[u] = False
        return False

    # if len(path) > 0 and self.nums[u] * self.nums[path[-1]] < 0:
    #     # self.visit[u] = False
    #     return False

    if u in path:
        if self.nums[u] * self.nums[path[-1]] > 0:
            self.visit[u] = True
            return True
        else:
            self.visit[u] = False
            return False

    path.append(u)
    f = self.travel_loop(path, self.nums[u] + u)
    path.pop()
    return f


def containsNearbyDuplicate(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    values = {}
    for i, v in enumerate(nums):
        if v not in values:
            values[v] = []
        values[v].append(i)

    for v in values.values():
        last = -1
        for vi in v:
            # print(v, vi)
            if last == -1:
                last = vi
                continue
            if abs(vi - last) <= k:
                return True
            last = vi
    return False


def countPrimes(self, n):
    """
    :type n: int
    :rtype: int
    """
    n -= 1
    self.nums = {}
    nums = self.nums

    def isPrime(n):
        if n in nums:
            return True if nums[n] == 1 else False
        if n == 2 or n == 3 or n == 5:
            return True
        i = 2
        b = int(math.sqrt(n))
        if b + 1 <= n:
            b += 1
        while i < b:
            if n % i == 0:
                return False
            i += 1
        return True

    ans = 0
    ni = 2
    while ni <= n:
        if ni in nums:
            ans += nums[ni]
        else:
            if isPrime(ni):
                ans += 1
                t = ni
                while t <= n:
                    nums[t] = 0
                    t += ni
            else:
                t = ni
                while t <= n:
                    nums[t] = 0
                    t += ni
        ni += 1

    return ans


def pathSum3(self, root, sum_):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: List[List[int]]
    """

    def sum2(nums, v):

        if len(nums) == 1:
            return nums[0] == v

        nums.sort()
        i = 0
        j = len(nums) - 1
        while i < j:
            t = nums[i] + nums[j]
            if t == v:
                return True
            if t < v:
                i += 1
            else:
                j -= 1
        return False

    def sum2_(nums, v):
        if len(nums) == 0:
            return False
        s = 0
        for i in range(len(nums) - 1, -1, -1):
            s += nums[i]
            if s == sum_:
                return True
        return False

    ans = [0]

    def dfs(p, path):
        if p is None:
            return
        path.append(p.val)
        # print(path)
        # v = sum(path)
        if sum2(path[:], sum_):
            ans[0] += 1
        dfs(p.left, path)
        dfs(p.right, path)
        path.pop()

    dfs(root, [])
    return ans


def minDepth(self, root: TreeNode):
    """
    :type root: TreeNode
    :rtype: int
    """

    def travel(p, v):
        if p is None:
            return v
        if p.left is None and p.right is None:
            return v + 1
        return min(travel(p.left, v + 1), travel(p.right, v + 1))

    return travel(root, 0)


def pathSum(self, root, sum_):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: List[List[int]]
    """
    ans = []

    def dfs(p, path):
        if p is None:
            return
        path.append(p.val)
        # print(path)
        v = sum(path)
        if p.left is None and p.right is None and v == sum_:
            ans.append(path.copy())
        dfs(p.left, path)
        dfs(p.right, path)
        path.pop()

    dfs(root, [])
    return ans


def invertTree(self, root: TreeNode):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    ans = []

    def travel(p, s):
        if p is None:
            return
        v = sum(s)
        if p.left is None and p.right is None:
            s.append(p.val)
            ans.append(s.copy())
            s.pop()
        s.append(p.val)
        travel(p.left, s)
        travel(p.right, s)
        s.pop()

    travel(root, [])
    return ans


def circularArrayLoop(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    self.length = len(nums)
    self.visit = {}
    path = []
    self.nums = nums
    for i, v in enumerate(nums):
        if i not in self.visit:
            if self.travel_loop(path, i):
                return True
                # self.visit[i] = False

    return False


def longestPalindrome(self, s):
    """
    :type s: str
    :rtype: int
    """
    letters = {}
    for i in s:
        if i not in letters:
            letters[i] = 0
        letters[i] += 1
    ans = 0

    maxk = 0
    maxv = 0
    for k, v in letters.items():
        if v % 2 != 0 and maxv < v:
            maxv = v
            maxk = k

    for k, v in letters.items():
        if v % 2 == 0:
            ans += v
        elif k == maxk:
            ans += v
        else:
            ans += v - 1
    return ans


def matrixReshape(self, nums, r, c):
    """
    :type nums: List[List[int]]
    :type r: int
    :type c: int
    :rtype: List[List[int]]
    """
    if len(nums) * len(nums[0]) != r * c:
        return nums

    n = len(nums)
    m = len(nums[0])

    ret = [[0 for j in range(c)] for i in range(r)]

    v = 0
    for i in range(r):
        for j in range(c):
            x = v // m
            y = v % m
            v += 1
            ret[i][j] = nums[x][y]
    return ret


def simplifyPath(self, path: str):
    """
    :type path: str
    :rtype: str
    """
    ret = []

    i = 0
    length = len(path)

    while i < length:
        while i < length and path[i] == '/':
            i += 1
        s = ''
        while i < length and path[i] != '/':
            s += path[i]
            i += 1
        if s == '':
            continue
        if s == '..':
            if len(ret) != 0:
                ret.pop()
        elif s == '.':
            continue
        else:
            ret.append(s)
            # if s == '.':
            # i += 1
    s = ''
    if len(ret) == 1:
        s = '/' + ret[0]
    elif len(ret) > 1:
        for v in ret:
            s = s + '/' + v
    else:
        s = '/'
    return s


def updateMatrix(self, matrix: list):
    """
    :type matrix: List[List[int]]
    :rtype: List[List[int]]
    """

    n = len(matrix)
    m = len(matrix[0])
    self.n = n
    self.m = m
    queue = []
    for i, vi in enumerate(matrix):
        for j, vij in enumerate(vi):
            if vij == 0:
                queue.append((i, j))
            else:
                matrix[i][j] = 2147483647
    while len(queue) != 0:
        u, v = queue.pop()
        # print(u, v)
        for i, j in self.fourDirection(u, v):
            if matrix[i][j] > matrix[u][v]:
                matrix[i][j] = matrix[u][v] + 1
                queue.append((i, j))
    return matrix


def groupAnagrams(self, strs):
    """
    :type strs: List[str]
    :rtype: List[List[str]]
    """
    d = {}
    for s in strs:
        tk = [0 for i in range(26)]
        for i in s:
            tk[ord(i) - ord('a')] += 1
        tk = tuple(tk)
        if tk not in d:
            d[tk] = []
        d[tk].append(s)
    ret = []
    for k, v in d.items():
        ret.append(v)
    return ret


def __int__(self):
    self.dep = {}
    self.maxPaths = 0


def travel(self, p: TreeNode, depth: int):
    if p is None:
        return
    if depth not in self.dep:
        self.dep[depth] = p.val
    else:
        self.dep[depth] = max((p.val, self.dep[depth]))

    self.travel(p.left, depth + 1)
    self.travel(p.right, depth + 1)


def largestValues(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    self.travel(root)

    ret = []
    i = 0
    while i in self.dep:
        ret.append(self.dep[i])
        i += 1
    return ret


def trave_max(self, p: TreeNode):
    if p is None:
        return -2147483648

    # if p.left is None and p.right is None:
    #     ret = p.val

    a = self.trave_max(p.left)
    b = self.trave_max(p.right)

    # a -> b <- c

    ret = max((p.val, p.val + a, p.val + b))
    self.maxPaths = max((self.maxPaths, ret, p.val + a + b))
    return ret


def maxPathSum(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    self.maxPaths = -2147483648
    self.trave_max(root)
    return self.maxPaths


def createTree(d: list, p, i):
    p = TreeNode(d[0])

    createTree(d, )


if __name__ == '__main__':
    sol = Solution2()

    m = [[0, 0, 0],
         [0, 1, 0],
         [1, 1, 1]]

    root = TreeNode(1)

    l = TreeNode(2)
    r = TreeNode(3)
    root.left = l

    # root.right = r

    # d = [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1]

    # root2 = TreeNode(-3)

    # t = [20, 34, 8, 30, 26, 33, 28, 19, 21, 28, 22, 15, 33, 19, 12, 9, 17, 9, 11, 7, 5, 14, 31, 14, 12, 6, 29, 20, 27,
    #      24, 23, 34, 23, 18, 29, 6, 8, 23, 20, 25, 8, 30, 27, 7, 6, 34, 11, 10, 8, 9, 34, 30, 10]
    # print(sol.simplifyPath('/home/'))
    # print(sol.simplifyPath('/a/./b/../../c/'))
    # print(sol.isUgly(-2147483648))
    # print(sol.matrixReshape([[1, 2], [3, 4]], 1, 4))
    # print(sol.findTheDifference('abcd', 'abcda'))
    # print(sol.circularArrayLoop([2, -1, 1, 2, 2]))
    # print(sol.circularArrayLoop([-2, 1, -1, -2, -2]))
    # print(sol.circularArrayLoop([-1, 2]))
    # print(sol.containsNearbyDuplicate([1, 2, 4, 3, 2], 3))
    # print(sol.maxProfit([1, 2]))
    # print(sol.countPrimes(1))
    # print(sol.countPrimes(2))
    # print(sol.countPrimes(10))
    # print(sol.countPrimes(100))
    # print(sol.countPrimes(999983))
    # print(sol.subarraySum([-1, -1, 1], 0)) - -----===\
    # print(sol.minimumTotal([[1], [2, 3]]))
    # a = NestedIterator([[1, 1], 2, [1, 1]])
    # v = []
    # while a.hasNext():
    #     v.append(a.next())
    # print(v)
    print(sol.missingNumber([1]))
