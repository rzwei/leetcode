class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.stack = []

        def dfs(p):
            if not p:
                return
            dfs(p.left)
            self.stack.append(p.val)
            dfs(p.right)

        dfs(root)

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) != 0

    def next(self):
        """
        :rtype: int
        """
        return self.stack.pop()


class Solution:
    # def medianSlidingWindow(self, nums, k):
    #     """
    #     :type nums: List[int]
    #     :type k: int
    #     :rtype: List[float]
    #     """
    #     # 480
    #     window = queue.deque()
    #     pass
    # def multiply(self, num1, num2):
    #     """
    #     :type num1: str
    #     :type num2: str
    #     :rtype: str
    #     """
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 53
        ret = nums[0]
        max_v = ret
        for i in range(1, len(nums)):
            ret = max(nums[i], ret + nums[i])
            max_v = max(max_v, ret)
        return max_v

    def maximumSwap(self, num):
        """
        :type num: int
        :rtype: int
        """
        # 670
        nums = list(map(int, list(str(num))))
        ret = num
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                t = nums[i]
                nums[i] = nums[j]
                nums[j] = t
                n = 0
                for k in nums:
                    n = n * 10 + k
                nums[j] = nums[i]
                nums[i] = t
                ret = ret if ret > n else n
        return ret

    # def checkSubarraySum(self, nums, k):
    #     """
    #     :type nums: List[int]
    #     :type k: int
    #     :rtype: bool
    #     """
    #

    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # 62
        n -= 1
        m -= 1

        def fact(n):
            ret = 1
            for i in range(2, n + 1):
                ret *= i
            return ret

        return fact(m + n) // fact(n) // fact(m)

    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        # 518

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 26
        if len(nums) < 2:
            return len(nums)
        id = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[id] = nums[i]
                id += 1
        return id

    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        # 541
        sl = list(s)

        def reverse(i):
            j = i + k - 1
            if j >= len(sl):
                j = len(sl) - 1
            while i < j:
                # print(i,j)
                sl[i], sl[j] = sl[j], sl[i]
                i += 1
                j -= 1

        i = 0
        while i < len(sl):
            if i % 2 == 0:
                reverse(i * k)
            i += 1
        ret = ''
        for i in sl:
            ret += i
        return ret

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


if __name__ == '__main__':
    sol = Solution()
    # print(sol.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    # print(sol.maximumSwap(2736))
    # print(sol.uniquePaths(3, 7))
    # print(sol.uniquePaths(2, 2))
    # v = [1, 1, 2]
    # print(sol.removeDuplicates(v))
    # print(v)
    # print(sol.reverseStr('abcdefg', 2))
    print(sol.find24([3, 3, 8, 8]))
