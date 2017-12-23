import bisect
import heapq
from typing import List


class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

    def __repr__(self):
        return f'I[{self.start},{self.end}]'


class SummaryRanges:
    # 352. Data Stream as Disjoint Intervals
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.intervals = []

    def left(self, intervals, t):
        l = 0
        r = len(intervals)
        while l < r:
            m = (l + r) // 2
            if t > intervals[m].end:
                l = m + 1
            else:
                r = m
        return l

    def addNum(self, val):
        """
        :type val: int
        :rtype: void
        """
        idx = self.left(self.intervals, val)
        s = val
        e = val
        if idx > 0 and self.intervals[idx - 1].end + 1 >= val:
            idx -= 1
        while idx < len(self.intervals) and val + 1 >= self.intervals[idx].start and val - 1 <= self.intervals[idx].end:
            s = min(s, self.intervals[idx].start)
            e = max(e, self.intervals[idx].end)
            del self.intervals[idx]
        self.intervals.insert(idx, Interval(s, e))

    def getIntervals(self):
        """
        :rtype: List[Interval]
        """
        return self.intervals


class NumMatrix(object):
    # 304. Range Sum Query 2D - Immutable
    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        m = len(matrix)
        if m == 0:
            self.sums = []
            return
        n = len(matrix[0])
        self.sums = [[0] * (n + 1) for _ in range(m + 1)]
        sums = self.sums
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                sums[i][j] = matrix[i - 1][j - 1] + sums[i - 1][j] + sums[i][j - 1] - sums[i - 1][j - 1]

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        sums = self.sums

        r = sums[row2 + 1][col2 + 1]
        r -= sums[row2 + 1][col1]
        r -= sums[row1][col2 + 1]
        r += sums[row1][col1]
        return r


class Solution:
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        218. The Skyline Problem  not ac for heap remove
        """

        # def heapRemove(h, v, i):
        #     if i >= len(h) // 2:
        #         return False
        #     if v == h[i]:
        #         h[i], h[-1] = h[-1], h[i]
        #         h.pop()
        #         return True
        #     if v < h[i]:
        #         return False
        #     if heapRemove(h, v, i * 2 + 1):
        #         return True
        #     if heapRemove(h, v, i * 2 + 2):
        #         return True
        #     return False

        result = []
        height = []
        for b in buildings:
            height.append((b[0], -b[2]))
            height.append((b[1], b[2]))
        height.sort()
        h = []
        heapq.heappush(h, 0)
        prev = 0
        for xi, hi in height:
            # print(h)
            if hi < 0:
                heapq.heappush(h, hi)
            else:
                h.remove(-hi)
                # heapRemove(h, -hi, 0)
                heapq.heapify(h)
            cur = h[0]
            if prev != cur:
                result.append([xi, -cur])
                prev = cur
        return result

    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        220. Contains Duplicate III
        """
        if t < 0:
            return False
        w = t + 1
        d = {}
        Len = len(nums)
        for i in range(Len):
            idx = nums[i] // w
            if idx in d:
                return True
            if idx + 1 in d and abs(nums[i] - d[idx + 1]) < w:
                return True
            if idx - 1 in d and abs(nums[i] - d[idx - 1]) < w:
                return True
            d[idx] = nums[i]
            if i >= k:
                del d[nums[i - k] // w]
        return False

    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        224. Basic Calculator
        """
        stack = []
        result = 0
        number = 0
        sign = 1
        for i in range(len(s)):
            c = s[i]
            if c.isdigit():
                number = number * 10 + int(c)
            elif c == '+':
                result += sign * number
                number = 0
                sign = 1
            elif c == '-':
                result += sign * number
                number = 0
                sign = -1
            elif c == '(':
                stack.append(result)
                stack.append(sign)
                result = 0
                sign = 1
            elif c == ')':
                result += sign * number
                number = 0
                result *= stack.pop()
                result += stack.pop()
        if number != 0:
            result += sign * number
        return result

    def calculate2(self, s):
        """
        :type s: str
        :rtype: int
        224. Basic Calculator
        """
        # stack = []
        # result = 0
        # number = 0
        # sign = '+'
        #
        # def operate(v1, v2, op):
        #     if op == '+':
        #         return v1 + v2
        #     elif op == '-':
        #         return v1 - v2
        #     elif op == '*':
        #         return v1 * v2
        #     else:
        #         return v1 // v2
        #
        # for i in range(len(s)):
        #     c = s[i]
        #     # print(stack, c, result, sign)
        #     if c.isspace():
        #         continue
        #     if c.isdigit():
        #         number = number * 10 + int(c)
        #     elif c == '+':
        #         result = operate(result, number, sign)
        #         if stack:
        #             last_sign = stack.pop()
        #             last_result = stack.pop()
        #             result = operate(last_result, result, last_sign)
        #         number = 0
        #         sign = '+'
        #     elif c == '-':
        #         result = operate(result, number, sign)
        #         if stack:
        #             last_sign = stack.pop()
        #             last_result = stack.pop()
        #             result = operate(last_result, result, last_sign)
        #         number = 0
        #         sign = '-'
        #     elif c == '*':
        #         if not stack:
        #             stack.append(result)
        #             stack.append(sign)
        #             result = number
        #         else:
        #             result = operate(result, number, sign)
        #         number = 0
        #         sign = '*'
        #     elif c == '/':
        #         if not stack:
        #             stack.append(result)
        #             stack.append(sign)
        #             result = number
        #         else:
        #             result = operate(result, number, sign)
        #         number = 0
        #         sign = '/'
        #
        # if number != 0:
        #     result = operate(result, number, sign)
        #     if stack:
        #         last_sign = stack.pop()
        #         last_result = stack.pop()
        #         result = operate(last_result, result, last_sign)
        # return result
        Len = len(s)
        if not s:
            return 0
        stack = []
        sign = '+'
        num = 0
        for i in range(Len):
            c = s[i]
            if c.isdigit():
                num = num * 10 + int(c)
            if not c.isdigit() and c != ' ' or i == Len - 1:
                if sign == '-':
                    stack.append(-num)
                elif sign == '+':
                    stack.append(num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                elif sign == '/':
                    stack.append(stack.pop() // num)
                sign = c
                num = 0
        ret = 0
        for i in stack:
            ret += i
        return ret

    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        273. Integer to English Words
        """
        LESS_20 = [
            "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
            "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        TENS = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        THOUSANDS = ["", "Thousand", "Million", "Billion"]

        def threeNum(n):
            ret = ''
            if n // 100 != 0:
                ret += LESS_20[n // 100] + ' Hundred'
            if n % 100 == 0:
                return ret
            if ret:
                ret += ' '
            if 0 < n % 100 < 20:
                ret += LESS_20[n % 100]
            else:
                ret += TENS[n // 10 % 10]
                if n % 10 != 0:
                    ret += ' ' + LESS_20[n % 10]
            return ret

        if num == 0:
            return 'Zero'
        thousand = 0
        ret = ''
        while num:
            if num % 1000:
                n = threeNum(num % 1000)
                n += ' ' + THOUSANDS[thousand]
                ret = n + ' ' + ret
            num //= 1000
            thousand += 1
        return ret.strip()

    def hIndex(self, citations: List[int]):
        """
        :type citations: List[int]
        :rtype: int
        274. H-Index
        """
        # 275. H-Index II
        # if not citations:
        #     return 0
        # # citations.sort()
        # Len = len(citations)
        # i = 0
        # j = Len - 1
        # while i < j:
        #     mid = (i + j) // 2
        #     if citations[mid] < Len - mid:
        #         i = mid + 1
        #     else:
        #         j = mid
        # if i == j and citations[i] < Len - i:
        #     return 0
        # return Len - j
        # 275. H - Index II
        Len = len(citations)
        if Len == 0:
            return 0
        bucket = [0] * (Len + 1)
        for i in range(Len):
            if citations[i] > Len:
                bucket[Len] += 1
            else:
                bucket[citations[i]] += 1
        t = 0
        for i in reversed(range(Len + 1)):
            t += bucket[i]
            if t >= i:
                return i
        return 0

    def addOperators(self, num, target):
        """
        :type num: str
        :type target: int
        :rtype: List[str]
        282. Expression Add Operators
        """
        Len = len(num)
        ret = []

        def dfs(expression, i, eval, multed):
            print(expression)
            if i == Len:
                if eval == target:
                    ret.append(expression)
                return
            for j in range(i + 1, Len + 1):
                if num[i] == '0' and j > i + 1:
                    break
                cur = num[i:j]
                if i == 0:
                    dfs(expression + cur, j, int(cur), int(cur))
                else:
                    dfs(expression + '+' + cur, j, eval + int(cur), int(cur))
                    dfs(expression + '-' + cur, j, eval - int(cur), -int(cur))
                    dfs(expression + '*' + cur, j, eval - multed + int(cur) * multed, int(cur) * multed)

        dfs('', 0, 0, 0)
        return ret
        # Len = len(num)
        # res = []
        #
        # def dfs(path, pos, eval, multed):
        #     if pos == Len:
        #         if eval == target:
        #             res.append(path)
        #         return
        #     for i in range(pos, Len):
        #         if i != pos and num[pos] == '0':
        #             break
        #         cur = int(num[pos:i + 1])
        #         if pos == 0:
        #             dfs(path + num[pos:i + 1], i + 1, cur, cur)
        #         else:
        #             dfs(path + '+' + num[pos:i + 1], i + 1, eval + cur, cur)
        #             dfs(path + '-' + num[pos:i + 1], i + 1, eval - cur, -cur)
        #             dfs(path + '*' + num[pos:i + 1], i + 1, eval - multed + multed * cur, multed * cur)
        #
        # dfs('', 0, 0, 0)
        # return res

    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        746. Min Cost Climbing Stairs
        """
        Len = len(cost)
        dp = [0] * Len
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, Len):
            dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
        return min(dp[-1], dp[-2])

    def shortestCompletingWord(self, licensePlate, words):
        """
        :type licensePlate: str
        :type words: List[str]
        :rtype: str
        748. Shortest Completing Word
        """
        d = {}
        for i in licensePlate:
            if i.isalpha():
                d[i.lower()] = d.get(i.lower(), 0) + 1
        ret = None
        for word in words:
            t = {}
            for i in word:
                if i.isalpha():
                    i = i.lower()
                    t[i] = t.get(i, 0) + 1
            f = 1
            for k, v in d.items():
                if k not in t or v > t[k]:
                    f = 0
                    break
            if f:
                if not ret:
                    ret = word
                elif len(ret) > len(word):
                    ret = word

        return ret

    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        301. Remove Invalid Parentheses
        """
        ans = []

        def dfs(S, lasti, lastj, par):
            stack = 0
            for i in range(lasti, len(S)):
                if S[i] == par[0]: stack += 1
                if S[i] == par[1]: stack -= 1
                if stack >= 0:
                    continue
                for j in range(lastj, i + 1):
                    if S[j] == par[1] and (j == lastj or S[j - 1] != par[1]):
                        dfs(S[:j] + S[j + 1:], i, j, par)
                return
            S2 = S[::-1]
            if par[0] == '(':
                dfs(S2, 0, 0, [')', '('])
            else:
                ans.append(S2)

        dfs(s, 0, 0, ['(', ')'])
        return ans

    def maxCoins(self, numbers: List):
        """
        :type nums: List[int]
        :rtype: int
        312. Burst Balloons
        """
        nums = [1] * (len(numbers) + 2)
        Len = 1
        for i in numbers:
            if i > 0:
                nums[Len] = i
                Len += 1
        Len += 1
        dp = [[0] * Len for _ in range(Len)]
        for l in range(2, Len):
            for i in range(Len - l):
                j = i + l
                for k in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], nums[k] * nums[i] * nums[j] + dp[i][k] + dp[k][j])
                    # dp[i][j] = max(dp[i][j], nums[k] * nums[i] * nums[j] + dp[i][k] + dp[k][j])
        return dp[0][Len - 1]

    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        315. Count of Smaller Numbers After Self
        """
        L = []

        def binarySearch(t):
            i = 0
            j = len(L)
            while i < j:
                mid = (i + j) // 2
                if t > L[mid]:
                    i = mid + 1
                else:
                    j = mid
            return j

        ret = [0] * len(nums)
        for i in reversed(range(len(nums))):
            ret[i] = binarySearch(nums[i])
            L.insert(ret[i], nums[i])
        return ret
        # L = []
        # ret = [0] * len(nums)
        # for i in reversed(range(len(nums))):
        #     ret[i] = bisect.bisect_left(L, nums[i])
        #     bisect.insort_left(L, nums[i])
        # return ret

    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        316. Remove Duplicate Letters
        """
        if not s:
            return ''
        d = {}
        for i in s:
            d[i] = d.get(i, 0) + 1
        pos = 0
        for i in range(len(s)):
            if s[pos] > s[i]:
                pos = i
            d[s[i]] -= 1
            if d[s[i]] == 0:
                break
        return s[pos] + self.removeDuplicateLetters(s[pos + 1:].replace(s[pos], ''))

    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        357. Count Numbers with Unique Digits
        """
        if n == 0:
            return 1
        available = 9
        unique = 9
        res = 10
        while n > 1:
            unique *= available
            res += unique
            available -= 1
            n -= 1
        return res

    def countRangeSum(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: int
        327. Count of Range Sum not ac tle
        """
        # Len = len(nums)
        # sums = [0] * (Len + 1)
        # for i in range(Len):
        #     sums[i + 1] = sums[i] + nums[i]
        # ans = 0
        # for i in range(Len):
        #     for j in range(i + 1, Len + 1):
        #         if lower <= sums[j] - sums[i] <= upper:
        #             ans += 1
        # return ans
        Len = len(nums)
        sums = [0] * (Len + 1)
        for i in range(Len):
            sums[i + 1] = sums[i] + nums[i]

        def sort(lo, hi):
            mid = (lo + hi) // 2
            if mid == lo:
                return 0
            count = sort(lo, mid) + sort(mid, hi)
            i = j = mid
            for left in sums[lo:mid]:
                while i < hi and sums[i] - left < lower: i += 1
                while j < hi and sums[j] - left <= upper: j += 1
                count += j - i
            sums[lo:hi] = sorted(sums[lo:hi])
            return count

        return sort(0, len(sums))

    def minPatches(self, nums, n):
        """
        :type nums: List[int]
        :type n: int
        :rtype: int
        330. Patching Array
        """
        miss = 1
        res = 0
        i = 0
        Len = len(nums)
        while miss <= n:
            if i < Len and nums[i] <= miss:
                miss += nums[i]
                i += 1
            else:
                miss += miss
                res += 1
        return res

    def maxEnvelopes(self, envelopes: List[List[int]]):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        354. Russian Doll Envelopes
        """
        Len = len(envelopes)
        if Len == 0:
            return 0
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        dp = [0] * Len
        l = 0
        for i in range(Len):
            idx = bisect.bisect_left(dp, envelopes[i][1], 0, l)
            dp[idx] = envelopes[i][1]
            if idx == l:
                l += 1
        return l

    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        300. Longest Increasing Subsequence
        """
        Len = len(nums)
        if Len == 0:
            return 0
        # nums.sort()
        dp = [0] * Len
        l = 0
        for i in nums:
            idx = bisect.bisect_left(dp, i, 0, l)
            dp[idx] = i
            if idx == l:
                l += 1
            print(dp)
        return l

    def trapRainWater(self, heightMap: List[List[int]]):
        """
        :type heightMap: List[List[int]]
        :rtype: int
        407. Trapping Rain Water II
        """
        row = len(heightMap)
        if row < 2:
            return 0
        col = len(heightMap[0])
        if col < 2:
            return 0
        h = []
        visited = [[0] * col for _ in range(row)]

        for i in range(row):
            heapq.heappush(h, [heightMap[i][0], i, 0])
            heapq.heappush(h, [heightMap[i][col - 1], i, col - 1])
            visited[i][0] = 1
            visited[i][col - 1] = 1
        for i in range(col):
            heapq.heappush(h, [heightMap[0][i], 0, i])
            heapq.heappush(h, [heightMap[row - 1][i], row - 1, i])
            visited[0][i] = 1
            visited[row - 1][i] = 1
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        res = 0
        while h:
            height, x, y = heapq.heappop(h)
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < row and 0 <= ny < col and not visited[nx][ny]:
                    visited[nx][ny] = 1
                    res += max(0, height - heightMap[nx][ny])
                    heapq.heappush(h, [max(heightMap[nx][ny], height), nx, ny])
        return res


if __name__ == '__main__':
    sol = Solution()
    heightMap = [
        [1, 4, 3, 1, 3, 2],
        [3, 2, 1, 3, 2, 4],
        [2, 3, 3, 2, 3, 1]
    ]
    print(sol.trapRainWater(heightMap))
    # print(sol.lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]))
    # print(sol.maxEnvelopes([[4, 5], [4, 6], [6, 7], [2, 3], [1, 1]]))
    # s = SummaryRanges()
    # s.addNum(1)
    # print(s.getIntervals())
    # s.addNum(3)
    # print(s.getIntervals())
    # s.addNum(7)
    # print(s.getIntervals())
    # s.addNum(2)
    # print(s.getIntervals())
    # s.addNum(6)
    # print(s.getIntervals())
    # print(sol.minPatches([1, 5, 10], 20))
    # print(sol.countNumbersWithUniqueDigits(10))
    # print(sol.countNumbersWithUniqueDigits(2))
    # print(sol.countNumbersWithUniqueDigits(3))
    # print(sol.countRangeSum([-2, 5, -1], -2, 2))
    # print(sol.countNumbersWithUniqueDigits(2))
    # print(sol.countRangeSum([-2, 5, -1], -2, 2))
    # print(sol.countRangeSum([-2, 5, 1],1,1))
    # print(sol.removeDuplicateLetters('bcabc'))
    # print(sol.maxCoins([3, 1, 5, 8]))
    # print(sol.countSmaller([5, 2, 6, 1]))
    # tfun([1, 2, 34, 5, 6, 3, 5])
    # print(sol.removeInvalidParentheses('()())()'))
    # print(sol.shortestCompletingWord("1s3 PSt", ["step", "steps", "stripe", "stepple"]))
    # print(sol.shortestCompletingWord("1s3 456", ["looks", "pest", "stew", "show"]))
    # print(sol.minCostClimbingStairs([10, 15, 20]))
    # print(sol.minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))
    # matrix = [
    #     [3, 0, 1, 4, 2],
    #     [5, 6, 3, 2, 1],
    #     [1, 2, 0, 1, 5],
    #     [4, 1, 0, 1, 7],
    #     [1, 0, 3, 0, 5]
    # ]
    # n = NumMatrix(matrix)
    # print(n.sumRegion(2, 1, 4, 3))
    # print(n.sumRegion(1, 1, 2, 2))
    # print(n.sumRegion(1, 2, 2, 4))

    # sol.addOperators("3456237490", 9191)
    # print(sol.addOperators('123', 6))
    # print(sol.addOperators('2320', 8))
    # print(sol.addOperators("3456", 45))
    # print(sol.addOperators("105", 5))
    # print(sol.hIndex([0, 1, 3, 5, 6]))
    # print(sol.hIndex([1, 2, 3, 4, 5, 6, 7]))
    # print(sol.hIndex([0]))
    # lines = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
    # print(sol.getSkyline(lines))
    # expression = '1+2+3*2/4+5-3'
    # print(sol.calculate2(expression), eval(expression.replace('/', '//')))
    # print(sol.numberToWords(1000))
