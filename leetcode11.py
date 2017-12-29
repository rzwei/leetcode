import bisect
import collections
import heapq
from typing import List


class Trie:
    def __init__(self, word):
        self.word = word
        self.isword = False
        self.next = {}

    def __repr__(self):
        return f'{self.word},{self.isword}'


class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b


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

    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        395. Longest Substring with At Least K Repeating Characters
        """

        def dv(start, end):
            if end - start < k:
                return 0
            d = [0] * 26
            for i in s[start:end]:
                d[ord(i) - ord('a')] += 1
            for i in range(start, end):
                if 0 < d[ord(s[i]) - ord('a')] < k:
                    return max(dv(start, i), dv(i + 1, end))
            return end - start

        return dv(0, len(s))

    def canIWin(self, maxChoosableInteger, desiredTotal):
        """
        :type maxChoosableInteger: int
        :type desiredTotal: int
        :rtype: bool
        464. Can I Win
        """
        if (1 + maxChoosableInteger) * maxChoosableInteger / 2 < desiredTotal:
            return False
        cache = {}

        def dfs(nums, desire):
            k = str(nums)
            if k in cache:
                return cache[k]
            if nums[-1] >= desire:
                return True
            for i in range(len(nums)):
                if not dfs(nums[0:i] + nums[i + 1:], desire - nums[i]):
                    cache[k] = True
                    return True
            cache[k] = False
            return False

        return dfs(list(range(1, maxChoosableInteger + 1)), desiredTotal)

    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        747. Largest Number Greater Than Twice of Others
        """
        idx = 0
        for i, v in enumerate(nums):
            if v > nums[idx]:
                idx = i
        for i, v in enumerate(nums):
            if i == idx or v == 0:
                continue
            if nums[idx] // v < 2:
                return -1
        return idx

    def ipToCIDR(self, ip, n):
        """
        :type ip: str
        :type n: int
        :rtype: List[str]
        751. IP to CIDR
        """

        def ip2number(ip):
            numbers = list(map(int, ip.split(".")))
            n = (numbers[0] << 24) + (numbers[1] << 16) + (numbers[2] << 8) + numbers[3]
            return n

        def number2ip(n):
            return ".".join([str(n >> 24 & 255), str(n >> 16 & 255), str(n >> 8 & 255), str(n & 255)])

        def ilowbit(x):
            for i in range(32):
                if x & (1 << i):
                    return i

        def lowbit(x):
            return 1 << ilowbit(x)

        number = ip2number(ip)
        result = []
        while n > 0:
            lb = lowbit(number)
            while lb > n:
                lb = lb // 2

            n = n - lb

            result.append(number2ip(number) + "/" + str(32 - ilowbit(lb)))
            number = number + lb
        return result

        # def ipToCIDR(self, ip, n):
        # tokens = ip.split('.')
        # num = 0
        # s = 1
        # for token in reversed(tokens):
        #     num += int(token) * s
        #     s <<= 8
        #
        # def fun(ip_num):
        #     ret = []
        #     mask = 0xff
        #     for i in range(4):
        #         ret.append(str(ip_num & mask))
        #         ip_num >>= 8
        #     return '.'.join(reversed(ret))
        #
        # upper = num + n
        # cur = upper - 1
        # ret = []
        # while cur >= num:
        #     m = 0
        #     k = 0
        #     t = cur
        #     while t & 1 == 0:
        #         m = (m << 1) | 1
        #         k += 1
        #         t >>= 1
        #     if cur + m < upper:
        #         ret.append("%s/%d" % (fun(cur), 32 - k))
        #         cur -= m + 1
        #     else:
        #         ret.append("%s/%d" % (fun(cur), 32))
        #         cur -= 1
        # return ret

    def openLock(self, deadends, target):
        """
        :type deadends: List[str]
        :type target: str
        :rtype: int
        752. Open the Lock
        """

        def successors(src):
            res = []
            for i, ch in enumerate(src):
                num = int(ch)
                res.append(src[:i] + str(9 if num == 0 else num - 1) + src[i + 1:])
                res.append(src[:i] + str(0 if num == 9 else num + 1) + src[i + 1:])
            return res

        q = collections.deque(['0000'])
        dep = 0
        visited = set()
        for dead in deadends:
            visited.add(dead)
        while q:
            size = len(q)
            for i in range(size):
                node = q.popleft()
                if node == target:
                    return dep
                if node in visited:
                    continue
                visited.add(node)
                q.extend(successors(node))
            dep += 1
        return -1

    def openLock_(self, deadends, target):
        '''
        :param deadends: List[str]
        :param target: int
        :return: int
        752. Open the Lock double bfs
        '''
        if target == '0000':
            return 0
        if '0000' in deadends:
            return -1
        visited = set()
        for i in deadends:
            visited.add(i)
        leftSet = {'0000'}
        rightSet = {target}
        res = 1
        while leftSet and rightSet:
            if len(leftSet) > len(rightSet):
                leftSet, rightSet = rightSet, leftSet
            # print(leftSet, rightSet)
            temp = leftSet
            L = set()
            for one in temp:
                visited.add(one)
                tokens = list(one)
                for i in range(4):
                    u = chr((ord(tokens[i]) - ord('0') + 1) % 10 + ord('0'))
                    l = chr((ord(tokens[i]) - ord('0') - 1) % 10 + ord('0'))
                    old = tokens[i]

                    tokens[i] = u

                    k = ''.join(tokens)
                    if k not in visited:
                        if k in rightSet:
                            return res
                        L.add(k)

                    tokens[i] = l

                    k = ''.join(tokens)
                    if k not in visited:
                        if k in rightSet:
                            return res
                        L.add(k)

                    tokens[i] = old
            leftSet = L
            res += 1
        return -1

    # def openLock(self, deadends, target):
    #     marker, depth = 'x', 0
    #     visited, q, deadends = set(), collections.deque(['0000', marker]), set(deadends)
    #
    #     def successors(src):
    #         res = []
    #         for i, ch in enumerate(src):
    #             num = int(ch)
    #             res.append(src[:i] + str(9 if num == 0 else num - 1) + src[i + 1:])
    #             res.append(src[:i] + str(0 if num == 9 else num + 1) + src[i + 1:])
    #         return res
    #
    #     while q:
    #         node = q.popleft()
    #         if node == target:
    #             return depth
    #         if node in visited or node in deadends:
    #             continue
    #         if node == marker and not q:
    #             return -1
    #         if node == marker:
    #             q.append(marker)
    #             depth += 1
    #         else:
    #             visited.add(node)
    #             q.extend(successors(node))
    #     return -1
    def crackSafe(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        753. Cracking the Safe
        """
        target = 1
        for i in range(n):
            target *= k

        def dfs(nums, k, visited):
            if len(visited) == target:
                return nums
            visited.add(''.join(nums[-n:]))
            for i in range(k):
                nums.append(str(i))
                tk = ''.join(nums[-n:])
                if tk not in visited:
                    visited.add(tk)
                    r = dfs(nums, k, visited)
                    if r:
                        return r
                    visited.remove(tk)
                nums.pop()
            return None

        r = dfs(['0'] * n, k, set())
        return ''.join(r)

    def findAllConcatenatedWordsInADict(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        472. Concatenated Words
        """
        # root = Trie(-1)
        # for word in words:
        #     cur = root
        #     for i in word:
        #         if i not in cur.next:
        #             cur.next[i] = Trie(i)
        #         cur = cur.next[i]
        #     cur.isword = True
        #
        # def dfs(word, cur: Trie, wi, n):
        #     if wi == len(word) - 1 and cur.isword and n > 1:
        #         return True
        #
        #     for i in range(wi, len(word)):
        #         if cur.isword:
        #             if i == len(word) - 1 and n > 1:
        #                 return True
        #             if i + 1 < len(word) and word[i + 1] in root.next:
        #                 if dfs(word, root.next[word[i + 1]], i + 1, n + 1):
        #                     return True
        #         if i + 1 >= len(word):
        #             break
        #         if word[i + 1] in cur.next:
        #             cur = cur.next[word[i + 1]]
        #         else:
        #             return False
        #
        #     return False
        #
        # ret = []
        # for word in words:
        #     if not word:
        #         continue
        #     if dfs(word, root.next[word[0]], 0, 1):
        #         ret.append(word)
        # return ret
        # d = {}
        # for word in words:
        #     L = len(word)
        #     if L not in d:
        #         d[L] = {word}
        #     else:
        #         d[L].add(word)
        #
        # def dfs(word, wi, n):
        #     print(word, wi, n)
        #     if wi == len(word) and n > 2:
        #         return True
        #     for i in range(wi + 1, len(word) + 1):
        #         L = i - wi
        #         if L in d and word[wi:i] in d[L]:
        #             if dfs(word, i, n + 1):
        #                 return True
        #     return False
        #
        # ret = []
        # for word in words:
        #     if dfs(word, 0, 1):
        #         ret.append(word)
        # return ret
        res = []
        words_set = set(words)

        def check(word):
            if word in words_set:
                return True
            for i in reversed(range(1, len(word) + 1)):
                if word[:i] in words_set and check(word[i:]):
                    return True
            return False

        for word in words:
            words_set.remove(word)
            if check(word):
                res.append(word)
            words_set.add(word)

        return res

    def canCross(self, stones: List[int]):
        """
        :type stones: List[int]
        :rtype: bool
        403. Frog Jump
        """
        Len = len(stones)
        idx = {}
        for i, v in enumerate(stones):
            idx[v] = i
        memo = set()
        if stones[1] - stones[0] > 1:
            return False

        def dp(i, k):
            print(i, k)
            if (i, k) in memo:
                return False
            if i == Len - 1:
                return True
            for di in range(-1, 2):
                if di + k <= 0:
                    continue
                step = di + k
                if stones[i] + step in idx and dp(idx[stones[i] + step], step):
                    return True
            memo.add((i, k))
            return False

        return dp(0, 1)

    def arrayNesting(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        565. Array Nesting
        """
        res = 0
        vis = [False] * (len(nums) + 1)
        for i in nums:
            if vis[i]:
                continue
            cur = i
            n = 0
            while not vis[cur]:
                n += 1
                vis[cur] = True
                cur = nums[cur]
            res = max(res, n)
        return res

    def findPaths(self, m, n, N, srci, srcj):
        """
        :type m: int
        :type n: int
        :type N: int
        :type i: int
        :type j: int
        :rtype: int
        576. Out of Boundary Paths
        """

        MOD = 10 ** 9 + 7
        count = [[0] * n for _ in range(m)]
        count[srci][srcj] = 1
        res = 0
        for i in range(N):
            temp = [[0] * n for _ in range(m)]
            for i in range(m):
                for j in range(n):
                    for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                        if x < 0 or x >= m or y < 0 or y >= n:
                            res = (res + count[i][j]) % MOD
                        else:
                            temp[x][y] = (temp[x][y] + count[i][j]) % MOD
            count = temp
        return res

    def checkRecord(self, n):
        """
        :type n: int
        :rtype: int
        552. Student Attendance Record II
        """
        if n == 1:
            return 3
        if n == 2:
            return 8
        MOD = 10 ** 9 + 7
        A = [0] * n
        P = [0] * n
        L = [0] * n

        P[0] = 1

        L[0] = 1
        L[1] = 3

        A[0] = 1
        A[1] = 2
        A[2] = 4
        for i in range(1, n):
            A[i - 1] %= MOD
            P[i - 1] %= MOD
            L[i - 1] %= MOD
            P[i] = A[i - 1] + P[i - 1] + L[i - 1]
            if i > 1:
                L[i] = A[i - 1] + P[i - 1] + A[i - 2] + P[i - 2]
            if i > 2:
                A[i] = A[i - 1] + A[i - 2] + A[i - 3]
        return (A[n - 1] + P[n - 1] + L[n - 1]) % MOD

    def outerTrees(self, points: List[Point]):
        """
        :type points: List[Point]
        :rtype: List[Point]
        587. Erect the Fence
        """
        """Computes the convex hull of a set of 2D points.

        Input: an iterable sequence of (x, y) pairs representing the points.
        Output: a list of vertices of the convex hull in counter-clockwise order,
          starting from the vertex with the lexicographically smallest coordinates.
        Implements Andrew's monotone chain algorithm. O(n log n) complexity.
        """

        # Sort the points lexicographically (tuples are compared lexicographically).
        # Remove duplicates to detect the case we have just one unique point.
        # points = sorted(set(points))
        points = sorted(points, key=lambda p: (p.x, p.y))

        # Boring case: no points or a single point, possibly repeated multiple times.
        if len(points) <= 1:
            return points

        # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
        # Returns a positive value, if OAB makes a counter-clockwise turn,
        # negative for clockwise turn, and zero if the points are collinear.
        def cross(o, a, b):
            # return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
                upper.pop()
            upper.append(p)

        # Concatenation of the lower and upper hulls gives the convex hull.
        # Last point of each list is omitted because it is repeated at the
        # beginning of the other list.
        # return lower[:-1] + upper[:-1]
        return list(set(lower[:-1] + upper[:-1]))

    def findIntegers(self, num):
        """
        :type num: int
        :rtype: int
        600. Non-negative Integers without Consecutive Ones
        """
        P0 = [0] * 34
        P1 = [0] * 34
        P0[0] = 1
        P1[0] = 1
        for i in range(1, 34):
            P1[i] = P0[i - 1]
            P0[i] = P1[i - 1] + P0[i - 1]
        n = 0
        t = num
        while t:
            t >>= 1
            n += 1
        res = P0[n - 1] + P1[n - 1]
        for i in reversed(range(n - 1)):
            if (num >> i) & 1 and (num >> (i + 1)) & 1:
                break
            if (num >> i) & 1 == 0 and (num >> (i + 1)) & 1 == 0:
                res -= P1[i]

        return res

    def findDuplicate(self, paths: List[str]):
        """
        :type paths: List[str]
        :rtype: List[List[str]]
        609. Find Duplicate File in System
        """
        d = {}
        for one in paths:
            tokens = one.split(' ')
            path = tokens[0]
            for file in tokens[1:]:
                idx = file.index('(')
                name = file[:idx]
                content = file[idx:-1]
                if content not in d:
                    d[content] = [path + '/' + name]
                else:
                    d[content].append(path + '/' + name)
        ret = []
        for v in d.values():
            if len(v) > 1:
                ret.append(v)
        return ret

    def checkSubarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        523. Continuous Subarray Sum
        """
        # Len = len(nums)
        # if Len == 0:
        #     return False
        #
        # if k == 0:
        #     if Len <= 1:
        #         return False
        #     for i in nums:
        #         if i != 0:
        #             return False
        #     return True
        #
        # sums = [0] * (Len + 1)
        # d = {0: -1}
        # for i in range(Len):
        #     sums[i + 1] = (sums[i] + nums[i]) % k
        #     v = sums[i + 1]
        #     if v not in d:
        #         d[v] = i + 1
        #     else:
        #         if i + 1 - d[v] >= 2:
        #             return True
        # return False

        # dict will be {modulus, key}
        dict = {0: -1}
        cumSum = 0

        for i in range(len(nums)):
            cumSum += nums[i]
            if k != 0:
                cumSum = cumSum % k

            if cumSum not in dict:
                dict[cumSum] = i
            elif i - dict[cumSum] >= 2:
                return True

        return False

    def findMinMoves(self, machines):
        """
        :type machines: List[int]
        :rtype: int
        517. Super Washing Machines
        """
        Len = len(machines)
        s = sum(machines)
        if s % Len != 0:
            return -1
        res = 0
        avg = s // Len
        cnt = 0
        for i in machines:
            cnt += i - avg
            res = max(res, abs(cnt), i - avg)
        return res

    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        493. Reverse Pairs
        """
        prev = []
        ans = 0
        for i in nums:
            ans += len(prev) - bisect.bisect_right(prev, i + i)
            bisect.insort_left(prev, i)
        return ans


if __name__ == '__main__':
    sol = Solution()
    print(sol.reversePairs([1, 3, 2, 3, 1]))
    print(sol.reversePairs([1, 1, 1, 1, 1, 1]))
    print(sol.reversePairs([2, 4, 3, 5, 1]))
    # print(sol.findMinMoves([1, 0, 5]))
    # print(sol.checkSubarraySum([0, 0], -1))
    # print(sol.checkSubarraySum([0], 0))
    # print(sol.checkSubarraySum([23, 2, 4, 6, 7], 6))
    # print(sol.checkSubarraySum([23, 2, 6, 4, 7], 6))
    # print(sol.findDuplicate(
    #     ["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]))
    # print(sol.findIntegers(5))
    # print(sol.checkRecord(93573))
    # print(sol.findPaths(2, 2, 2, 0, 0))
    # print(sol.arrayNesting([5, 4, 0, 3, 1, 6, 2]))
    # print(sol.removeBoxes([1, 2, 1]))
    # print(sol.removeBoxes([1, 1, 2, 2, 2, 3, 4, 3, 1]))
    # print(sol.canCross([0, 2]))
    # print(sol.canCross([0, 1, 2, 3, 4, 8, 9, 11]))
    # print(sol.canCross([0, 1, 3, 5, 6, 8, 12, 17]))
    # print(sol.crackSafe(3, 8))
    # print(sol.crackSafe(3, 6))
    # print(sol.crackSafe(2, 10))
    # print(sol.crackSafe(1, 2))
    # print(sol.crackSafe(2, 2))
    # print(sol.findAllConcatenatedWordsInADict(['a', 'aaaa']))
    # print(sol.findAllConcatenatedWordsInADict([""]))
    # print(sol.findAllConcatenatedWordsInADict(
    #     ["cat", "cats", "catsdogcats", "dog", "dogcatsdog", "hippopotamuses", "rat", "ratcatdogcat"]))
    # print(sol.ipToCIDR("255.0.0.7", 10))
    # print(sol.openLock_(["0201", "0101", "0102", "1212", "2002"], '0202'))
    # print(sol.openLock_([], '5555'))
    # print(sol.openLock_(["1002", "1220", "0122", "0112", "0121"], "1200"))
    # print(sol.openLock_(["1002", "1220", "0122", "0112", "0121"], "0000"))
    # print(sol.openLock_(["8887", "8889", "8878", "8898", "8788", "8988", "7888", "9888"], "8888"))
    # print(sol.openLock_(["0000"], "8888"))
    # print(sol.openLock_(["0201", "0101", "0102", "1212", "2002"], '0202'))
    # print(sol.openLock_(["1002", "1220", "0122", "0112", "0121"], "1200"))
    # print(sol.openLock_(["1002", "1220", "0122", "0112", "0121"], "0000"))
    # print(sol.openLock_(["8887", "8889", "8878", "8898", "8788", "8988", "7888", "9888"], "8888"))
    # print(sol.openLock_(["0000"], "8888"))
    # sol.ipToCIDR('0.0.0.0', 10)
    # print(sol.dominantIndex([3, 6, 1, 0]))
    # print(sol.dominantIndex([1, 2, 3, 4]))
    # print(sol.canIWin(10, 11))
    # matrix = [
    #     [1, 0, 1],
    #     [0, -2, 3]
    # ]
    # print(sol.maxSumSubmatrix(matrix, 2))
    # print(sol.maxSumSubmatrix([[2, 2, -1]], 0))
    # print(sol.lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]))
    # print(sol.longestSubstring("bbaaacbd", 3))
    # print(sol.longestSubstring('ababacb', 3))
    # print(sol.longestSubstring('aaabb', 3))
    # print(sol.longestSubstring('ababbc', 2))
    # print(sol.longestSubstring('aaabbb', 3))
    # heightMap = [
    #     [1, 4, 3, 1, 3, 2],
    #     [3, 2, 1, 3, 2, 4],
    #     [2, 3, 3, 2, 3, 1]
    # ]
    # print(sol.trapRainWater(heightMap))
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
