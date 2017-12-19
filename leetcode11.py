import heapq
from typing import List


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


if __name__ == '__main__':
    sol = Solution()
    print(sol.maxCoins([3, 1, 5, 8]))
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
