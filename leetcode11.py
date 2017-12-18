import heapq
from typing import List


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

        def calc(expression):
            stack = []
            number = 0
            sign = '+'
            for i, c in enumerate(expression):
                if c.isdigit():
                    number = number * 10 + int(c)
                if not c.isdigit() or i == len(expression) - 1:
                    if sign == '+':
                        stack.append(number)
                    elif sign == '-':
                        stack.append(-number)
                    elif sign == '*':
                        stack.append(stack.pop() * number)
                    sign = c
                    number = 0
            return sum(stack)

        def dfs(expression, i, f):
            print(expression)

            if f==1:
                for j in range()

            if i == Len:
                if calc(expression) == target:
                    print(expression)
                return
            for c in '+-*':
                for j in reversed(range(i + 1, Len)):
                    cur = expression + c + num[i:j]
                    dfs(cur, j)



if __name__ == '__main__':
    sol = Solution()
    sol.addOperators('123', 6)
    # print(sol.hIndex([0, 1, 3, 5, 6]))
    # print(sol.hIndex([1, 2, 3, 4, 5, 6, 7]))
    # print(sol.hIndex([0]))
    # lines = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
    # print(sol.getSkyline(lines))
    # expression = '1+2+3*2/4+5-3'
    # print(sol.calculate2(expression), eval(expression.replace('/', '//')))
    # print(sol.numberToWords(1000))
