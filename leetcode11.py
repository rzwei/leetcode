import heapq


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


if __name__ == '__main__':
    sol = Solution()
    # lines = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
    # print(sol.getSkyline(lines))
    # expression = '1+2+3*2/4+5-3'
    # print(sol.calculate2(expression), eval(expression.replace('/', '//')))
    print(sol.numberToWords(1000))
