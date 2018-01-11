from typing import List
import bisect
import collections
import heapq


class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

    def __repr__(self):
        ''.format()
        return f'[{self.start} {self.end}]'


def buildIntervals(inters: List[List[int]]):
    ans = []
    for s, e in inters:
        ans.append(Interval(s, e))
    return ans

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

#284. Peeking Iterator
class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.cur = None
        self.hasnext = False
        if iterator.hasNext():
            self.hasnext = True
            self.cur = iterator.next()

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.cur

    def next(self):
        """
        :rtype: int
        """
        t = self.cur
        self.hasnext = self.iterator.hasNext()
        self.cur = self.iterator.next()
        return t

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.hasnext

class Solution:
    def intersectionSizeTwo(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        759. Set Intersection Size At Least Two
        """
        intervals.sort(key=lambda x: (x[1], x[0]))
        s = intervals[0][1]
        e = intervals[0][1]
        for interval in intervals:
            if e <= interval[0]:
                e = min(interval[1], interval[0] + 1)
            elif s >= interval[1]:
                s = max(interval[0], interval[1] - 1)
        return e - s + 1

    def findMaximizedCapital(self, k, W, Profits: List, Capital: List):
        """
        :type k: int
        :type W: int
        :type Profits: List[int]
        :type Capital: List[int]
        :rtype: int
        502. IPO
        """
        cap = [[i, j] for i, j in zip(Capital, Profits)]
        pro = []
        heapq.heapify(cap)
        ans = W
        for _ in range(k):
            while cap and cap[0][0] <= ans:
                v = heapq.heappop(cap)
                heapq.heappush(pro, [-v[1], v[0]])
            if not pro:
                break
            v = heapq.heappop(pro)
            ans += -v[0]
        return ans

    def shoppingOffers(self, price: List, special: List[List[int]], needs: List):
        """
        :type price: List[int]
        :type special: List[List[int]]
        :type needs: List[int]
        :rtype: int
        638. Shopping Offers
        """
        d = {}

        def dfs(needs):
            if tuple(needs) in d:
                return d[tuple(needs)]

            Len = len(needs)
            if not any(needs):
                return 0
            mini = []

            for spi, sp in enumerate(special):
                f = 1

                for i in range(Len):
                    if sp[i] <= needs[i]:
                        continue
                    f = 0
                    break
                if f:
                    t = 0
                    for i in range(Len):
                        t += sp[i] * price[i]
                    if t < sp[-1]:
                        continue
                    mini.append(spi)
            res = 0
            for i in range(Len):
                res += price[i] * needs[i]

            for i in mini:
                for j in range(Len):
                    needs[j] -= special[i][j]
                res = min(res, self.shoppingOffers(price, special, needs) + special[i][-1])
                for j in range(Len):
                    needs[j] += special[i][j]
            d[tuple(needs)] = res
            return res

        return dfs(needs)

    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        714. Best Time to Buy and Sell Stock with Transaction Fee
        """
        h = []
        l = []
        Len = len(prices)
        dp = [0] * (Len)
        for i in range(Len):
            if i > 0:
                dp[i] = dp[i - 1]
            while h and prices[i] - h[0][0] > fee:
                v, idx = heapq.heappop(h)
                dp[i] = max(dp[i], dp[idx] + prices[i] - v - fee)
                l.append((v, idx))
            for v, idx in l:
                heapq.heappush(h, (v, idx))
            heapq.heappush(h, (prices[i], i))
        return dp[-1]

    def anagramMappings(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        760. Find Anagram Mappings
        """
        d = {}
        for i, v in enumerate(B):
            if v not in d:
                d[v] = [i]
            else:
                d[v].append(i)
        ans = []
        for i in A:
            ans.append(d[i].pop())

        return ans

    def boldWords(self, words, S):
        """
        :type words: List[str]
        :type S: str
        :rtype: str
        758. Bold Words in String
        """
        Len = len(S)
        hit = [0] * (Len + 1)
        for word in words:
            for i in range(Len - len(word) + 1):
                if S[i:i + len(word)] == word:
                    hit[i] += 1
                    hit[i + len(word)] -= 1
        for i in range(Len):
            hit[i + 1] += hit[i]
        ans = ''
        i = 0
        while i < Len:
            if hit[i] > 0:
                ans += '<b>'
                while i < Len and hit[i] > 0:
                    ans += S[i]
                    i += 1
                i -= 1
                ans += '</b>'
            else:
                ans += S[i]
            i += 1
        return ans

    def employeeFreeTime(self, avails: List[List[Interval]]):
        """
        :type avails: List[List[Interval]]
        :rtype: List[Interval]
        759. Employee Free Time
        """
        if not avails:
            return []
        intervals = []
        for i in avails:
            intervals.extend(i)
        intervals.sort(key=lambda x: (x.start, x.end))
        prev = intervals[0].end
        ans = []
        for inter in intervals:
            if prev >= inter.end:
                continue
            if inter.start > prev:
                ans.append(Interval(prev, inter.start))
            prev = inter.end
        return ans


if __name__ == '__main__':
    sol = Solution()
    r = buildIntervals([[1, 2], [5, 6], [1, 3], [4, 10]])
    # r = buildIntervals([[1, 3], [6, 7], [2, 4], [2, 5], [9, 12]])
    print(sol.employeeFreeTime([r]))
    # print(sol.boldWords(['ab', 'bc'], "aabc"))
    # print(sol.maxProfit([1, 3, 2, 8, 4, 9], fee=2))
    # print(sol.shoppingOffers([2, 5], [[3, 0, 5], [1, 2, 10]], [3, 2]))
    # print(sol.shoppingOffers([2, 3, 4], [[1, 1, 0, 4], [2, 2, 1, 9]], [1, 2, 1]))
    # print(sol.findMaximizedCapital(2, 0, [1, 2, 3], [0, 1, 1]))
    # nums = [[1, 3], [1, 4], [2, 5], [3, 5]]
    # nums = [[2, 10], [3, 7], [3, 15], [4, 11], [6, 12], [6, 16], [7, 8], [7, 11], [7, 15], [11, 12]]
    # print(sol.intersectionSizeTwo(nums))
