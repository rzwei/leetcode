from typing import List
import bisect
import collections
import heapq


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
        Len = len(needs)
        if not any(needs):
            return 0
        mini = []
        minv = -1

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
                minv = 1
                mini.append(spi)
        res = 0
        for i in range(Len):
            res += price[i] * needs[i]

        if minv != -1:
            for i in mini:
                for j in range(Len):
                    needs[j] -= special[i][j]
                res = min(res, self.shoppingOffers(price, special, needs) + special[i][-1])
                for j in range(Len):
                    needs[j] += special[i][j]
            return res
        return res


if __name__ == '__main__':
    sol = Solution()
    # print(sol.shoppingOffers([2, 5], [[3, 0, 5], [1, 2, 10]], [3, 2]))
    print(sol.shoppingOffers([2, 3, 4], [[1, 1, 0, 4], [2, 2, 1, 9]], [1, 2, 1]))
    # print(sol.findMaximizedCapital(2, 0, [1, 2, 3], [0, 1, 1]))
    # nums = [[1, 3], [1, 4], [2, 5], [3, 5]]
    # nums = [[2, 10], [3, 7], [3, 15], [4, 11], [6, 12], [6, 16], [7, 8], [7, 11], [7, 15], [11, 12]]
    # print(sol.intersectionSizeTwo(nums))
