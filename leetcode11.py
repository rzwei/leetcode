import heapq


class Solution:
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        218. The Skyline Problem
        """
        h = []
        for li, ri, hi in buildings:
            heapq.heappush(h, (li, hi, ri))
        ret = []
        last_l = 0
        last_h = 0
        last_r = 0
        while h:
            l, h, r = heapq.heappop(h)




if __name__ == '__main__':
    pass
