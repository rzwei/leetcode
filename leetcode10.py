import heapq


class TreeNode:
    def __init__(self, s, e):
        self.start = s
        self.end = e
        self.right = None
        self.left = None


class MyCalendar:
    def __init__(self):
        self.times = {}
        self.books = TreeNode(-1, -1)

    def search(self, p, s, e):
        if p.start >= e:
            if not p.left:
                p.left = TreeNode(s, e)
                return True
            else:
                return self.search(p.left, s, e)
        if p.end <= s:
            if not p.right:
                p.right = TreeNode(s, e)
                return True
            else:
                return self.search(p.right, s, e)
        return False

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        return self.search(self.books, start, end)


# class MyCalendar:
#     def __init__(self):
#         self.start = []
#         self.end = []
#
#     def book(self, start, end):
#         i = bisect.bisect_right(self.end, start)
#         j = bisect.bisect_left(self.start, end)
#         if i == j:
#             self.start.index(i, start)
#             self.end.index(i, end)
#             return True
#         return False


class Solution:
    def areSentencesSimilar(self, words1, words2, pairs):
        """
        :type words1: List[str]
        :type words2: List[str]
        :type pairs: List[List[str]]
        :rtype: bool
        """
        # 734. Sentence Similarity
        if len(words1) != len(words2):
            return False
        d = {}
        for l, r in pairs:
            if l not in d:
                d[l] = {r}
            else:
                d[l].add(r)
            if r not in d:
                d[r] = {l}
            else:
                d[r].add(l)
        for w1, w2 in zip(words1, words2):
            if w1 == w2 or w1 in d.get(w2, []):
                continue
            return False
        return True

    def areSentencesSimilarTwo(self, words1, words2, pairs):
        """
        :type words1: List[str]
        :type words2: List[str]
        :type pairs: List[List[str]]
        :rtype: bool
        """
        if len(words1) != len(words2):
            return False
        d = {}

        def father(x):
            if x != d[x]:
                return father(d[x])
            else:
                return x

        def union(x, y):
            x = father(x)
            y = father(y)
            d[x] = y

        for w1, w2 in pairs:
            if w1 not in d:
                d[w1] = w1
            if w2 not in d:
                d[w2] = w2
            union(w1, w2)
        for w1, w2 in zip(words1, words2):
            if w1 == w2:
                continue
            if w1 not in d or w2 not in d:
                return False
            if father(w1) == father(w2):
                continue
            print(w1, w2)
            return False
        return True

    def smallestRange(self, nums):
        """
        :type nums: List[List[int]]
        :rtype: List[int]
        """
        # 632. Smallest Range
        pq = [(row[0], i, 0) for i, row in enumerate(nums)]
        heapq.heapify(pq)

        right = max(row[0] for row in nums)

        ans = -1e9, 1e9

        while pq:
            left, i, j = heapq.heappop(pq)
            if right - left < ans[1] - ans[0]:
                ans = left, right
            if j + 1 == len(nums[i]):
                return ans
            v = nums[i][j + 1]
            right = max(right, v)
            heapq.heappush(pq, (v, i, j + 1))

    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 697. Degree of an Array
        s, t = 0, 0
        Len = len(nums)
        d = {}
        Level = -1
        for i in nums:
            d[i] = d.get(i, 0) + 1
            if d[i] > Level:
                Level = d[i]

        d.clear()
        ret = Len

        def level():
            if d.values():
                return max(d.values())
            return 0

        while t < Len:
            if level() < Level:
                d[nums[t]] = d.get(nums[t], 0) + 1
            if level() == Level:
                while level() == Level:
                    ret = min(ret, t - s + 1)
                    d[nums[s]] -= 1
                    s += 1
            t += 1

        return ret


if __name__ == '__main__':
    sol = Solution()
    # print(sol.areSentencesSimilarTwo(["great", "acting", "skills"],
    #                                  ["fine", "painting", "talent"],
    #                                  [["great", "fine"], ["drama", "acting"], ["skills", "talent"]]))
    print(sol.findShortestSubArray([1, 2, 2, 3, 1]))
    print(sol.findShortestSubArray([1, 2, 2, 3, 1, 4, 2]))
