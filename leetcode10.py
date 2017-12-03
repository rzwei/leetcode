import bisect


# class TreeNode:
#     def __init__(self, s, e):
#         self.start = s
#         self.end = e
#         self.right = None
#         self.left = None
#
#
# class MyCalendar:
#     def __init__(self):
#         self.times = {}
#         self.books = TreeNode(-1, -1)
#
#     def search(self, p, s, e):
#         if p.start >= e:
#             if not p.left:
#                 p.left = TreeNode(s, e)
#                 return True
#             else:
#                 return self.search(p.left, s, e)
#         if p.end <= s:
#             if not p.right:
#                 p.right = TreeNode(s, e)
#                 return True
#             else:
#                 return self.search(p.right, s, e)
#         return False
#
#     def book(self, start, end):
#         """
#         :type start: int
#         :type end: int
#         :rtype: bool
#         """
#         return self.search(self.books, start, end)
class MyCalendar:
    def __init__(self):
        self.start = []
        self.end = []

    def book(self, start, end):
        i = bisect.bisect_right(self.end, start)
        j = bisect.bisect_left(self.start, end)
        if i == j:
            self.start.index(i, start)
            self.end.index(i, end)
            return True
        return False


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


if __name__ == '__main__':
    sol = Solution()
    # print(sol.areSentencesSimilar(['great', 'acting', 'skills'], ['fine', 'drama', 'talent'],
    #                               [["great", "fine"], ["acting", "drama"], ["skills", "talent"]]))
    calendar = MyCalendar()
    for v1, v2 in [[97, 100], [33, 51], [89, 100], [83, 100], [75, 92], [76, 95], [19, 30], [53, 63], [8, 23], [18, 37],
                   [87, 100], [83, 100], [54, 67], [35, 48], [58, 75], [70, 89], [13, 32], [44, 63], [51, 62], [2, 15]]:
        print(calendar.book(v1, v2))
    # print(calendar.times.items())
