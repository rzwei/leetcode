import random


class canFinish4:
    def dfs(self, u, visited: list, mat: list):
        if visited[u] == -1:
            return False
        if visited[u] == 1:
            return 1

        visited[u] = -1
        for j in mat[u]:
            if not self.dfs(j, visited, mat):
                return False
        visited[u] = 1
        return True

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        visit = [0 for i in range(numCourses)]
        mat = [[] for i in range(numCourses)]

        for p in prerequisites:
            mat[p[0]].append(p[1])

        for i in range(numCourses):
            if not self.dfs(i, visit, mat):
                return False

        return True


class Tweet:
    def __init__(self):
        self.tweetid = 0
        self.tweet_index = 0

    def getIndex(self):
        return self.tweet_index


class Twitter(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.tweets_index = 0
        self.twitter = {}
        self.tweets = {}

    def check(self, userid):
        if userid in self.twitter:
            return
        self.twitter[userid] = {'tweets': [], 'follow': set()}

    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        tweet = Tweet()
        tweet.tweetid = tweetId
        tweet.tweet_index = self.tweets_index

        self.tweets_index += 1

        self.check(userId)
        self.twitter[userId]['tweets'].append(tweet)

    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        self.check(userId)

        ret = self.twitter[userId]['tweets'][:]
        # ret = []
        for i in self.twitter[userId]['follow']:
            self.check(i)
            ret.extend(self.twitter[i]['tweets'])

        ret = sorted(ret, key=lambda x: -x.getIndex())
        rrr = []
        for i in ret[:10]:
            rrr.append(i.tweetid)
        print(rrr)
        return rrr

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId == followeeId:
            return
        self.check(followerId)
        self.check(followeeId)
        self.twitter[followerId]['follow'].add(followeeId)

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.check(followerId)
        if followeeId in self.twitter[followerId]['follow']:
            self.twitter[followerId]['follow'].remove(followeeId)


# Definition for a  binary tree node
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.cur = root

    def hasNext(self):
        """
        :rtype: bool
        """
        return True if self.cur is not None else False

    def next(self):
        """
        :rtype: int
        """
        if self.cur.left is not None:
            return self


class canFinish3:
    def dfs(self, u, notvisited: set, mat: list, path: list):
        pass

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        visited = set()
        mat = [[] for i in range(numCourses)]

        inv = [0 for i in range(numCourses)]
        for p in prerequisites:
            mat[p[0]].append(p[1])
            inv[p[1]] += 1
        queue = []
        for i, v in enumerate(inv):
            if v == 0:
                queue.append(i)
        ret = []

        while len(queue) != 0:
            u = queue.pop()
            ret.append(u)
            print(u)
            visited.add(u)
            for i, vi in enumerate(mat[u]):
                # if i in visited:
                inv[vi] -= 1
                if inv[vi] == 0:
                    queue.append(vi)
        if len(visited) != numCourses:
            return []
        else:
            ret.reverse()
            return ret


class canFinish2:
    def getV0(self, notVisited: set, mat: list):
        now = None
        for u in notVisited:
            flag = 1
            for x in mat[u]:
                if x != -1:
                    flag = 0
                    break
            if flag:
                now = u
                break
        return now

    def dfs(self, u, notvisited: set, mat: list, path: list):

        if len(notvisited) == 0:
            path.reverse()
            print(path)
            return True

        t = [0 for i in range(len(mat))]

        for i in range(len(mat)):
            t[i] = mat[i][u]
            mat[i][u] = -1

        for v in notvisited:
            flag = 1
            for j in mat[v]:
                if j != -1:
                    flag = 0
                    break
            if not flag:
                continue
            notvisited.remove(v)
            path.append(v)
            self.dfs(v, notvisited, mat, path)
            path.pop()
            notvisited.add(v)
        for i in range(len(mat)):
            mat[i][u] = t[i]

            # for i in range(len(mat)):
            #     mat[i][u] = -1
            # # v = self.getV0(notvisited, mat)
            #
            # if v is None:
            #     return False
            # path.append(v)
            # notvisited.remove(v)
            #
            # # return self.dfs(v, notvisited, mat, path)
            # f = self.dfs(v, notvisited, mat, path)
            # if not f:
            #     return False
            # notvisited.add(v)
            # mat[i][u] = 1

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        mat = [[-1 for i in range(numCourses)] for j in range(numCourses)]

        for p in prerequisites:
            mat[p[0]][p[1]] = 1

        notvisited = set(range(numCourses))

        for i in range(numCourses):
            ff = 1
            for j in mat[i]:
                if j != -1:
                    ff = 0
            if not ff:
                continue
            u = i
            # u = self.getV0(notvisited, mat)
            # if u is None:
            #     return False
            notvisited.remove(u)
            path = [u]
            self.dfs(u, notvisited, mat, path)
            notvisited.add(u)
        return True


class canFinish:
    def getV0(self, notVisited: set, mat: list):
        now = None
        for u in notVisited:
            flag = 1
            for x in mat[u]:
                if x != -1:
                    flag = 0
                    break
            if flag:
                now = u
                break
        return now

    def dfs(self, u, notVisited: set, mat: list, path: list):
        if len(notVisited) == 0:
            path.reverse()

            print(path)

            return True
        for i in range(len(mat)):
            mat[i][u] = -1
        v = self.getV0(notVisited, mat)
        if v is None:
            return False
        path.append(v)
        notVisited.remove(v)

        return self.dfs(v, notVisited, mat, path)
        # f = self.dfs(v, notVisited, mat, path)
        # if not f:
        #     return False
        # notVisited.add(v)
        # mat[i][u] = 1

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        mat = [[-1 for i in range(numCourses)] for j in range(numCourses)]

        for p in prerequisites:
            mat[p[0]][p[1]] = 1

        notVisited = set(range(numCourses))

        while len(notVisited) != 0:
            u = self.getV0(notVisited, mat)
            if not u:
                return False
            notVisited.remove(u)
            path = [u]
            if not self.dfs(u, notVisited, mat, path):
                return False
                # notVisited.add(u)
        return True


class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums_set = nums
        for i in nums_set:
            if target - i in nums_set:

                if i != target - i:
                    return [nums.index(i), nums.index(target - i)]
                else:
                    ii = nums.index(i)
                    jj = nums.index(target - i, ii + 1)
                    return [ii, jj]

    def __init__(self):
        self.fast = dict()
        self.fast[0] = 0
        self.fast[1] = 1
        self.fast[4] = 1
        self.fast[9] = 1
        self.fast[16] = 1

    def min(self, a, b):
        if a >= b:
            return b
        else:
            return a

    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """

        if n in self.fast:
            return self.fast[n]

        ret = 10000000

        bound = n

        for i in range(1, n + 1):
            if i * i > n:
                bound = i
                break
        for j in range(bound, 0, -1):
            if n >= j * j:
                ret = self.min(self.numSquares(n - j * j) + 1, ret)
        self.fast[n] = ret
        return ret

    def dfs(self, u, notvisited, mat, path):
        notvisited.remove(u)
        for i, v in enumerate(mat[u]):
            if i in path:
                return False
            if i not in notvisited or v == -1:
                continue
            path.append(i)
            if not self.dfs(i, notvisited, mat, path):
                return False
            path.pop()
        notvisited.remove(u)
        return True

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        mat = [[-1 for i in range(numCourses)] for j in range(numCourses)]

        for p in prerequisites:
            mat[p[0]][p[1]] = 1

        not_visited = set(range(numCourses))
        for i in not_visited:
            if not self.dfs(i, not_visited, mat, [i]):
                return False
        return True

    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """

    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        ab = {}

        for i, a in enumerate(nums):
            for j, b in enumerate(nums):
                if i == j:
                    continue
                k = a + b
                v = [i, j]
                v.sort()
                v = tuple(v)

                if k not in ab:
                    ab[k] = set()
                ab[k].add(v)
        ans = set()

        for k in ab:
            u = target - k
            if u not in ab:
                continue
            for a_l in ab[k]:
                for b_l in ab[u]:
                    ts = set(a_l + b_l)
                    if len(ts) == 4:
                        ts = list(ts)
                        ts.sort()
                        ans.add(tuple(ts))
        ans = list(ans)
        ret = set()
        for i, j, k, l in ans:
            t = [nums[i], nums[j], nums[k], nums[l]]
            t.sort()
            ret.add(tuple(t))
        ans = []
        for one in ret:
            ans.append(list(one))
        return ans

    def threeSum2(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ab = {}
        for i, a in enumerate(nums):
            for j, b in enumerate(nums):
                if i == j:
                    continue
                k = a + b
                if k not in ab:
                    ab[k] = set()
                v = (i, j)
                ab[k].add(v)
        ans_set = set()

        for ci, c in enumerate(nums):
            if -c not in ab:
                continue
            tab = list(ab[-c])
            for i, j in tab:
                if i == ci or j == ci:
                    continue
                tans = [nums[i], nums[j], c]
                tans.sort()
                ans_set.add(tuple(tans))

        # for ci, c in enumerate(nums):
        #     if -c not in ab:
        #         continue
        #     for ab_v in ab[-c]:
        #         ab_v += (ci,)
        #         tv = set(ab_v)
        #         if len(tv) != 3:
        #             continue
        #         tvv = []
        #         for m in tv:
        #             tvv.append(nums[m])
        #         tvv.sort()
        #         ans_set.add(tuple(tvv))

        ret = []
        for one in ans_set:
            ret.append(list(one))
        return ret

    def sum2(self, nums: list, s, t, target):
        ret = []
        while s < t:
            v = nums[s] + nums[t]
            if v == target:
                ret.append([nums[s], nums[t]])
                while s < t and nums[s] == nums[s + 1]:
                    s += 1
                while s < t and nums[t] == nums[t - 1]:
                    t -= 1
                s += 1
                t -= 1
            elif v < target:
                s += 1
            else:
                t -= 1
        return ret

    def threeSum(self, nums: list):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) < 3:
            return []

        nums.sort()

        ans_set = set()

        length = len(nums)

        for i in range(length - 2):
            if i == 0 or (i > 0 and nums[i] != nums[i - 1]):
                xs = self.sum2(nums, i + 1, length - 1, -nums[i])
                for x in xs:
                    t = []
                    t.extend(x)
                    t.append(nums[i])
                    t.sort()
                    ans_set.add(tuple(t))
        ret = []
        for one in ans_set:
            ret.append([one[0], one[1], one[2]])
        return ret

    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        queue = []
        for i, vi in enumerate(matrix):
            for j, vij in vi:
                if vij == 0:
                    queue.append((i, j))

    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        ab = {}

        for a in A:
            for b in B:
                v = a + b
                if v in ab:
                    ab[v] += 1
                else:
                    ab[v] = 1

        ans = 0
        for c in C:
            for d in D:
                p = -(c + d)
                ans += ab.get(p, 0)
        return ans

    def travel(self, p: TreeNode, l):
        if p is None:
            return None, None
        if p.left is None and p.right is None:
            return l, p.val

        h1 = -1
        h2 = -1

        if p.left is not None:
            h1, f1 = self.travel(p.left, l + 1)

        if p.right is not None:
            h2, f2 = self.travel(p.right, l + 1)
        if h1 < h2:
            return h2, f2
        else:
            return h1, f1

    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        h, f = self.travel(root, 0)
        return f


class RandomizedCollection(object):
    def __init__(self):

        self.n = 0
        self.data = {}
        self.keys = set()
        self.values = []

    def insert(self, val):
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        self.n += 1
        if val in self.keys:
            self.data[val].append(len(self.values) - 1)
            self.values.append(val)
            return False
        self.data[val] = [len(self.values) - 1]
        return True

    def remove(self, val):
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.data:
            return False

        self.n -= 1
        i = self.data[val].pop()
        if len(self.data[val]) == 0:
            self.data.pop(val)

            # self.keys.remove(val)
        return True

    def getRandom(self):
        """
        Get a random element from the collection.
        :rtype: int
        """
        f = random.randint(0, self.n - 1)
        for k, v in self.data.items():
            f -= v
            if f <= 0:
                return k


if __name__ == '__main__':
    # data = np.loadtxt('data/data_of_lab.csv')
    # tests = Solution()
    # print(tests.canFinish(4, [[1, 0], [0, 2], [3, 0], [3, 2]]))

    # can = canFinish3()
    # print(can.canFinish(4, [[1, 0], [0, 2], [3, 0], [3, 2]]))
    # print(can.canFinish(2, [[0, 1]]))
    # twitter = Twitter()
    # twitter.postTweet(1, 5)
    # twitter.getNewsFeed(1)
    # twitter.follow(1, 2)

    # twitter.postTweet(2, 6)
    # twitter.getNewsFeed(1)
    # twitter.unfollow(1, 2)
    # twitter.getNewsFeed(1)
    sol = Solution()
    nums = [1, 0, -1, 0, -2, 2]

    print(sol.threeSum(nums))

    s = [
        [-3, -2, 2, 3],
        [-3, -1, 1, 3],
        [-3, 0, 0, 3],
        [-3, 0, 1, 2],
        [-2, -1, 0, 3],
        [-2, -1, 1, 2],
        [-2, 0, 0, 2],
        [-1, 0, 0, 1]
    ]
