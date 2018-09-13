import random


class RandomizedSet(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.d = {}
        self.nums = []

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.d:
            return False
        self.d[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.d:
            return False
        i = self.d[val]
        lasti = self.d[self.nums[-1]]
        self.d[self.nums[-1]] = i
        self.nums[i] = self.nums[lasti]
        del self.d[val]
        self.nums.pop()
        return True

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return self.nums[random.randint(0, len(self.nums) - 1)]


class Solution:
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums
        # self.indexes = {}
        # for i, vi in enumerate(nums):
        #     if vi not in self.indexes:
        #         self.indexes[vi] = [i]
        #     else:
        #         self.indexes[vi].append(i)
        # def pick(self, target):
        #     """
        #     :type target: int
        #     :rtype: int
        #     """
        #     indexes = self.indexes[target]
        #     r = indexes[0]
        #     for i in range(1, len(indexes)):
        #         if random.randint(1, i + 1) == i:
        #             r = indexes[i]
        #     return r

    def pick(self, target):
        """
        :type target: int
        :rtype: int
        """
        r = -1
        n = -1
        for i, vi in enumerate(self.nums):
            if vi == target:
                n += 1
                if r == -1:
                    r = i
                elif r != -1:
                    if random.randint(1, n + 1) == n:
                        r = i
        return r


if __name__ == '__main__':
    sol = Solution([1, 2, 3, 4, 3, 32, 2, 3, 3, 2, 12, 2, 3, 1, 2])
    print(sol.pick(1))
