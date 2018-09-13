def binarysearch(num, m, n):
    ret = 0
    for row in range(1, m + 1):
        i = 1
        j = n + 1
        while i < j:
            mid = (i + j) // 2
            if row * mid > num:
                j = mid
            else:
                i = mid + 1
        ret += i - 1
    return ret


if __name__ == '__main__':
    # print(binarysearch(6, 2, 3))
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    d = {}
    for i, v in enumerate(words):
        d[i] = set(v)

    S = set([chr(i + ord('a')) for i in range(26)])
    for s in d.values():
        S &= s
    print('S', S)
    for k, v in d.items():
        print(k, v - S)
