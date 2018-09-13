import os
import shutil

FILEPATH = 'word.txt'


def copyText(path, dep=3):
    if dep == 0:
        for i in range(10):
            filenamme = path + "/word" + str(i) + ".txt"
            shutil.copy(FILEPATH, filenamme)
        return
    for i in range(2):
        cur = path + '/' + str(i)
        os.mkdir(cur)
        copyText(cur, dep - 1)
