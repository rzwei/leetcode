import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

# board = np.zeros(shape=(100, 100), dtype=np.uint8)
board = np.random.randint(2, size=(100, 100), dtype=np.uint8)


def gameOfLife(board):
    m = len(board)
    n = len(board[0])

    def neighbors(x, y):
        res = 0
        for i in range(-1, 1 + 1):
            for j in range(-1, 1 + 1):
                if i == 0 and j == 0:
                    continue
                nx = x + i
                ny = y + j
                if nx < 0:
                    nx = nx + m
                elif nx >= m:
                    nx -= m
                if ny < 0:
                    ny += n
                elif ny >= n:
                    ny -= n
                if board[nx][ny] & 1:
                    res += 1
        return res

    for x in range(m):
        for y in range(n):
            count = neighbors(x, y)
            if board[x][y] & 1:
                if count < 2:
                    board[x][y] &= 1
                elif count < 4:
                    board[x][y] |= 2
                else:
                    board[x][y] &= 1
            elif count == 3:
                board[x][y] |= 2
    for x in range(m):
        for y in range(n):
            board[x][y] >>= 1
            # return board


fig = plt.figure()
im = plt.imshow(board)


def animate(_):
    gameOfLife(board)
    im.set_array(board)
    return [im]


anim = animation.FuncAnimation(fig, animate, interval=1000 / 25, blit=True)
# anim.save('gameOfLife.mp4', fps=60)
plt.show()
