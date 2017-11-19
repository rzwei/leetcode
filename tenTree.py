import random
import time

TREE_NODE_NUM = 7
NONE_NODE_CHAR = 'X'
SPLIT_CHAR = '#'


class Node:
    def __init__(self, val=None):
        if not val:
            self.val = random.randint(0, 1000000)
        else:
            self.val = int(val)
        self.next = [None] * TREE_NODE_NUM


def build_rand(deep=7):
    # if deep == 0:
    #     return None
    if deep == 1:
        if random.randint(1, 10) > 3:
            return Node()
        else:
            return None
    head = Node()
    for i in range(TREE_NODE_NUM):
        head.next[i] = build_rand(deep - 1)
    return head


def inorder(p: Node):
    if not p:
        return NONE_NODE_CHAR
    print(p.val, end=',')
    for i_ in p.next:
        inorder(i_)


class Codec:
    def preorder(self, node: Node):
        if not node:
            return NONE_NODE_CHAR
        child = ''
        for i in range(TREE_NODE_NUM):
            child += SPLIT_CHAR + self.preorder(node.next[i])
        return str(node.val) + child

    def serialize(self, root: Node):
        return self.preorder(root)

    def build(self, nodes):
        cur = nodes.pop(0)
        if cur == NONE_NODE_CHAR:
            return None
        node = Node(cur)
        for i in range(TREE_NODE_NUM):
            node.next[i] = self.build(nodes)
        return node

    def deserialize(self, data: str):
        nodes = data.split(SPLIT_CHAR)
        return self.build(nodes)


class Codec_bfs:
    def serialize(self, head: Node):
        queue = [head]
        data = ''
        while queue:
            cur = queue.pop(0)
            if not cur:
                data += SPLIT_CHAR + NONE_NODE_CHAR
                continue
            data += SPLIT_CHAR + str(cur.val)
            for i in range(TREE_NODE_NUM):
                queue.append(cur.next[i])
        return data[len(SPLIT_CHAR):]

    def deserialize(self, data: str):
        nodes = data.split(SPLIT_CHAR)
        head = nodes.pop(0)
        if head == NONE_NODE_CHAR:
            # head=Node()
            return None
        head = Node(head)
        queue = [head]
        while queue:
            cur = queue.pop(0)
            for i in range(TREE_NODE_NUM):
                t = nodes.pop(0)
                if t != NONE_NODE_CHAR:
                    cur.next[i] = Node(t)
                    queue.append(cur.next[i])
        return head


def test_(root):
    # root = build_rand()
    codec = Codec()
    times = 3
    cur = time.time()
    for _ in range(times):
        data = codec.serialize(root)
    print('serialize avg time: ', (time.time() - cur) / times)
    # with open('tree_.txt', 'w', encoding='utf-8') as fin:
    #     fin.write(data)
    cur = time.time()
    for i in range(times):
        codec.deserialize(data)
    print('deserialize avg time: ', (time.time() - cur) / times)


def test_bfs(root):
    # root = build_rand()
    codec = Codec_bfs()
    times = 3
    cur = time.time()
    for _ in range(times):
        data = codec.serialize(root)
    print('serialize avg time: ', (time.time() - cur) / times)
    # with open('tree_bfs.txt', 'w', encoding='utf-8') as fin:
    #     fin.write(data)
    cur = time.time()
    for i in range(times):
        codec.deserialize(data)
    print('deserialize avg time: ', (time.time() - cur) / times)


if __name__ == '__main__':
    root = build_rand(6)
    test_(root)
    test_bfs(root)
