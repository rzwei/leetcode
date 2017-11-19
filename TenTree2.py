import collections
import random
import time

TREE_NODE_NUM = 10
NONE_NODE_CHAR = 'X'
SPLIT_CHAR = '#'


class Node:
    def __init__(self, val=None):
        if not val:
            self.val = random.randint(0, 1000000)
        else:
            self.val = int(val)
        self.next = [None] * TREE_NODE_NUM


def min_height(root: Node):
    if not root:
        return 0
    ret = 100
    for q in root.next:
        ret = min(ret, min_height(q) + 1)
    return ret


def max_height(root: Node):
    if not root:
        return 0
    ret = 0
    for q in root.next:
        ret = max(ret, max_height(q) + 1)
    return ret


def node_count(p: Node):
    if not p:
        return 0
    ret = 0
    for q in p.next:
        ret += node_count(q)
    return ret + 1


def leaves_count(p: Node):
    if not p:
        return 0
    ret = 0
    for q in p.next:
        ret += leaves_count(q)
    if ret == 0:
        return 1
    else:
        return ret


def build_rand(deep=7):
    # if deep == 0:
    #     return None
    if deep == 1:
        # head = Node()
        # for i in range(TREE_NODE_NUM):
        #     if random.randint(1, 10) > 3:
        #         head.next[i] = Node()
        # return head
        if random.randint(0, 9) > 3:
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
    child = ''
    for i in p.next:
        child += SPLIT_CHAR + inorder(i)
    return str(p.val) + child


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
        cur = nodes.popleft()
        if cur == NONE_NODE_CHAR:
            return None
        node = Node(cur)
        for i in range(TREE_NODE_NUM):
            node.next[i] = self.build(nodes)
        return node

    def deserialize(self, data: str):
        nodes = data.split(SPLIT_CHAR)
        nodes = collections.deque(nodes)
        return self.build(nodes)


class Codec_bfs:
    def serialize(self, head: Node):
        queue = collections.deque()
        queue.append(head)
        data = ''
        while queue:
            cur = queue.popleft()
            if not cur:
                data += SPLIT_CHAR + NONE_NODE_CHAR
                continue
            data += SPLIT_CHAR + str(cur.val)
            for i in range(TREE_NODE_NUM):
                queue.append(cur.next[i])
        return data[len(SPLIT_CHAR):]

    def deserialize(self, data: str):
        nodes = data.split(SPLIT_CHAR)
        nodes = collections.deque(nodes)
        head = nodes.popleft()
        if head == NONE_NODE_CHAR:
            return None
        head = Node(head)
        queue = collections.deque([head])
        while queue:
            cur = queue.popleft()
            for i in range(TREE_NODE_NUM):
                t = nodes.popleft()
                if t != NONE_NODE_CHAR:
                    cur.next[i] = Node(t)
                    queue.append(cur.next[i])
        return head


def test(root, codec):
    data = codec.serialize(root)
    # data2 = codec.serialize(codec.deserialize(data))
    # print('compare ', data == data2)
    print('serialize test')
    times = 1
    for i in range(times):
        cur = time.time()
        codec.serialize(root)
        print('step {} serialize time: {}'.format(i + 1, (time.time() - cur)))

    print('deserialize test')
    for i in range(times):
        cur = time.time()
        codec.deserialize(data)
        print('step {} deserialize time: {}'.format(i + 1, (time.time() - cur)))


if __name__ == '__main__':
    root = build_rand(7)
    data = inorder(root)
    print('generate random tree finished!')

    print('max height', max_height(root))
    print('min height', min_height(root))
    print('total nodes count', node_count(root))
    print('leaves count', leaves_count(root))

    with open('tree.txt', 'w', encoding='utf-8') as fin:
        fin.write(data)

    codec = Codec()
    codec_bfs = Codec_bfs()
    test(root, codec_bfs)
    test(root, codec)
