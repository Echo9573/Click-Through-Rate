class DoubleLinkNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:  # 每次淘汰那些最久没被使用的数据

    def __init__(self, capacity: int):
        # 哈希表表征当前缓存数据, key是当前的索引值，cache的value存储的是节点
        self.cache = {}
        self.head = DoubleLinkNode()
        self.tail = DoubleLinkNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 如果key存在，先通过哈希表定位，再修改value并移动到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            # 如果key不存在，创建一个新节点
            node = DoubleLinkNode(key, value)
            # 添加进哈希表cache
            self.cache[key] = node
            # 添加到双向链表的头部
            self.addTohead(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表尾节点 & 哈希表种对应的key和value
                removed = self.removeTail()  # 这里返回的是个节点，所以在哈希表删除的时候，要获取它的key
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # 如果key存在，先通过哈希表定位，再修改value并移动到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)

    def addTohead(self, node):  # next尾部是recent
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node  # 备注 step3 和step4不能颠换顺序，step2和step1可以换顺序
        self.head.next = node

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def removeTail(self):  # tail尾部是最久远的
        node = self.tail.prev
        self.removeNode(node)
        return node

    def moveToHead(self, node):
        self.removeNode(node)
        self.addTohead(node)

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


from collections import defaultdict, OrderedDict
class LFUCache:
    def __init__(self, capacity):
        self.min_freq = 0
        self.key_to_value_freq = {}
        self.freq_to_values = defaultdict(OrderedDict)
        self.capacity = capacity

    def put(self, key, value):
        if self.capacity <= 0:
            return
        # 第一种情况: key 不在：①满了，先删除频次最少&访问最久的 ②没满 ③加入，频次为1
        if key not in self.key_to_value_freq:
            if len(self.key_to_value_freq) == self.capacity:
                # 找到频次最少&访问最久的key
                least_min_key = next(iter(self.freq_to_values[self.min_freq]))
                del self.key_to_value_freq[least_min_key]
                del self.freq_to_values[self.min_freq][least_min_key]
            self.key_to_value_freq[key] = [value, 1]
            self.freq_to_values[1][key] = None
            self.min_freq = 1 # 一个纯新的加入后，min_freq重置为1
        # 第二种情况: key 在，更新最新的value值，【频次加1，self.freq_to_values原先的旧频次删除这个key,新频次（旧频次+1）加入这个key】==这里等价于一个get操作
        else:
            self.key_to_value_freq[key][0] = value
            self.get(key)

    def get(self, key): # 返回当前key的value值
        # 元素不在，则返回-1
        if key not in self.key_to_value_freq:
            return -1

        # 增加频次 & 原先的旧频次删除这个key
        old_freq = self.key_to_value_freq[key][1]
        self.key_to_value_freq[key][1] += 1
        del self.freq_to_values[old_freq][key]

        # 更新最小频次
        if not self.freq_to_values[old_freq] and old_freq == self.min_freq:
        # if len(self.freq_to_values[old_freq]) == 0 and old_freq == self.min_freq:
            self.min_freq += 1

        # 新频次（旧频次+1）加入这个key
        new_freq = old_freq + 1
        self.freq_to_values[new_freq][key] = None
        return self.key_to_value_freq[key][0]



