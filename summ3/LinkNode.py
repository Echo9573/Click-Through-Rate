class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class DoublyListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class Solution(object):
    def swapPairs(self, head): # 24. 两两交换链表中的节点
        dummy_node = ListNode(-1)
        dummy_node.next = head
        cur = dummy_node
        while cur.next and cur.next.next:
            node1, node2 = cur.next, cur.next.next
            cur.next = node2
            node1.next = node2.next
            node2.next = node1
            cur = node1
        return dummy_node.next

    def deleteDuplicates(self, head):  # 删除已排序的链表中的重复元素（多余的删除）
        if not head:
            return None
        cur = head
        while cur and cur.next:
            if cur.next.val == cur.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

    def deleteDuplicates2(self, head):  # 删除已排序的链表中的重复元素（重复的删除）
        dummy_node = ListNode(-1)
        dummy_node.next = head
        cur = dummy_node
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                temp = cur.next
                while temp and temp.next and (temp.val == temp.next.val):
                    temp = temp.next
                cur.next = temp.next
            else:
                cur = cur.next
        return dummy_node.next

    def reverseList(self, head):  # 翻转链表
        # # 方法1：
        # pre = None
        # cur = head
        # while cur:
        #     temp = cur.next
        #     cur.next = pre
        #     pre = cur
        #     cur = temp
        # return pre

        # 方法2：递归
        if not head or (not head.next):
            return head
        last = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return last

    def reverseBeween(self, head, left, right):  # 92.反转链表 II
        # 方法1：非递归
        if not head:
            return None
        dummy_node = ListNode(0)
        dummy_node.next = head
        cur_0 = dummy_node
        index = 1
        while cur_0.next and index < left:
            cur_0 = cur_0.next
            index += 1
        prev = cur_0
        cur = prev.next
        while cur and index <= right:
            temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp
            index += 1
        cur_0.next.next = cur
        cur_0.next = prev
        return dummy_node.next

    def reverseBeween_way2(self, head, left, right):
        def reverseTopk(head, n):
            if n == 1:
                return head
            last = reverseTopk(head.next, n - 1)
            succ = head.next.next
            head.next.next = head
            head.next = succ
            return last
        m, n = left, right
        if n == 1:
            return head
        if m == 1:
            return reverseTopk(head, n)
        head.next = self.reverseBetween(head.next, m - 1, n - 1)
        return head

    def reverseDoubleList(self, head):  # 翻转双向链表
        pre = None
        cur = head
        while cur:
            temp = cur.next
            cur.next = pre
            cur.prev = temp
            pre = cur
            cur = temp
        return pre

    
    def reverseKGroup(self, head, k):  # k个一组翻转链表
        def reverseList_between(a, b):  # 翻转链表(a, b)之间
            pre = ListNode(-1)
            cur = a
            while cur != b:
                temp = cur.next
                cur.next = pre
                pre = cur
                cur = temp
            return pre
            
        if not head:
            return None
        a = head
        b = head
        for i in range(k):  # 找到和a相距k的b[a, b)
            if not b:  # 不足k个，不需要反转
                return head
            b = b.next
        # 反转前k个元素
        new_head = self.reverseList_between(a, b)
        a.next = self.reverseKGroup(b, k)
        return new_head

    def oddEvenList(self, head): # 奇偶链表
        if not head or (not head.next) or (not head.next.next):
            return head
        evenhead = head.next
        odd, even = head, evenhead
        isOdd = True
        cur = head.next.next
        while cur:
            if isOdd:
                odd.next = cur
                odd = cur
            else:
                even.next = cur
                even = cur
            isOdd = not isOdd
            cur = cur.next
        odd.next = evenhead
        # 记得这个尾指针一定要加上，要不然有问题
        even.next = None
        return head


    def isPalindrome(self, head):  # 回文链表（快慢指针法）
        # way 1：快慢指针法
        if not head:
            return head
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        if fast:
            slow = slow.next
        right = self.reverseList(slow)
        left = head
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        return True

    def mergeTwoLists(self, left, right):  # 合并两个有序链表
        dummy_head = ListNode(-1)
        cur = dummy_head
        while left and right:
            if left.val < right.val:
                cur.next = left
                left = left.next
            else:
                cur.next = right
                right = right.next
            cur = cur.next
        if left:
            cur.next = left
        if right:
            cur.next = right
        return dummy_head.next
    def sortList(self, head):   # 链表的归并排序
        if not head or not head.next:
            return head
        slow, fast = head, head.next # fast是head.next
        # 二分法找中间节点，取划分链表
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        left, right = head, slow.next
        slow.next = None # slow.next必须指向空
        return self.mergeTwoLists(self.sortList(left), self.sortList(right))

    def merge_sort(self, lists, left, right):
        if left == right:
            return lists[left]
        if left > right:  # 这个条件必须加上，要不然死循环
            return None
        mid = left + (right - left) // 2
        return self.mergeTwoLists(self.merge_sort(lists, left, mid), self.merge_sort(lists, mid + 1, right))

        return
    def mergeKLists(self, lists):  # 合并K个升序链表（归并）
        return self.merge_sort(lists, 0, len(lists) - 1)


    def reOrderList(self, head):# 重排链表 l0-ln-l1-ln-1-l2-ln-2
        # 链表双指针
        vec = []
        cur = head
        while cur:
            vec.append(cur)
            cur = cur.next
        left, right = 0, len(vec) - 1
        while left < right:
            vec[left].next = vec[right]
            left += 1
            vec[right].next = vec[left]
            right -= 1

        vec[left].next = None

    def hasCycle(self, head):
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    def detectCycle(self, head):
        fast = slow = head
        hascycle = 0
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                hascycle = 1
                break  # 一定要break
        if hascycle:
            slow = head
            while fast != slow:
                fast = fast.next
                slow = slow.next
            return slow
        else:
            return None

    def getIntersectionNode(self, headA, headB):  # 链表相交
        l1, l2 = headA, headB
        if not headA or not headB:
            return None
        while l1 != l2:
            if not l1:
                l1 = headB
            else:
                l1 = l1.next
            if not l2:
                l2 = headA
            else:
                l2 = l2.next
        return l1

    def tailKNode(self, head, k):  # 链表中倒数第K个节点
        # 快慢指针
        dummy_node = ListNode(-1)
        dummy_node.next = head
        fast = head
        slow = dummy_node
        while cnt > 0:
            fast = fast.next
            cnt -= 1
        while fast:
            fast = fast.next
            slow = slow.next
        return slow.next

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:  # 删除链表倒数第K个节点
        # 方法2：
        dummy_node = ListNode(-1)
        dummy_node.next = head
        cur = self.findEndn(dummy_node, n + 1)
        cur.next = cur.next.next
        return dummy_node.next
        # # 方法1
        # dummy_node = ListNode(-1)
        # dummy_node.next = head
        # fast = head
        # slow = dummy_node
        # while n > 0:
        #     fast = fast.next
        #     n -= 1
        # while fast:
        #     fast = fast.next
        #     slow = slow.next
        # slow.next = slow.next.next
        # return dummy_node.next # 因为有可能返回空链表，所以必须是从空的哑node开始

    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        # 使用字典记录原节点和新节点
        nodedict = {}
        # 先复制链表中对应的节点
        cur = head
        while cur:
            nodedict[cur] = Node(cur.val, None, None)
            cur = cur.next
        cur = head
        # 再链接链表中节点指向性关系
        while cur:
            if cur.next:
                nodedict[cur].next = nodedict[cur.next]
            if cur.random:
                nodedict[cur].random = nodedict[cur.random]
            cur = cur.next
        return nodedict[head]

    def addTwoNumbers(self, l1, l2):  # 两数相加 （数字是逆序存放的例如：[2, 4, 3]-> 342
        dummy_node = ListNode(-1)
        cur = dummy_node
        carry = 0
        while l1 or l2 or carry:
            if l1:
                x1 = l1.val
                l1 = l1.next
            else:
                x1 = 0
            if l2:
                x2 = l2.val
                l2 = l2.next
            else:
                x2 = 0
            sum = x1 + x2 + carry
            carry = sum // 10
            b = sum % 10
            cur.next = ListNode(b)
            cur = cur.next

        return dummy_node.next

    def addTwoNumbers2(self, l1, l2):  # 两数相加 （数字是顺序存放的例如：[2, 4, 3]-> 243
        L1 = self.reverseList(l1)
        L2 = self.reverseList(l2)
        L1plusL2 = self.addTwoNumbers(L1, L2)
        return self.reverseList(L1plusL2)
