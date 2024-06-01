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

    def reverseDoubleList(self, head):  # 翻转链表
        # 方法1：
        pre = None
        cur = head
        while cur:
            temp = cur.next
            cur.next = pre
            cur.prev = temp
            pre = cur
            cur = temp
        return pre

    def reverseList_between(self, a, b):  # 翻转链表(a, b)之间
        pre = ListNode(-1)
        cur = a
        while cur != b:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre
    def reverseKGroup(self, head, k):  # k个一组翻转链表
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
        slow, fast = head, head.next
        # 二分法找中间节点，取划分链表
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        left, right = head, slow.next
        slow.next = None
        return self.mergeTwoLists(self.sortList(left), self.sortList(right))

    def merge_sort(self, lists, left, right):
        if left == right:
            return lists[left]
        if left > right:
            return None
        mid = left + (right - left) // 2
        return self.mergeTwoLists(self.merge_sort(lists, left, mid), self.merge_sort(lists, mid + 1, right))

        return
    def mergeKLists(self, lists):  # 合并K个升序链表（归并）
        return self.merge_sort(lists, 0, len(lists) - 1)


    def reOrderList(self, head):# 重排链表 l0-ln-l1-ln-1-l2-ln-2
        # 链表双指针
        vec = []
        node = head
        while node:
            vec.append(node)
            node = node.next
        left, right = 0, len(vec) - 1
        while left < right:
            vec[left].next = vec[right]
            left += 1
            if left == right:
                break
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
                break
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

    def addTwoNumbers(self, l1, l2):  # 两数相加 （数字是逆序存放的例如：[2, 4, 3]-> 342
        dummy_node = ListNode(0)
        head = cur = dummy_node  # 存储结果的值
        carry = 0
        # 最后需要加一个以防止结果多出一位数
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
            sum = (x1 + x2 + carry)
            carry = sum // 10
            b = sum % 10
            # print("x1:", x1, "x2:", x2, "b", b, "carry", carry)
            cur.next = ListNode(b)
            cur = cur.next
        return head.next

    def addTwoNumbers2(self, l1, l2):  # 两数相加 （数字是顺序存放的例如：[2, 4, 3]-> 243
        L1 = self.reverseList(l1)
        L2 = self.reverseList(l2)
        L1plusL2 = self.addTwoNumbers(L1, L2)
        return self.reverseList(L1plusL2)








