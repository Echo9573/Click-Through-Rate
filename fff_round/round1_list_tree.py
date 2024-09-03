class ListNode:
    def __init__(self, val, next=None):
       self.val = val
       self.next = next
class Node:
    def __init__(self, val, next=None, random=None):
      self.val = val
      self.next = next
      self.random = random
class ListNodeSolution:
  # def swapPairs(self, head): # 1
  # def deleteDuplicates(self, head): # 2
  # def deleteDuplicates2(self, head): # 3
  # def reverseList(self, head): # 4
  # def reverseDoubleList(self, head): # 5
  # def reverseListBetween(self, a, b): # 6
  # def reverseKgroup(self, head, k): # 7
  # def oddEvenList(self, head): # 8
  # def isPalindomeList(self, head): # 9
  # def mergeTwoList(self, list1, list2): # 10
  # def mergeKLists(self, lists): # 11
  # def SortList(self, head): # 12
  # def reOrderList(self, head): # 13
  # def hasCycle(self, head): # 14
  # def detectCycle(self, head): # 15
  # def getInterSectNode(self, list1, list2): # 16
  # def tailKNode(self, head, k): # 17
  # def removeKthFromEnd(self, head, k): # 18
  # def copyRandomList(self, head): # 19
  # def addTowNum(self, head): # 20
  # def rotateList(self, head, k): # 21
  
  def swapPairs(self, head): # 1
    dummy_node = ListNode(-1)
    dummy_node.next = head
    cur = dummy_node
    while cur.next and cur.next.next:
      node1 = cur.next
      node2 = cur.next.next
      temp = node2.next
      cur.next = node2
      node2.next = node1
      node1.next =temp
      cur = node1
    return dummy_node.next
  
  def deleteDuplicates(self, head): # 2
    cur = head
    while cur and cur.next:
      if cur.val == cur.next.val:
        cur.next = cur.next.next
      else:
        cur = cur.next
    return head
  def deleteDuplicates2(self, head): # 3
    dummy_node = ListNode(-1)
    dummy_node.next = head
    cur = dummy_node
    while cur.next and cur.next.next:
      if cur.next.val == cur.next.next.val:
        temp = cur.next
        while temp and temp.next:
          if temp.val == temp.next.val:
            temp.next = temp.next.next
          else:
            break
        cur.next = temp.next
      else:
        cur = cur.next
    return dummy_node.next
  def reverseList(self, head): # 4
    # # way1:
    # prev = None
    # cur = head
    # while cur:
    #   temp = cur.next
    #   cur.next = prev
    #   prev = cur
    #   cur = temp
    # return prev
    # way2:
    if not head or (not head.next):
      return head
    last = self.reverseList(head.next)
    head.next.next = head
    head.next = None
    return last
  def reverseDoubleList(self, head): # 5
    prev = None
    cur = head
    while cur:
      temp = cur.next
      cur.next = prev
      cur.pre = temp
      prev = cur
      cur = temp
    return prev

  def reverseListBetweenLen(self, head, left, right):
    dummy_node = ListNode(-1)
    dummy_node.next = head
    cur0 = dummy_node
    index = 1
    while cur0.next and index < left:
      cur0 = cur0.next
      index += 1
    prev = cur0
    cur = cur0.next
    while cur and index <= right:
      temp = cur.next
      cur.next = prev
      prev = cur
      cur = temp
      index += 1
    cur0.next.next = cur
    cur0.next = prev
    return dummy_node.next
  def reverseListBetween(self, a, b): # 6
    prev = None
    cur = a
    while cur and cur != b:
      temp = cur.next
      cur.next = prev
      prev = cur
      cur = temp
    return prev
  def reverseKgroup(self, head, k): # 7
    if not head:
      return head
    a, b = head, head
    for i in range(k):
      if not b:
        return head
      b = b.next
    new_head = self.reverseListBetween(a, b)
    a.next = self.reverseKgroup(b, k)
    return new_head
  def oddEvenList(self, head): # 8
    if not head or (not head.next) or (not head.next.next):
      return head
    even_head = head.next
    odd, even = head, head.next
    cur = head.next.next
    is_odd = True
    while cur:
      if is_odd:
        odd.next = cur
        odd = cur
      else:
        even.next = cur
        even = cur
      is_odd = not is_odd
      cur = cur.next
    odd.next = even_head
    even.next = None
    return head
  def isPalindomeList(self, head): # 9
    if not head or (not head.next):
      return True
    slow, fast = head, head
    while fast and fast.next:
      fast = fast.next.next
      slow = slow.fast
    if fast:
      slow = slow.next
    right = self.reverseList(slow)
    left = head
    while right:
      if right.val != left.val:
        return False
      left = left.next
      right = right.next
    return True

  def mergeTwoList(self, list1, list2): # 10
    dummy_node = ListNode(-1)
    cur = dummy_node
    while list1 and list2:
      if list1.val < list2.val:
        cur.next = list1
        list1 = list1.next
      else:
        cur.next = list2
        list2 = list2.next
      cur = cur.next
    if list1:
      cur.next = list1
    if list2:
      cur.next = list2
    return dummy_node.next
  def merge_sort(self, lists, left, right):
    if left == right:
      return lists[left]
    if left > right:
      return None
    mid = left + (right - left) // 2
    list1 = self.merge_sort(lists, left, mid)
    list2 = self.merge_sort(lists, mid + 1, right)
    return self.mergeTwoList(list1, list2)
  def mergeKLists(self, lists): # 11
    return self.merge_sort(lists, 0, len(lists) - 1)

  def SortList(self, head): # 12
    if not head or (not head.next):
      return head
    # 因为是归并，所以slow不能等于fast，fast要从head.next开始
    slow, fast = head, head.next
    while fast and fast.next:
      fast = fast.next.next
      slow = slow.next
    right = slow.next
    slow.next = None
    left = head
    return self.mergeTwoList(self.SortList(left), self.SortList(right))
  def reOrderList(self, head): # 13
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
  def hasCycle(self, head): # 14
    if not head or (not head.next):
      return None
    slow, fast = head, head
    while fast and fast.next:
      fast = fast.next.next
      slow = slow.next
      if slow == fast:
        return True
    return False
  def detectCycle(self, head): # 15
    slow, fast = head, head
    hascycle = 0
    while fast and fast.next:
      fast = fast.next.next
      slow = slow.next
      if fast == slow:
        hascycle = 1
        break
    if hascycle:
      slow = head
      while slow != fast:
        slow = slow.next
        fast = fast.next
      return slow
    else:
      return None
  def getInterSectNode(self, list1, list2): # 16
    if not list1 or (not list2):
      return None
    l1, l2 = list1, list2
    while l1 != l2:
      if not l1:
        l1 = list2
      else:
        l1 = l1.next
      if not l2:
        l2 = list1
      else:
        l2 = l2.next
    return l1
  def tailKNode(self, head, k): # 17
    fast = head
    while k > 0:
      fast = fast.next
      k -= 1
    slow = head
    while fast and fast.next:
      slow = slow.next
      fast = fast.next
    return slow
  def removeKthFromEnd(self, head, k): # 18
    dummy_node = ListNode(-1)
    dummy_node.next = head
    prev = self.tailKNode(dummy_node, k + 1)
    prev.next = prev.next.next
    return dummy_node.next
  def copyRandomList(self, head): # 19
    if not head:
      return None
    cur = head
    nodedict = {}
    while cur:
      nodedict[cur] = Node(cur.val, None, None)
      cur = cur.next
    cur = head
    while cur:
      if cur.next:
        nodedict[cur].next = nodedict[cur.next]
      if cur.random:
        nodedict[cur].random = nodedict[cur.random]
      cur = cur.next
    return nodedict[head]
  def addTowNum(self, list1, list2): # 20
    dummy_node = ListNode(0)
    cur = dummy_node
    carry = 0
    while list1 or list2 or carry:
      if list1:
        x1 = list1.val
        list1 = list1.next
      else:
        x1 = 0
      if list2:
        x2 = list2.val
        list2 = list2.next
      else:
        x2 = 0
      sums = x1 + x2 + carry
      carry = sums // 10
      cur.next = ListNode(sums % 10)
      cur = cur.next
    return dummy_node.next

  def rotateList(self, head, k): # 21
    if not head or (not head.next):
      return head
    cur = head
    lens = 0
    while cur and cur.next:
      lens += 1
      cur = cur.next
    cur.next = head
    m = lens - (k % lens)
    cur = head
    while m > 0:
      cur = cur.next
      m -= 1
    new_head = cur.next
    cur.next = None
    return new_head
  
  def rotateList_way2(self, head, k):
    dummy_node = ListNode(-1)
    prev_node = self.tailKNode(dummy_node, k + 1)
    cur = head
    while cur:
      cur = cur.next
    cur.next = head
    new_head = prev_node.next
    prev_node.next = None
    return new_head
  
# ================================================================================================================================
class TreeNode:
  def __init__(self, val=-1, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

class TreeNodeSolution:
    # def preorderTravel(self, root): # 1 前序遍历
    # def inorderTravel(self, root): # 2 中序遍历
    # def postorderTravel(self, root): # 3 后序遍历
    # def levelTravel(self, root): # 4 层序遍历
    # def zigzagTravel(self, root): # 5 zigzag遍历
    # def widthofTree(self, root): # 6 树的宽度
    # def rightSideTree(self, root): # 7 二叉树的右视图
    # def isCompleteTree(self, root): # 8 二叉树完整性检验
    # def buildInPreTree(self, inorder, preorder): # 9 从前序和中序遍历结果构建二叉树
    # def buildInPostTree(self, inorder, postorder): # 10 从后序和中序遍历结果构建二叉树
    # def maxDepth(self, root): # 11 最大深度
    # def minDepth(self, root): # 12 最小深度
    # def isBalance(self, root): # 13 是否是平衡树
    # def diameterTree(self, root): # 14 二叉树最大直径
    # def maxPathSum(self, root): # 15 根节点到叶子节点最大路径和
    # def hasPathSum(self, root, targetSum): # 16 112.路径总和
    # def pathSum(self, root, targetSum): # 17 113.路径总和 II
    # def lowestCommenAncestor(self, root, p, q):  # 18 236.二叉树的最近公共祖先
    # def inverTree(self, root): # 19 226.翻转二叉树
    # def isduicheng(self, root): # 20 101.对称二叉树
    # def isSameTree(self, p, q):  # 21 100.相同的树
    # def isSubTree(self, root, subroot): # 22 572.另一棵树的子树
    # def preorderSerialize(self, root): # 23  297.二叉树的序列化与反序列化-前序
    # def preorderDeserialize(self, root): # 24 297.二叉树的序列化与反序列化-前序
    # def postorderSerialize(self, root): # 25 297.二叉树的序列化与反序列化-后序
    # def postorderDeserialize(self, root): # 26 297.二叉树的序列化与反序列化-后序
    # def flatten(self, root): # 27 114. 二叉树展开为链表
    # def isValidBST(self, root): # 28 98. 验证二叉搜索树
    # def verifyTreeOrder(self, postorder): # 29  LCR 152.验证二叉搜索树的后序遍历序列
    # def deleteNode(self, root, key): # 30 450.删除二叉搜索树中的节点
    # def findLargetstKNode(self, root, k): # 31  二叉搜索树中第K大的元素
    # def findSmallestKNode(self, root, k): # 32 230.二叉搜索树中第K小的元素
    # def longestIncreasingPath(self, matrix): # 33 0329.矩阵中的最长递增路径
    # def numIslands(self, grid): # 34 200.岛屿数量
    # def maxAreaOfIsland(self, grid): # 35 lcr105岛屿最大面积
    # def sumNumbers(self, root): # 36 129.求根节点到叶节点数字之和
    # def canFinish(self, numCourse, prepre): # 37 207.课程表
    # def canFinish2(self, numCourse, prepre): # 38 210.课程表 II
    # def numDifferentBinaryTree(self, n): # 39 0096.不同的二叉搜索树

    def preorderTravel(self, root): # 1 前序遍历
        if not root:
            return []
        return [root.val] + self.preorderTravel(root.left) + self.preorderTravel(root.right)
    def inorderTravel(self, root): # 2 中序遍历
        if not root:
            return []
        return self.preorderTravel(root.left) + [root.val] + self.preorderTravel(root.right)
    def postorderTravel(self, root): # 3 后序遍历
        if not root:
            return []
        return self.preorderTravel(root.left) + self.preorderTravel(root.right) + [root.val]
    def levelTravel(self, root): # 4 层序遍历
        if not root:
            return []
        res = []
        q = [root]
        while q:
            num_level = len(q)
            level = []
            for i in range(num_level):
                cur = q.pop(0)
                level.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            if level:
                res.append(level)
        return res
    def zigzagTravel(self, root): # 5 zigzag遍历
        if not root:
            return []
        res = []
        q = [root]
        isOdd = True
        while q:
            num_level = len(q)
            level = []
            for i in range(num_level):
                cur = q.pop(0)
                level.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            if level:
                if isOdd:
                    res.append(level)
                else:
                    res.append(level[::-1])
            isOdd = not isOdd
        return res
    def widthofTree(self, root): # 6 树的宽度
        if not root:
            return 0
        q = [[root, 0]]
        res = 1
        while q:
            res = max(res, q[-1][1] - q[0][1] + 1)
            num_level = len(q)
            for i in range(num_level):
                cur = q.pop()
                if cur[0].left:
                    q.append([cur[0].left, 2 * cur[1] + 1])
                if cur[0].right:
                    q.append([cur[0].right, 2 * cur[1] + 2])
        return res
    def rightSideTree(self, root): # 7 二叉树的右视图
        if not root:
            return []
        res = []
        q = [root]
        while q:
            num_level = len(q)
        for i in range(num_level):
            cur = q.pop(0)
            if i == num_level - 1:
                res.append(cur.val)
            if cur.left:
                q.append(cur.left)
            if cur.right:
                q.append(cur.right)
        return res
    def isCompleteTree(self, root): # 8 二叉树完整性检验
        if not root:
            return False
        q = [root]
        is_empty = False
        while q:
            num_level = len(q)
            for _ in range(num_level):
                cur = q.pop(0)
                if not cur:
                    is_empty = True
                else:
                    if is_empty:
                        return False
                    q.append(cur.left)
                    q.append(cur.right)
        return True
    def buildInPreTree(self, inorder, preorder): # 9 从前序和中序遍历结果构建二叉树
        if not inorder or not preorder:
            return None
        root_val = preorder[0]
        index_x = inorder.index(root_val)
        root = TreeNode(root_val)
        root.left = self.buildInPreTree(inorder[:index_x], preorder[1:index_x + 1])
        root.right = self.buildInPreTree(inorder[index_x + 1:], preorder[index_x + 1:])
        return root
    def buildInPostTree(self, inorder, postorder): # 10 从后序和中序遍历结果构建二叉树
        if not inorder or not postorder:
            return None
        root_val = postorder[-1]
        index_x = inorder.index(root_val)
        root = TreeNode(root_val)
        root.left = self.buildInPostTree(inorder[:index_x], postorder[:index_x])
        root.right = self.buildInPostTree(inorder[index_x + 1:], postorder[index_x:-1])
        return root
    def maxDepth(self, root): # 11 最大深度
        if not root:
           return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
    def minDepth(self, root): # 12 最小深度
        if not root:
           return 0
        if not root.left:
           return self.minDepth(root.right) + 1
        if not root.right:
           return self.minDepth(root.left) + 1
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
    def isBalance(self, root): # 13 是否是平衡树
        def height(root):
            if not root:
                return 0
            if self.is_balance:
                return 0
            left_height = height(root.left)
            right_height = height(root.right)
            if abs(left_height - right_height) > 1:
                self.balance = False
                return 0
            return max(left_height, right_height) + 1
        self.is_balance = True
        height(root)
        return self.is_balance
    def diameterTree(self, root): # 14 二叉树最大直径
        def dfs(root):
            if not root:
                return 0
            left_d = dfs(root.left)
            right_d = dfs(root.right)
            self.maxx = max(self.maxx, left_d + right_d)
            return max(left_d, right_d) + 1
        self.maxx = 0
        dfs(root)
        return self.maxx
    def maxPathSum(self, root): # 15 根节点到叶子节点最大路径和
        def dfs(root):
            if not root:
                return 0
            left_sum = max(0, dfs(root.left))
            right_sum = max(0, dfs(root.right))
            self.max_sum = max(left_sum + right_sum + root.val, self.max_sum)
            return max(left_sum, right_sum) + root.val
        self.max_sum = 0
    def hasPathSum(self, root, targetSum): # 16 
        if not root:
            return False
        if not root.left and root.right:
            return root.val == targetSum
        left_res = self.hasPathSum(root.left, targetSum - root.val)
        right_res = self.hasPathSum(root.right, targetSum - root.val)
        return left_res or right_res
    def pathSum(self, root, targetSum): # 17
        def track(root, targetsum, path):
            if not root:
                return False
            path.append(root.val)
            if not root.left and (not root.right) and root.val == targetSum:
                self.res.append(path[:])
            track(root.left, targetsum - root.val, path)
            track(root.right, targetsum - root.val, path)
            path.pop()
        self.res = []
        track(root, targetSum, [])
        return self.res
    def lowestCommenAncestor(self, root, p, q):  # 18
        if not root:
            return None
        if root == p or root == q:
            return root
        ancestor_left = self.lowestCommenAncestor(root.left, p, q)
        ancestor_right = self.lowestCommenAncestor(root.right, p, q)
        if not ancestor_left:
            return ancestor_right
        elif not ancestor_right:
            return ancestor_left
        else:
            return root
    def inverTree(self, root): # 19
        if not root:
            return None
        left_ = self.inverTree(root.left)
        right_ = self.inverTree(root.right)
        root.left = right_
        root.right_ = left_
        return root
    def isduicheng(self, root): # 20
        def check(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            return left.val == right.val and check(left.left, right.right) and check(left.right, right.left)
        if not root:
            return True
        return check(root.left, root.right)
    def isSameTree(self, p, q):  # 21
        if not p and not q:
            return True
        if not p or not q:
            return False
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    def isSubTree(self, root, subroot): # 22
        if not root:
            return False
        if self.isSameTree(root, subroot):
            return True
        return self.isSubTree(root.left, subroot) or self.isSubTree(root.right, subroot)
    def preorderSerialize(self, root): # 23 297. 二叉树的序列化与反序列化-前序遍历
        if not root:
            return "NULL"
        return root.val + "," + self.preorderSerialize(root.left) + "," + self.preorderSerialize(root.right)
    def preorderDeserialize(self, data): # 24 297. 二叉树的序列化与反序列化-前序遍历
        def helper(datalist):
            root_val = datalist.pop(0)
            if root_val == "NULL":
                return None
            root = TreeNode(root_val)
            root.left = helper(datalist)
            root.right = helper(datalist)
            return root
        datalist = data.split(",")
        return helper(datalist)
    def postorderSerialize(self, root): # 25 297. 二叉树的序列化与反序列化-后序遍历
        if not root:
            return "NULL"
        return self.preorderSerialize(root.left) + "," + self.preorderSerialize(root.right)+ "," + root.val
    def postorderDeserialize(self, data): # 26 297. 二叉树的序列化与反序列化-后序遍历
        def helper(datalist):
            root_val = datalist.pop()
            if root_val == "NULL":
                return None
            root = TreeNode(root_val)
            root.right = helper(datalist)
            root.left = helper(datalist)
            return root
        datalist = data.split(",")
        return helper(datalist)
    def flatten(self, root): # 27 114. 二叉树展开为链表
        if not root:
            return None
        self.flatten(root.left)
        self.flatten(root.right)
        temp_right = root.right
        root.right = root.left
        root.left = None
        cur = root.right
        if cur:
            while cur.right:
                cur = cur.right
            cur.right = temp_right
        else:
            root.right = temp_right
        return root
    def isValidBST(self, root): # 28 98. 验证二叉搜索树
        def dfs(root, minv, maxv):
            if not root:
                return True
            if minv < root.val < maxv:
                return dfs(root.left, minv, root.val) and dfs(root.right, root.val, maxv)
            return False
        return dfs(root, float('-inf'), float('inf'))
    def verifyTreeOrder(self, postorder): # 29  LCR 152. 验证二叉搜索树的后序遍历序列
        def dfs(left, right):
            if left >= right:
                return True
            index = left
            while postorder[index] < postorder[right]:
                index += 1
            mid = index
            while postorder[index] > postorder[right]:
                index += 1
            return index == right and dfs(left, mid - 1) and dfs(mid, right - 1)
        if len(postorder) <= 2:
            return True
        return dfs(0, len(postorder) - 1)
    def deleteNode(self, root, key): # 30 450.删除二叉搜索树中的节点
        if not root:
            return None
        if root.val < key:
            root.right =  self.deleteNode(root.right, key)
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            else:
                cur = root.right
                while cur.left:
                    cur = cur.left
                root.val = cur.val
                root.right = self.deleteNode(root.right, cur.val)
        return root
    def findLargetstKNode(self, root, k): # 31  二叉搜索树中第K大的元素
        def dfs(root):
            if not root:
                return None
            dfs(root.right)
            self.k -= 1
            if self.k == 0:
                self.res = root.val
                return
            dfs(root.left)
            return 
        self.res = 0
        self.k = k
        dfs(root)
        return self.res
    def findSmallestKNode(self, root, k): # 32 230.二叉搜索树中第K小的元素
        def dfs(root):
            if not root:
                return None
            dfs(root.left)
            self.k -= 1
            if self.k == 0:
                self.res = root.val
                return
            dfs(root.right)
            return
        self.res = 0
        self.k = k
        dfs(root)
        return self.res
    def longestIncreasingPath(self, matrix): # 33 0329.矩阵中的最长递增路径
        def dfs(row, col):
            if cache[row][col] != 0:
                return cache[row][col]
            direct = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            cur_len = 1
            for i in direct:
                new_row, new_col = row + i[0], col + i[1]
                if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]) and matrix[new_row][new_col] > matrix[row][col]
                    cur_len = max(dfs(new_row, new_col) + 1, cur_len)
            return cur_len
        cache = [[0 for _ in range(matrix[0])] for _ in range(len(matrix))]
        max_res = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0]))
                max_res = max(max_res, dfs(i, j))
        return max_res
    def numIslands(self, grid): # 34 200.岛屿数量
        def dfs(row, col):
            if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] == "0":
                return
            grid[row][col] = "0"
            directs = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            for i in directs:
                new_row, new_col = row + i[0], col + i[1]
                dfs(new_row, new_col)
        counts = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    dfs(i, j)
                    counts += 1
        return counts
    def maxAreaOfIsland(self, grid): # 35 695. 岛屿的最大面积
        def dfs(row, col):
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]) or grid[row][col] == 0:
                return 0
            grid[row][col] = 0
            cur_count = 1
            directs = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            for i in directs:
                new_row, new_col = row + i[0], col + i[1]
                cur_count += dfs(new_row, new_col)
            return cur_count
        max_res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    max_res = max(max_res, dfs(i, j))
        return max_res
    def sumNumbers(self, root): # 36 129.求根节点到叶节点数字之和
        def dfs(total, root):
            if not root:
                return 0
            cur_total = 10 * total + root.val
            if not root.left and not root.right:
                return cur_total
            return dfs(cur_total, root.left) + dfs(cur_total, root.right)
        return dfs(0, root)
    def canFinish(self, numCourses, prerequisites): # 37 207.课程表
        from collections import defaultdict
        out_dic = defaultdict(list)
        in_dic = [0 for i in range(numCourses)]
        for x in prerequisites:
            out_dic[x[1]].append(x[0])
            in_dic[x[0]] += 1
        q = [x for x in range(len(in_dic)) if in_dic[x] == 0]
        while q:
            m = q.pop(0)
            for j in out_dic[m]:
                in_dic[j] -= 1
                if in_dic[j] == 0:
                    q.append(j)
        if sum(in_dic) == 0:
            return True
        return False
    def canFinish2(self, numCourses, prerequisites): # 38 210.课程表 II
        from collections import defaultdict
        out_dic = defaultdict(list)
        in_dic = [0 for i in range(numCourses)]
        for x in prerequisites:
            out_dic[x[1]].append(x[0])
            in_dic[x[0]] += 1
        q = [x for x in range(len(in_dic)) if in_dic[x] == 0]
        res = []
        while q:
            m = q.pop(0)
            res.append(m)
            for j in out_dic[m]:
                in_dic[j] -= 1
                if in_dic[j] == 0:
                    q.append(j)
        if sum(in_dic) == 0:
            return res
        return []
    def numDifferentBinaryTree(self, n): # 39 0096.不同的二叉搜索树
        # dp[i]:i个节点能构成的二叉搜索树的数量 = 求和 dp[j - 1] * dp[i - j]
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        return dp[n]

