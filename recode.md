

```python
def search():
  print("Hello, world!" )
  return
```





```c++ 
#include <iostream>
int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
```



# h1二分搜索

MARKDOWN语法 https://markdown.com.cn/basic-syntax/htmls.html

### 不大于target的数

#### 涉及题目：二维数组的搜索

```python
def search_left_bound(self, arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right if right >= 0 else -1
```

### 左边界&右边界

```python
def right_loc(self, nums, target):
    # 右边界
    left, right = 0, len(nums) - 1
    while left <= right:  # 闭区间
        mid = left + (right - left) // 2
        if nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    if left < 0 or nums[right] != target:
        return -1
    return right

def left_loc(self, nums, target):
    # 左边界
    left, right = 0, len(nums) - 1
    while left <= right:  # 闭区间
        
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    if left > len(nums) or nums[left] != target:
        return -1
    return left

```

# H1 二叉树



### 144. 二叉树的前序遍历
解析：两种方法：①递归②模拟栈（while True)

[144. 二叉树的前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/submissions/494183488/  144. 二叉树的前序遍历)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # # 方法一：递归
        # res = []
        # def preorder(root):
        #     if not root:
        #         return 
        #     res.append(root.val)
        #     preorder(root.left)
        #     preorder(root.right)
        
        # preorder(root)
        # return res

        # # 方法二：显示栈
        # if not root:
        #     return []
        # res = []
        # stack = [root]
        # while stack: # 栈不为空
        #     node = stack.pop()
        #     res.append(node.val)
        #     if node.right:
        #         stack.append(node.right)
        #     if node.left:
        #         stack.append(node.left)
        # return res

        # 方法三
        if root is None:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)
```

​    

### 94. 二叉树的中序遍历
解析：两种方法：①递归②模拟栈（while True)

[94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/submissions/494183841/  94. 二叉树的中序遍历)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # 方法一：递归
        res = []
        def preorder(root):
            if not root:
                return 
            preorder(root.left)
            res.append(root.val)
            preorder(root.right)
        
        preorder(root)
        return res

        # # 方法二：显示栈
        # if not root:
        #     return []
        # res = []
        # stack = []
        # while True: # 栈不为空
        #     while root:
        #         stack.append(root) # 根结点进栈
        #         root = root.left # 作子节点作为新节点
        #     if not stack: # 栈为空退出
        #         return res
        #     # 出栈
        #     node = stack.pop()
        #     res.append(node.val)
        #     # 右子节点作为新节点
        #     root = node.right

        # # 方法三
        # if root is None:
        #     return []
        # return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

​    

### 0145. 二叉树的后序遍历
解析：两种方法：①递归②模拟栈（while True)

[0145. 二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/submissions/530626594/  0145. 二叉树的后序遍历)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]
        
```

​    

### 102. 二叉树的层序遍历
解析：广度优先搜索deque level

[102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/submissions/494184742/  102. 二叉树的层序遍历)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            level = []
            size = len(queue)
            for _ in range(size):
                curr= queue.pop(0)
                level.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            if level:
                res.append(level)
        return res
        
```

​    

### 103. 二叉树的锯齿形层序遍历
解析：同层序遍历，增加记住奇数和偶数的层ID

[103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/submissions/494185650/  103. 二叉树的锯齿形层序遍历)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        from collections import deque
        if not root:
            return []
        queue = [root]
        res = []
        odd = True
        while queue:
            level = []
            for i in range(len(queue)):
                node = queue.pop(0)
                if odd:
                    level.append(node.val)
                else:
                    level.insert(0, node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if level:
                res.append(level)
            odd = not odd
        return res
        
```

​    

### 104. 二叉树的最大深度
解析：递归max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

[104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/submissions/494186867/  104. 二叉树的最大深度)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
        
```

​    

### 111. 二叉树的最小深度
解析：递归，但是比最大深度复杂，分别判断当前最小值和左子树、右子树进行比较

[111. 二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/submissions/528561851/  111. 二叉树的最小深度)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        left_h = self.minDepth(root.left)
        right_h = self.minDepth(root.right)
        min_depth = float('inf')
        if root.left:
            min_depth = min(min_depth, left_h)
        if root.right:
            min_depth = min(min_depth, right_h)
        return min_depth + 1
```

​    

### 662. 二叉树最大宽度
解析：因为题目要求空节点也算数，因此对节点进行编号，节点为index的点，左节点是2*index，右节点是2*index + 1计算每层宽度时，用每层节点的最大编号减去最小编号再加 111 即为宽度

[662. 二叉树最大宽度](https://leetcode.cn/problems/maximum-width-of-binary-tree/submissions/495578124/  662. 二叉树最大宽度)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def widthOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # # 方法1 广度优先搜索
        # if not root:
        #     return []
        # queue = [[root, 0]]
        # anx = 0
        # while queue:
        #     anx = max(anx, queue[-1][1] - queue[0][1] + 1)
        #     size = len(queue)
        #     for i in range(size):
        #         curr, index = queue.pop(0)
        #         if curr.left:
        #             queue.append([curr.left, 2 * index + 1])
        #         if curr.right:
        #             queue.append([curr.right, 2 * index + 2])
        # return anx


        # 方法2 深度优先搜索
        def dfs(node, depth, index, level_info):
            if not node:
                return 
            if depth == len(level_info):
                level_info.append([index, index])
            else:
                level_info[depth][0] = min(level_info[depth][0], index)
                level_info[depth][1] = max(level_info[depth][1], index)
            dfs(node.left, depth+1, index*2+1, level_info)
            dfs(node.right, depth+1, index*2+2, level_info)

        level_info = []
        dfs(root, 0, 0, level_info)
        return max(x[1] - x[0] + 1 for x in level_info)
```

​    

### 124. 二叉树中的最大路径和
解析：配置dfs函数返回每个节点的最大贡献值；中间计算当前包含左右根的新路径 new_path = root.val + leftGain + rightGain

[124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/submissions/494410856/  124. 二叉树中的最大路径和)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def dfs(root):
            if not root:
                return 0
            left = max(0, dfs(root.left))
            right = max(0, dfs(root.right))
            ret = root.val + max(left, right)
            lmr = root.val + left + right
            self.maxx = max(self.maxx, lmr)
            return ret
        # self.maxx = float("-inf")
        self.maxx = root.val
        dfs(root)
        return self.maxx
```

​    

### 101. 对称二叉树
解析：递归校验左子树和又子树，递归self.check(left.left, right.right) and self.check(left.right, right.left)：如果一棵二叉树是对称的，左子树和右子树的外侧节点的节点值相等，并且其左子树和右子树的内侧节点的节点值也相等

[101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/submissions/494405545/  101. 对称二叉树)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self.check(root.left, root.right)
    def check(self, left, right):
        if (not left) and (not right):
            return True
        elif ((not left) and right):
            return False
        elif ((not right) and left):
            return False
        elif left.val != right.val:
            return False
        return self.check(left.left, right.right) and self.check(left.right, right.left)
```

​    

### 112. 路径总和
解析：返回结果是是否满足这样条件的路径，True or  False

[112. 路径总和](https://leetcode.cn/problems/path-sum/submissions/494398025/  112. 路径总和)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """
        if not root:
            return False
        if (root.left == None) and (root.right == None):
            return targetSum == root.val
        left_res = self.hasPathSum(root.left, targetSum-root.val)
        right_res = self.hasPathSum(root.right, targetSum-root.val)
        return left_res or right_res
        
```

​    

### 113. 路径总和 II
解析：返回结果是所有满足这样条件的路径，具体的路径；这道题有点像回溯，path中先把当前节点加入，递归调用当前节点的左边和右边子树；递归完成后再pop掉当前节点。递归过程中若满足条件，则结果集合中加入path[:]（浅拷贝）

[113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/submissions/530896995/  113. 路径总和 II)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        res = []
        path = []
        def dfs(root, sums):
            if not root:
                return 
            path.append(root.val)
            if (not root.left) and (not root.right) and (sums == root.val):
                res.append(path[:])
            dfs(root.left, sums-root.val)
            dfs(root.right, sums-root.val)
            path.pop()
        dfs(root, targetSum)
        return res
```

​    

### 236. 二叉树的最近公共祖先
解析：自下而上，找到左右节点为父节点的子树均满足条件的（包含当前p或者q)

[236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/submissions/494186470/  236. 二叉树的最近公共祖先)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root == p or root == q:
            return root
        if root:
            node_left = self.lowestCommonAncestor(root.left, p, q)
            node_right = self.lowestCommonAncestor(root.right, p, q)
            if node_left and node_right:
                return root
            elif not node_left:
                return node_right
            else:
                return node_left
        return None
        
```

​    

### 0199. 二叉树的右视图
解析：使用广度优先搜索对二叉树进行层次遍历。在遍历每层节点的时候，只需要将最后一个节点加入结果数组即可

[0199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/submissions/530900908/  0199. 二叉树的右视图)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            level = []
            size = len(queue)
            for i in range(size):
                curr= queue.pop(0)
                level.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            res.append(level[-1])
            # if i == size - 1:
            #     res.append(curr.val)
        return res
```

​    

### 226. 翻转二叉树
解析：自下而上，翻转；root.left = right  root.right = left

[226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/submissions/494411944/  226. 翻转二叉树)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        root.left = right
        root.right = left
        return root
        
```

​    

### 958. 二叉树的完全性检验
解析：自上而下，每次加入一层，如果在遍历过程中在遇到第一个空节点之后，又出现了非空节点，则该二叉树不是完全二叉树。维护一个布尔变量 is_empty 用于标记是否遇见了空节点。

[958. 二叉树的完全性检验](https://leetcode.cn/problems/check-completeness-of-a-binary-tree/submissions/530902587/  958. 二叉树的完全性检验)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isCompleteTree(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return False
        queue = [root]
        # 如果在遍历过程中在遇到第一个空节点之后，又出现了非空节点，则该二叉树不是完全二叉树。
        # 维护一个布尔变量 is_empty 用于标记是否遇见了空节点。
        is_empty = False
        while queue:
            size = len(queue)
            for _ in range(size):
                cur = queue.pop(0)
                if not cur:
                    is_empty = True
                else:
                    if is_empty:
                        return False
                    queue.append(cur.left)
                    queue.append(cur.right)
        return True
        
```

​    

### 572. 另一棵树的子树
解析：如果找到val相等的节点，就同步判断这个点的子树是否相等

[572. 另一棵树的子树](https://leetcode.cn/problems/subtree-of-another-tree/submissions/495582974/  572. 另一棵树的子树)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubtree(self, root, subRoot):
        """
        :type root: TreeNode
        :type subRoot: TreeNode
        :rtype: bool
        """
        def check(root, subRoot):
            if not root and not subRoot:
                return True
            if (not root and subRoot) or (root and not subRoot) or (root.val != subRoot.val):
                return False
            return check(root.left, subRoot.left) and check(root.right, subRoot.right)
        
        def dfs(root, subRoot):
            if not root:
                return False
            # if root.val == subRoot.val:
            #     return check(root, subRoot)
            return check(root, subRoot) or dfs(root.left, subRoot) or dfs(root.right, subRoot)
        return dfs(root, subRoot)
```

​    

### 100. 相同的树
解析：若两数相同，则root节点相同&节点值也相等&递归结果也相等

[100. 相同的树](https://leetcode.cn/problems/same-tree/submissions/528677991/  100. 相同的树)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

​    

### 111. 二叉树的最小深度
解析：先判断是否为空树、只有一个节点的树

[111. 二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/submissions/528561851/  111. 二叉树的最小深度)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        left_h = self.minDepth(root.left)
        right_h = self.minDepth(root.right)
        min_depth = float('inf')
        if root.left:
            min_depth = min(min_depth, left_h)
        if root.right:
            min_depth = min(min_depth, right_h)
        return min_depth + 1
```

​    

### LCR 151. 彩灯装饰记录 III
解析：同103题，增加奇数和偶数行判定

[LCR 151. 彩灯装饰记录 III](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/submissions/529046357/  LCR 151. 彩灯装饰记录 III)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def decorateRecord(self, root: Optional[TreeNode]) -> List[List[int]]:
        # 层序遍历，使用变量记住当前的行数是奇数还是偶数
        if not root:
            return []
        queue = [root]
        res = []
        is_odd = True
        while queue:
            cur_level = []
            level_size = len(queue)
            for _ in range(level_size):
                curr = queue.pop(0)
                cur_level.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            if is_odd:
                res.append(cur_level)
            else:
                res.append(cur_level[::-1])
            is_odd = not is_odd
        return res
```

​    

### 207.课程表
解析：方法1：广度优先搜索，从入度为0的节点出发，【从低到高】；方法2：深度优先搜索，从出度为0的节点出发，【从高到低，最后将结果翻转】

[207.课程表](https://leetcode.cn/problems/course-schedule/submissions/529043299/  207.课程表)

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # # 方法1：广度优先搜索，从入度为0的节点出发，【从低到高】
        # from collections import defaultdict
        # # 记录每个节点的出度(备注：这里的默认值必须写list)
        # out_edg = defaultdict(list) 
        # # 记录每个节点的入度
        # in_deg = [0] * numCourses
        # for i in prerequisites:
        #     out_edg[i[1]].append(i[0])
        #     in_deg[i[0]] += 1
        # # 把入度为0的节点加入队列
        # q = [i for i in range(numCourses) if in_deg[i] == 0]
        # # 判断最后循环结束时，res是否等于课程数（无环）
        # res = 0
        # while q:
        #     res += 1
        #     i = q.pop(0)
        #     for v in out_edg[i]:
        #         # 入度为当前节点的节点入度减一，若为0，加入队列
        #         in_deg[v] -= 1
        #         if in_deg[v] == 0:
        #             q.append(v)
        # return res == numCourses


        # 方法2：深度优先搜索，从出度为0的节点出发，【从高到低，最后将结果翻转】
        # 记录三个状态，0：未搜索、1：搜索中、2：已搜索
        from collections import defaultdict
        out_edg = defaultdict(list)
        for i in prerequisites:
            out_edg[i[1]].append(i[0])  # key是入度，value是所有出度
        visited = [0] * numCourses
        result = []
        valid = True

        # 从任意一个出度为0的点出发，dfs函数
        def dfs(u):
            # nonlocal关键字用于在嵌套函数内部修改外层函数的变量值
            nonlocal valid
            visited[u] = 1
            for v in out_edg[u]:  # 只要发现有环，立刻停止搜索
                if visited[v] == 0:  # 如果「未搜索」那么搜索相邻节点
                    dfs(v)
                    if not valid:
                        return
                elif visited[v] == 1:  # 如果「搜索中」说明找到了环
                    valid = False
                    return
            visited[u] = 2  # 将节点标记为「已完成」
            result.append(u)  # 将节点入栈
        for i in range(numCourses):  # 每次挑选一个「未搜索」的节点，开始进行深度优先搜索
            if valid and not visited[i]:
                dfs(i)
        return valid and len(result) == numCourses

```

​    

### 210. 课程表 II
解析：广度优先搜索，从入度为0的节点出发，【从低到高】，深度优先搜索也可以，就是麻烦一点。

[210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii/submissions/529043183/  210. 课程表 II)

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # # 方法1：广度优先搜索，从入度为0的节点出发，【从低到高】
        # from collections import defaultdict
        # # 记录每个节点的出度(备注：这里的默认值必须写list)
        # out_edg = defaultdict(list) 
        # # 记录每个节点的入度
        # in_deg = [0] * numCourses
        # for i in prerequisites:
        #     out_edg[i[1]].append(i[0])
        #     in_deg[i[0]] += 1
        # # 把入度为0的节点加入队列
        # q = [i for i in range(numCourses) if in_deg[i] == 0]
        # # 判断最后循环结束时，res是否等于课程数（无环）
        # res = []
        # while q:
        #     i = q.pop(0)
        #     res.append(i)
        #     for v in out_edg[i]:
        #         # 入度为当前节点的节点入度减一，若为0，加入队列
        #         in_deg[v] -= 1
        #         if in_deg[v] == 0:
        #             q.append(v)
        # return res if len(res) == numCourses else []


        # 方法2：深度优先搜索，从出度为0的节点出发，【从高到低，最后将结果翻转】
        # 记录三个状态，0：未搜索、1：搜索中、2：已搜索
        from collections import defaultdict
        out_edg = defaultdict(list)
        for i in prerequisites:
            out_edg[i[1]].append(i[0])  # key是入度，value是所有出度
        visited = [0] * numCourses
        result = []
        valid = True

        # 从任意一个出度为0的点出发，dfs函数
        def dfs(u):
            # nonlocal关键字用于在嵌套函数内部修改外层函数的变量值
            nonlocal valid
            visited[u] = 1
            for v in out_edg[u]:  # 只要发现有环，立刻停止搜索
                if visited[v] == 0:  # 如果「未搜索」那么搜索相邻节点
                    dfs(v)
                    if not valid:
                        return
                elif visited[v] == 1:  # 如果「搜索中」说明找到了环
                    valid = False
                    return
            visited[u] = 2  # 将节点标记为「已完成」
            result.append(u)  # 将节点入栈
        for i in range(numCourses):  # 每次挑选一个「未搜索」的节点，开始进行深度优先搜索
            if valid and not visited[i]:
                dfs(i)

        return result[::-1] if valid and len(result) == numCourses else []


```

​    

### 200. 岛屿数量
解析：深度优先搜索和广度优先搜索均可以

[200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/submissions/494654775/  200. 岛屿数量)

```python
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        # # 方法1 dfs
        # def dfs(grid, i, j):
        #     if i < 0 or j<0 or i >= len(grid) or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == "0":
        #         return
        #     grid[i][j] = '0'
        #     dfs(grid, i-1, j)
        #     dfs(grid, i+1, j)
        #     dfs(grid, i, j-1)
        #     dfs(grid, i, j+1)
        # count = 0
        # for i in range(len(grid)):
        #     for j in range(len(grid[0])):
        #         if grid[i][j] == '1':
        #             dfs(grid, i, j)
        #             count += 1
        # return count

        # 方法2 bfs
        def bfs(grid, i, j):
            queue = [[i, j]]
            while queue:
                [i, j] = queue.pop(0)
                if i < len(grid) and i >= 0 and j < len(grid[0]) and j >= 0 and grid[i][j] == "1":
                    grid[i][j] = '0'
                    queue += [[i+1, j], [i-1, j], [i, j-1], [i, j+1]]
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '0':
                    continue
                bfs(grid, i, j)
                count += 1
        return count
```

 # H1动态规划   

### 322. 零钱兑换
解析：广度优先搜索（最短路径和的问题） 和 动态规划【dp[i]表示凑成金额i最少的硬币数量】。时间复杂度O(amount*size), 空间复杂度O(amount)

[322. 零钱兑换](https://leetcode.cn/problems/coin-change/submissions/498157075/  322. 零钱兑换)

```python
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        ## 方法1 广度优先搜索
        # amount 不为 0，初始化 visited 为 {11}，queue 为 [11]，step 为 0。
        # 进入主循环，queue 不为空。
        # step 加 1，变为 1。队列中有 1 个元素，即 cur = 11。
        # 遍历硬币，尝试用面值为 1 的硬币，得到新金额 cur - coin = 10，加入队列和集合。尝试用面值为 2 的硬币，得到新金额 cur - coin = 9，加入队列和集合。尝试用面值为 5 的硬币，得到新金额 cur - coin = 6，加入队列和集合。此时，visited 为 {11, 10, 9, 6}，queue 为 [10, 9, 6]。
        # 进入下一轮循环，step 加 1，变为 2。队列中有 3 个元素，分别处理 cur = 10, 9, 6。
        # 对于 cur = 10，尝试用面值为 1、2、5 的硬币，得到新金额分别为 9、8、5，其中 9 已经在集合中，将 8 和 5 加入队列和集合。
        # 对于 cur = 9，尝试用面值为 1、2、5 的硬币，得到新金额分别为 8、7、4，其中 8 已经在集合中，将 7 和 4 加入队列和集合。
        # 对于 cur = 6，尝试用面值为 1、2、5 的硬币，得到新金额分别为 5、4、1，其中 5 和 4 已经在集合中，将 1 加入队列和集合。此时，visited 为 {11, 10, 9, 6, 8, 5, 7, 4, 1}，queue 为 [8, 5, 7, 4, 1]。
        # 进入下一轮循环，step 加 1，变为 3。队列中有 5 个元素，分别处理 cur = 8, 5, 7, 4, 1。
        # 对于 cur = 8，尝试用面值为 1、2、5 的硬币，得到新金额分别为 7、6、3，其中 7 和 6 已经在集合中，将 3 加入队列和集合。
        # 对于 cur = 5，发现它等于面值为 5 的硬币，所以找到了一种硬币组合，返回 step，即 3。
        # 因此，凑成总金额 11 所需的最少硬币个数为 3（一个面值为 5 的硬币和两个面值为 1 的硬币）。

        # if amount == 0:
        #     return 0
        # visited = set([amount])
        # queue = [amount]
        # step = 0
        # while queue:
        #     step += 1
        #     size = len(queue)
        #     for _ in range(size):
        #         cur = queue.pop(0)
        #         for coin in coins:
        #             if coin == cur:
        #                 return step
        #             elif cur > coin and cur - coin not in visited:
        #                 queue.append(cur-coin)
        #                 visited.add(cur-coin)
        # return -1

        # 动态规划
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i < coin:
                    continue
                dp[i] = min(dp[i], dp[i - coin] + 1)
        if dp[amount] != amount + 1:
            return dp[amount]
        else:
            return -1
```

​    

### 518. 零钱兑换 II
解析：定义状态 dp[i] 表示为：凑成总金额为 i 的方案总数。# # 状态转移方程：dp[i] = 不使用当前coin的方案dp[i]+使用当前coin的方案dp[i - coin]

[518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/submissions/531704251/  518. 零钱兑换 II)

```python
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        # # 方法1：使用前i枚硬币凑金额j有多少种方案
        # dp[i][j]:只使用coins种的前i个硬币的面值,想凑出金额j,有dp[i][j]种凑法
        # dp = [[0 for _ in range(amount + 1)] for _ in range(len(coins) + 1)]
        # # dp[coins][amount]
        # for i in range(len(coins) + 1): # 凑出金额0值有：什么都不用做这1种方法
        #     dp[i][0] = 1
        
        # for i in range(1, len(coins) + 1):
        #     for j in range(1, amount + 1):
        #         if j >= coins[i - 1]: # 钱大于等于硬币
        #             dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]]
        #         else:
        #             dp[i][j] = dp[i - 1][j]
        # return dp[-1][-1]

        # 方法2：使用前i枚硬币凑金额j有多少种方案 - 调换循环顺序
        # dp = [[0 for _ in range(amount + 1)] for _ in range(len(coins) + 1)]
        # # dp[coins][amount]
        # for i in range(len(coins) + 1): # 凑出金额0值有：什么都不用做这1种方法
        #     dp[i][0] = 1
        
        # for i in range(1, amount + 1):
        #     for j in range(1, len(coins) + 1):
        #         if i >= coins[j - 1]: # 钱大于等于硬币
        #             dp[j][i] = dp[j - 1][i] + dp[j][i - coins[j - 1]]
        #         else:
        #             dp[j][i] = dp[j - 1][i]
        # return dp[-1][-1]

        # 方法12整理：
        dp = [[0 for _ in range(len(coins) + 1)] for _ in range(amount + 1)]
        for i in range(len(coins) + 1):
            dp[0][i] = 1  # 凑出金额0只有什么都不做这种做法
        for i in range(1, amount + 1):
            for j in range(1, len(coins) + 1):
                if i >= coins[j - 1]:  # 因为j是从1开始的，所以j - 1是从coins的索引0位置开始进行的
                    dp[i][j] = dp[i][j - 1] + dp[i - coins[j - 1]][j]
                else:
                    dp[i][j] = dp[i][j - 1]
        return int(dp[amount][len(coins)])

        # # 方法3：状态压缩
        # # dp[i]：对于硬币从 0 到 k，我们必须使用第k个硬币的时候，凑成金额i的方案
        # # 定义状态 dp[i] 表示为：凑成总金额为 i 的方案总数。
        # # 状态转移方程：dp[i] = 不使用当前coin的方案dp[i]+使用当前coin的方案dp[i - coin]
        # dp = [0 for _ in range(amount + 1)]
        # dp[0] = 1
        # for coin in coins:
        #     for i in range(1, amount + 1):
        #         if i >= coin:
        #             dp[i] = dp[i] + dp[i - coin]
        # return dp[-1]
```

​    

### 70. 爬楼梯
解析：等同斐波那契；题目延伸到一次最多爬k个

[70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/submissions/499266429/  70. 爬楼梯)

```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        # # way1
        # # 斐波那契函数
        # dp = [0] * (n+1)
        # dp[0] = 1
        # dp[1] = 1
        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[n]

        # # way2
        # bef, cur = 1, 1
        # for i in range(2, n + 1):
        #      temp = bef + cur
        #      bef = cur
        #      cur = temp
        # return cur

        # # way 3: 题目延伸，一次最多爬k个楼梯，有多少种爬法
        # dp = [0] * (n + 1)
        # dp[0] = 1
        # dp[1] = 1
        # steps = 2
        # for i in range(2, n + 1):
        #     for step in range(1, steps + 1):
        #         dp[i] += dp[i - step]
        # return dp[-1]

        # way 4: 进一步题目延伸，一次最多爬这三种[2, 3, 4]个楼梯，有多少种爬法
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        steps = [1, 2]
        for i in range(2, n + 1):
            for step in steps:
                if i >= step:
                    dp[i] += dp[i - step]
        return dp[-1]
```

​    

### 509. 斐波那契数
解析：自下而上；状态压缩

[509. 斐波那契数](https://leetcode.cn/problems/fibonacci-number/submissions/496640291/  509. 斐波那契数)

```python
### 方法一：dp数组迭代（自下而上）
# class Solution(object):
#     def fib(self, n):
#         """
#         :type n: int
#         :rtype: int
#         """
        # if n==0: return 0
        # if n==1: return 1
        # dp = [0]*(n+1)
        # dp[1] = 1
        # for i in range(2, n+1):
        #     dp[i] = dp[i-1] + dp[i-2]
        # return dp[n]

### 方法二：备忘录法（自上而下）
# class Solution(object):
#     def fib(self, n):
#         """
#         :type n: int
#         :rtype: int
#         """
#         memo = [0]*(n+1)
#         return self.helper(n, memo)
    
#     def helper(self, n, memo):
#         if n==0:
#             return 0
#         if n==1:
#             return 1
#         if memo[n] != 0:
#             return memo[n]
#         memo[n] = self.helper(n-1, memo) + self.helper(n-2, memo)
#         return memo[n]

### 方法三：状态压缩， 缩小dp表的大小
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0: return 0
        if n==1: return 1
        cur = 1
        prev = 0
        for i in range(2, n+1):
           prev, cur = cur, prev + cur
        return cur
```

​    

### 121.买卖股票的最佳时机
解析：注意是只能买一次，记录下当前看到的最小值和当前如果卖出的话，利润的最大值。

[121.买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/submissions/498156003/  121.买卖股票的最佳时机)

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # 一次遍历
        minPrice = 10001
        maxProfit = 0
        for i in range(len(prices)):
            if prices[i] < minPrice:
                minPrice = prices[i]
            else:
                maxProfit = max(maxProfit, prices[i] - minPrice)
        return maxProfit
```

​    

### 122.买卖股票的最佳时机 II
解析：可以买卖多次，方法1：动态规划（dp[i][0]:第i天手上没有股票的收益，dp[i][1]：第i天手上有1只股票的收益
）；方法2：贪心（maxres = maxres + prices[i] - prices[i-1]）

[122.买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/submissions/141701036/  122.买卖股票的最佳时机 II)

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        maxres = 0
        for i in range(1, len(prices)):
            if prices[i] - prices[i-1]>0:
                maxres = maxres + prices[i] - prices[i-1]
        return maxres
```

​    

### 123.买卖股票的最佳时机 III
解析：最多可以进行两次买卖.确定当前天数下，方法1：一共有几种状态（4种）buy1, sell1, buy2, sell2, 然后确定已知i-1天的状态下，当前是如何通过状态方程得到第i天的状态的；方法2：复用最多交易K次的代码（dp[n][k][0]:第n天，至今最多进行了k次交易，当前手上没有股票的收益，dp[n][k][1]:第n天，至今最多进行了k次交易，当前手上有股票的收益）

[123.买卖股票的最佳时机 III](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/submissions/520101141/  123.买卖股票的最佳时机 III)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 确定当前天数下，一共有几种状态（4种）buy1, sell1, buy2, sell2
        # 确定已知i-1天的状态下，当前是如何通过状态方程得到第i天的状态的
        buy1 = buy2 = -prices[0]
        sell1 = sell2 = 0
        for i in range(1, len(prices)):
            buy1 = max(buy1, -prices[i - 1])
            sell1 = max(sell1, buy1 + prices[i])
            buy2 = max(buy2, sell1 - prices[i])
            sell2 = max(sell2, buy2 + prices[i])
        return sell2
```

​    

### 188. 买卖股票的最佳时机 IV
解析：最多可以进行K次买卖.确定解题框架!!!最终返回 dp[n][k][0]:第n天，至今最多进行了k次交易，当前手上没有股票的收益

[188. 买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/submissions/520110019/  188. 买卖股票的最佳时机 IV)

```python
class Solution:
    def maxProfit(self, K: int, prices: List[int]) -> int:
        # 一共有N * K * 2种状态，有以下两类
        # dp[n][k][0]:第n天，至今最多进行了k次交易，当前手上没有股票的收益，最终返回dp[N][K][0]
        # 对应的状态转移方程是：dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
        # dp[n][k][1]:第n天，至今最多进行了k次交易，当前手上有股票的收益
        # 对应的状态转移方程是：dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i])
        # 注意，因为k可以为0，所以是k + 1, 因为i可以是0，所以是n + 1
        # dp = [[[0 for _ in range(len(prices) + 1)] for _ in range(K + 1)] for _ in range(2)]
        dp = [[[0] * 2 for _ in range(K + 1)] for _ in range(len(prices) + 1)]

        for i in range(0, len(prices) + 1):
            for k in range(K + 1):
                if i == 0 or k == 0:
                    dp[i][k][0] = 0
                    dp[i][k][1] = -prices[0]  # 这里是负无穷或者-prices[0]，用以迭代计算dp[1][1][0]
                else:
                    dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i - 1])
                    dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i - 1])
        return dp[-1][-1][0]
```

​    

### 309. 买卖股票的最佳时机含冷冻期
解析：注意循环是for i in range(len(prices) + 1)。有一天冷冻期，状态转移方程当天持有股票的最大收益，这个要修改一下，依赖于i-2，然后要注意处理i=1和i=2的情况

[309. 买卖股票的最佳时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/submissions/520118258/  309. 买卖股票的最佳时机含冷冻期)

```python

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 复用最多交易K次的代码，此处可以看做k=float('inf')，因此对原先的状态转移方程进行降维
        # 状态转移方程：dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        # 因为有冷冻期，dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i])
        # 注意 i = 0 和 i = 1时的特殊取值
        dp = [[0] * 2 for _ in range(len(prices) + 1)]
        for i in range(len(prices) + 1):
            if i == 0:
                dp[i][1] = -prices[0]
            elif i == 1: # 备注：这里必须是elif，三种情况只能走入一种
                dp[i][1] = max(dp[i - 1][1], -prices[i - 1])
            else:
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1])
                dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i - 1])
        return dp[-1][0]
```

​    

### 714. 买卖股票的最佳时机含手续费
解析：注意循环是for i in range(len(prices) + 1)。从每次购买股票的收益中减去这个手续费，注意初始化i=0时，dp[i][1] = -prices[0] - fee

[714. 买卖股票的最佳时机含手续费](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/submissions/520119440/  714. 买卖股票的最佳时机含手续费)

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        # 复用最多交易K次的代码，此处可以看做k=float('inf')，因此对原先的状态转移方程进行降维
        # 状态转移方程：dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        # 因为有手续费，dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee)
        dp = [[0] * 2 for _ in range(len(prices) + 1)]
        for i in range(len(prices) + 1):
            if i == 0:
                dp[i][1] = -prices[0] - fee
            else:
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1])
                dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1] - fee)
        return dp[-1][0]
```



## head2





