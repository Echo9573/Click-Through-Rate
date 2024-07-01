# 二叉树遍历
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def preorderTraversal(self, root):
        if root is None:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

    def inorderTraversal(self, root):
        if root is None:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

    def postorderTraversal(self, root):
        # way1：递归；way2:模拟栈
        if root is None:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

    def levelorderTraversal(self, root): # 层序遍历
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            queue_size = len(queue)
            level = []  # 当前这一层结果
            for _ in range(queue_size):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if level:
                res.append(level)
        return res

    def zigzagLevelOrder(self, root):
        if not root:
            return []
        queue = [root]
        res = []
        odd = True
        while queue:
            queue_size = len(queue)
            level = []  # 当前这一层结果
            for _ in range(queue_size):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if level:
                if odd:
                    res.append(level)
                else:
                    res.append(level[::-1])
            odd = not odd
        return res

    def widthOfBinaryTree(self, root):  # 二叉树的最大宽度
        # 空节点也算数
        if not root:
            return 0
        queue = [[root, 0]]
        res = 0
        while queue:
            res = max(queue[-1][1] - queue[0][1] + 1, res) # 注意，这里要+1
            queue_size = len(queue)
            for _ in range(queue_size):
                cur_node, index = queue.pop(0)
                if cur_node.left:
                    queue.append([cur_node.left, 2 * index + 1])
                if cur_node.right:
                    queue.append([cur_node.right, 2 * index + 2])
        return res

    def rightSideView(self, root): # 199. 二叉树的右视图
        # 层序遍历
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            level = []
            size = len(queue)
            for i in range(size):
                curr = queue.pop(0)
                level.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            res.append(level[-1])
        return res

    def isCompleteTree(self, root):  # 二叉树的完全性检验
        # 层序遍历
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
                    queue.append(cur.left) # 这里直接判断
                    queue.append(cur.right)
        return True

    def buildTree(self, preorder, inorder): # 105. 从前序与中序遍历序列构造二叉树
        if not preorder or not inorder:
            return None
        root_value = preorder[0]
        root = TreeNode(root_value)
        root_index = inorder.index(root_value)
        root.left = self.buildTree(preorder[1:1 + root_index], inorder[:root_index])
        root.right = self.buildTree(preorder[1 + root_index:], inorder[root_index + 1:])
        return root

    def buildTree2(self, inorder, postorder): # 106. 从中序与后序遍历序列构造二叉树
        if not postorder or not inorder:
            return None
        root_value = postorder[-1]
        root = TreeNode(root_value)
        root_index = inorder.index(root_value)
        root.left = self.buildTree(inorder[:root_index], postorder[:root_index], )
        root.right = self.buildTree(inorder[root_index + 1:], postorder[root_index:-1], )
        return root

    def maxDepth(self, root):  # 二叉树最大深度
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
    
    def isBalanced(self, root):  # 平衡二叉树：左右子树的高度差不超过1
        self.balanced = True
        def height(root):
            if not root:
                return 0
            if not self.balanced: # 发现即返回
                return 0
            left_h = height(root.left)
            right_h = height(root.right)
            if abs(left_h - right_h) > 1:
                self.balanced = False
                return 0   # 如果当前子树不平衡，直接返回，避免继续计算另一个子树的高度
            return max(left_h, right_h) + 1

        height(root)
        return self.balanced

    def minDepth(self, root):  # 二叉树最小深度
        # 从根节点到最近叶子节点的最短路径上的节点数量（要考虑没有叶子节点的情况）
        if not root:
            return 0
        if not root.left:
            return self.minDepth(root.right) + 1
        if not root.right:
            return self.minDepth(root.left) + 1
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

    def diameterOfBinaryTree(self, root):  # 二叉树最大直径（二叉树中任意两个节点路径长度中的最大值）
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            ret = 1 + max(left, right)
            lmr = left + right  # 注意这里
            self.maxx = max(self.maxx, lmr)
            return ret

        self.maxx = 0
        dfs(root)
        return self.maxx


    def maxPathSum(self, root):  # 124. 二叉树中的最大路径和
        def dfs(root):
            if not root:
                return 0
            left_gain = max(0, dfs(root.left))
            right_gain = max(0, dfs(root.right))
            new_path = root.val + left_gain + right_gain
            self.max_path = max(self.max_path, new_path)
            return root.val + max(left_gain, right_gain)

        self.max_path = root.val
        dfs(root)
        return self.max_path

    def hasPathSum(self, root, targetSum):  # 112. 路径总和 根节点到叶子节点
        if not root:
            return False  # return False
        if not root.left and not root.right:  # 叶子结点
            return root.val == targetSum
        left_res = self.hasPathSum(root.left, targetSum - root.val)
        right_res = self.hasPathSum(root.right, targetSum - root.val)
        return left_res or right_res

    def PathSum(self, root, targetSum):  # 113. 路径总和 II 根节点到叶子节点 返回路径
        res = []
        path = []
        def dfs(root, targetSum, path):
            if not root:
                return
            path.append(root.val)
            if not root.left and not root.right and root.val == targetSum:  # 叶子结点,满足条件，加入结果
                res.append(path[:])
            dfs(root.left, targetSum - root.val, path)
            dfs(root.right, targetSum - root.val, path)
            path.pop()

        dfs(root, targetSum, [])
        return res

    def lowestCommonAncestor(self, root, p, q): # 最近公共祖先
        if not root:
            return None
        if root == p or root == q:
            return root
        node_left = self.lowestCommonAncestor(root.left, p, q)
        node_right = self.lowestCommonAncestor(root.right, p, q)
        if node_left and node_right:
            return root
        elif not node_left:
            return node_right
        else:
            return node_left

    def invertTree(self, root):  # 翻转二叉树
         if not root:
             return None
         left = self.invertTree(root.left)
         right = self.invertTree(root.right)
         root.left = right
         root.right = left
         return root

    def isduicheng(self, root): # 对称二叉树
        def check(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            return left.val == right.val and check(left.left, right.right) and check(left.right, right.left)

        if not root:
            return True
        return check(root.left, root.right)


    def isSameTree(self, p, q):  # 相同的树
        if not p and not q:
            return True
        if not p or not q:
            return False
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


    def isSubTree(self, root, subRoot):  # 另一个树的子树
        if not root:
            return False
        if self.isSameTree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    # 前序遍历序列化 和 反序列化
    def pre_serialize(self, root):
        if not root:
            return "NULL"
        return str(root.val) + ',' + str(self.pre_serialize(root.left)) + ',' + str(self.pre_serialize(root.right))
    def pre_deserialze(self, data):
        def helper(datalist):
            val = datalist.pop(0)
            if val == "NULL":
                return None
            root = TreeNode(int(val))
            root.left = helper(datalist)
            root.right = helper(datalist)
            return root
        datalist = data.split(",")
        return helper(datalist)

    # 后序遍历序列化 和 反序列化
    def post_serialize(self, root):
        if not root:
            return "NULL"
        return str(root.val) + ',' + str(self.post_serialize(root.left)) + ',' + str(self.post_serialize(root.right))
    def post_deserialze(self, data):
        def helper(datalist):
            val = datalist.pop()
            if val == "NULL":
                return None
            root = TreeNode(int(val))
            root.right = helper(datalist)  # 一定先确定右子树，再确定左子树
            root.left = helper(datalist)
            return root

        datalist = data.split(",")
        return helper(datalist)
    
    def flatten(self, root):  # 二叉树展开成链表
        if not root:
            return None
        self.flatten(root.left)
        self.flatten(root.right)
        temp_right = root.right
        root.right = root.left
        root.left = None
        while root.right:
            root = root.right
        root.right = temp_right
        return root

    def isValidBST(self, root):  # 验证二叉搜索树
        def preorder(root, min_v, max_v): # 前序遍历
            if not root:
                return True
            if root.val < max_v and root.val > min_v:
                return preorder(root.left, min_v, root.val) and preorder(root.right, root.val, max_v)
            return False

        return preorder(root, float('-inf'), float('inf'))
    
    def verifyTreeOrder(self, postorder: List[int]) -> bool:  # 验证二叉搜索树的后续遍历
        def verify(left, right):
            if left >= right:
                return True
            index = left
            while postorder[index] < postorder[right]:
                index += 1
            mid = index
            while postorder[index] > postorder[right]:
                index += 1
            return index == right and verify(left, mid - 1) and verify(mid, right - 1)

        if len(postorder) <= 2:
            return True
        return verify(0, len(postorder) - 1)


    def deleteNode(self, root, key):  # 删除二叉搜索树中的节点key
        if not root:
            return None
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            curr = root.right
            while curr.left:
                curr = curr.left
            root.val = curr.val
            root.right = self.deleteNode(root.right, curr.val)
        return root

    def findLargestKNode(self, root, k):  # 二叉搜索树中第K大的数
        # 二叉树中序遍历（逆反,右，中，左）
        def dfs(root):
            if not root:
                return
            dfs(root.right)
            self.k -= 1
            if self.k == 0:
                self.res = root.val
                return
            dfs(root.left)
        
        self.k = cnt
        self.res = 0
        dfs(root)
        return self.res

    def findsmallestKNode(self, root, k):  # 二叉搜索树中第K小的数
        # 二叉树中序遍历（左，中，右）
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            self.k -= 1
            if self.k == 0:
                self.res = root.val
                return
            dfs(root.right)

        self.res = 0
        self.k = k
        dfs(root)
        return self.res

    def treeToDoubleList(self, root):  # 二叉搜索树转化为双向链表
        # 二叉树中序遍历（左中右）#因为要返回最小元素的指针，所以这里采用中序
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            if self.tail:
                self.tail.right = root
                root.left = self.tail
            else:
                self.head = root
            self.tail = root
            dfs(root.right)

        if not root:
            return None
        self.head, self.tail = None, None
        dfs(root)
        self.head.left = self.tail
        self.tail.right = self.head
        return self.head

    

    def longestIncreasingPath(self, matrix):  # 矩阵中最长的递增路径
        def dfs(row, col):
            if cache[row][col] != 0:
                return cache[row][col]
            max_len = 1
            all_direct = [(-1, 0), (1, 0), (0, 1), (0, -1)]
            for x, y in all_direct:
                new_row, new_col = row + x, col + y
                if 0 <= new_row < m and 0 <= new_col < n and matrix[new_row][new_col] > matrix[row][col]:
                    max_len = max(max_len, dfs(new_row, new_col)) + 1
            cache[row][col] = max_len
            return max_len

        result = 0
        m, n = len(matrix), len(matrix[0])
        cache = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                result = max(result, dfs(i, j))
        return result

    def numIslands(self, grid):  # 岛屿数量
        def dfs(grid, i, j):
            m, n = len(grid), len(grid[0])
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
                return 0
            grid[i][j] = '0'
            dfs(grid, i + 1, j)
            dfs(grid, i - 1, j)
            dfs(grid, i, j + 1)
            dfs(grid, i, j - 1)

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(grid, i, j)
                    count += 1
        return count

    def maxAreaOfIsland(self, grid): # 岛屿的最大面积
        # # # 方法1 dfs
        def dfs(grid, i, j):
            if i < 0 or j<0 or i >= len(grid) or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == 0:
                return 0
            anx = 1
            grid[i][j] = 0
            anx += dfs(grid, i-1, j)
            anx += dfs(grid, i+1, j)
            anx += dfs(grid, i, j-1)
            anx += dfs(grid, i, j+1)
            return anx
        anx = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    anx = max(anx, dfs(grid, i, j))
        return anx

    def sumNumbers(self, root):  # 根节点到叶子结点的数字之和
        def dfs(root, pre_total):
            if not root:
                return 0
            total = pre_total * 10 + root.val
            if not root.left and not root.right:
                return total
            return dfs(root.left, total)  + dfs(root.right, total)

        return dfs(root, 0)


    def canFinish(self, numCourses, prerequisites):  # 课程表，考虑入度为0的节点出发，广度优先，输出是否可以完成Bool
        # 方法1：广度优先搜索，从入度为0的节点出发，【从低到高】
        from collections import defaultdict
        out_edg = defaultdict(list)  # 记录每个节点的出度(备注：这里的默认值必须写list)
        in_deg = [0] * numCourses  # 记录每个节点的入度
        for i in prerequisites:
            out_edg[i[1]].append(i[0])
            in_deg[i[0]] += 1
        q = [i for i in range(numCourses) if in_deg[i] == 0]   # 把入度为0的节点加入队列
        res = 0
        while q:
            res += 1
            i = q.pop(0)
            for v in out_edg[i]:
                in_deg[v] -= 1  # 入度为当前节点的节点入度减一，若为0，加入队列
                if in_deg[v] == 0:
                    q.append(v)
        return res == numCourses  # 判断最后循环结束时，res是否等于课程数（无环）

    def canFinish2(self, numCourses, prerequisites):  # 课程表2，输出路径
        # 方法1：广度优先搜索，从入度为0的节点出发，【从低到高】
        from collections import defaultdict
        out_edg = defaultdict(list)  # 记录每个节点的出度(备注：这里的默认值必须写list)
        in_deg = [0] * numCourses  # 记录每个节点的入度
        for i in prerequisites:
            out_edg[i[1]].append(i[0])
            in_deg[i[0]] += 1
        q = [i for i in range(numCourses) if in_deg[i] == 0]   # 把入度为0的节点加入队列
        res = []
        while q:
            res += 1
            i = q.pop(0)
            res.append(i)
            for v in out_edg[i]:
                in_deg[v] -= 1  # 入度为当前节点的节点入度减一，若为0，加入队列
                if in_deg[v] == 0:
                    q.append(v)
        return res if len(res) == numCourses else []  # 判断最后循环结束时，res是否等于课程数（无环）

    def numDifferentBinarySearchTrees(self, n):  # 不同的二叉搜索树的个数
        # *** 竟然是推导得出的状态转移方程
        # dp[i]:表示为i个节点可以构成的二叉搜索树个数
        # 定义了f(i)表示以i为根结点的二叉搜索树的个数，g(i)表示i个结点可以构成的二叉搜索树的个数
        # 推导出g(i) = sum(1<=j<=i){g(j-1)*g(i-j)}
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        # print(dp)
        return dp[-1]
        return

    def coinChange(self, amount, coins):  # 零钱兑换（way1:广度优先搜索，找最短路径）O(amount * size)\O(amount);way2:动态规划
        if amount == 0:
            return 0
        visited = set([amount])
        queue = [amount]
        step = 0
        while queue:
            step += 1
            size = len(queue)
            for _ in range(size):
                cur = queue.pop(0)
                for coin in coins:
                    if coin == cur:
                        return step
                    elif cur > coin and cur - coin not in visited:
                        queue.append(cur-coin)
                        visited.add(cur-coin)
        return -1

        # # way2 动态规划
        # dp = [amount + 1] * (amount + 1)
        # dp[0] = 0
        # for i in range(1, amount + 1):
        #     for coin in coins:
        #         if i < coin:
        #             continue
        #         dp[i] = min(dp[i], dp[i - coin] + 1)
        # if dp[amount] != amount + 1:
        #     return dp[amount]
        # else:
        #     return -1












































