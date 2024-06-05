

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
    def buildTree(self, inorder, preorder):
        if not inorder or not preorder:
            return None
        root_val = preorder[0]
        root = TreeNode(root_val)
        root_index = inorder.index(root_val)
        root.left = self.buildTree(inorder[:root_index], preorder[1:1 + root_index])
        root.right = self.buildTree(inorder[root_index + 1:], preorder[root_index + 1:])
        return root

preorder = [3, 9, 20, 15, 7]
inorder = [9, 3, 15, 20, 7]
a = Solution()
root = a.buildTree(inorder, preorder)
print(a.preorderTraversal(root))
print(a.inorderTraversal(root))
