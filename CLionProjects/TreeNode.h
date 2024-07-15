#include <iostream>
#include <vector>
using namespace std;
#ifndef TREE_H
#define TREE_H

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(): val(0), left(nullptr), right(nullptr) {};
    TreeNode(int x): val(x), left(nullptr), right(nullptr) {};
};

class TreeNodeSolution {
public:
    vector<int> preorderTraversal(TreeNode *root); // 前向遍历
    vector<int> inorderTraversal(TreeNode *root); // 中序遍历
    vector<int> postorderTraversal(TreeNode *root); // 后序遍历
    int pathSum(TreeNode* root);
    int maxPathSum(TreeNode* root); // 最大路径和
    int dfs(TreeNode* root);
    int diameterOfBinaryTree(TreeNode* root);
    vector<vector<int>> levelOrder(TreeNode *root);

private:
    int max_sum;
    int max_diameter;

};


#endif
