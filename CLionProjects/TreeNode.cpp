#include <iostream>
#include <vector>
#include <queue>
#include <sstream>
#include <string>
#include "TreeNode.h"
using namespace std;

vector<int> TreeNodeSolution::preorderTraversal(TreeNode *root) {
    if (!root) {
        return {};
    }
    vector<int> res = {root->val};
    vector<int> left = preorderTraversal(root->left);
    vector<int> right = preorderTraversal(root->right);
    res.insert(res.end(), left.begin(), left.end());
    res.insert(res.end(), right.begin(), right.end());
    return res;
}

vector<int> TreeNodeSolution::inorderTraversal(TreeNode *root) {
    if (!root) {
        return {};
    }
    vector<int> left = preorderTraversal(root->left);
    vector<int> res = {root->val};
    vector<int> right = preorderTraversal(root->right);
    left.insert(left.end(), res.begin(), res.end());
    left.insert(left.end(), right.begin(), right.end());
    return left;
}

vector<int> TreeNodeSolution::postorderTraversal(TreeNode *root) {
    if (!root) {
        return {};
    }
    vector<int> left = postorderTraversal(root->left);
    vector<int> right = postorderTraversal(root->right);
    vector<int> res = {root->val};
    left.insert(left.end(), right.begin(), right.end());
    left.insert(left.end(), res.begin(), res.end());
    return left;
}

vector<vector<int>> TreeNodeSolution::levelOrder(TreeNode *root) {
    vector<vector<int>> res;
    if (!root) {
        return res;
    }
    std::queue<TreeNode*> queuex = {};
    queuex.push(root);
    while (!queuex.empty()) {
        int queue_size = queuex.size();
        vector<int> level;
        for (int i=0; i < queue_size; i++) {
            TreeNode *node = queuex.front();
            queuex.pop();
            level.push_back(node->val);
            if (!node->left) {queuex.push(node->left);}
            if (!node->right) {queuex.push(node->right);}
        }
        res.push_back(level);
    }
    return res;
}

vector<vector<int>> TreeNodeSolution::zigzagLevelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if (!root) return res;
    std::queue<TreeNode*> queuex = {};
    queuex.push(root);
    bool odd = true;
    while(!queuex.empty()) {
        int queue_size = queuex.size();
        vector<int> level = {};
        for (int i=0; i < queue_size; i++) {
            TreeNode* node = queuex.front();
            queuex.pop();
            level.push_back(node->val);
            if (node->left) queuex.push(node->left);
            if (node->right) queuex.push(node->right);
        }
        if (odd) {
            res.push_back(level);
        } else {
            reverse(level.begin(), level.end());
            res.push_back(level);
        }
        odd = not odd;
    }
    return res;
}

int TreeNodeSolution::widthOfBinaryTree(TreeNode* root) {
    if (!root) return 0;
    std::queue<std::pair<TreeNode*, int>> queuex;
    queuex.push(std::make_pair(root, 0));
    int max_lengh = 0;
    while (!queuex.empty()) {
        int queue_size = queuex.size();
        max_lengh = max(queuex.back().second - queuex.front().second + 1, max_lengh);
        for (int i=0; i<queue_size; i++) {
            TreeNode *node = queuex.front().first;
            long long index = queuex.front().second;
            queuex.pop();
            if (node->left) queuex.push(std::make_pair(node->left, 2 * index + 1));
            if (node->right) queuex.push(std::make_pair(node->right, 2 * index + 2));
        }
    }
    return max_lengh;
}

vector<int> TreeNodeSolution::rightSideView(TreeNode* root) {
    vector<int> res = {};
    if(!root) return {};
    std::queue<TreeNode*> queuex;
    queuex.push(root);
    while(!queuex.empty()) {
        int queue_size = queuex.size();
        for (int i=0; i < queue_size; i++) {
            TreeNode * node = queuex.front();
            queuex.pop();
            if(i == queue_size - 1) res.push_back(node->val);
            if (node->left) queuex.push(node->left);
            if (node->right) queuex.push(node->right);
        }
    }
    return res;
}
// 二叉树的完全性检验
bool TreeNodeSolution::isCompleteTree(TreeNode* root) {
    if (!root) return false;
    std::queue<TreeNode*> q;
    q.push(root);
    bool is_empty = false;
    while (!q.empty()) {
        int q_size = q.size();
        for(int i=0; i<q_size; i++) {
            TreeNode* node = q.front();
            q.pop();
            if(!node) {
                is_empty = true;
            } else {
                if (is_empty) return false;
                q.push(node->left);
                q.push(node->right);
            }
        }
    }
    return true;
}

TreeNode* TreeNodeSolution::buildTreeHelper(const vector<int>&preorder, int p_start, int p_end,
                          const vector<int>&inorder, int in_start, int in_end) {
    if (p_start == p_end) return NULL;
    int root_val = preorder[p_start];
    TreeNode* root = new TreeNode(root_val);
    int root_index = 0;
    for(int i=in_start; i<in_end; i++) {
        if(inorder[i] == root_val) {
            root_index = i;
            break;
        }
    }
    int left_size = root_index - in_start;
    root->left = buildTreeHelper(preorder, p_start+1, p_start+1+left_size,
                                 inorder, in_start, root_index);
    root->right = buildTreeHelper(preorder, p_start+1+left_size, p_end,
                                  inorder, root_index + 1, in_end);
    return root;
}
TreeNode* TreeNodeSolution::buildTree(vector<int>& preorder, vector<int>inorder) {
    return buildTreeHelper(preorder, 0, preorder.size(), inorder, 0, inorder.size());
}

TreeNode* TreeNodeSolution::buildTreeHelper2(const vector<int>& inorder, int in_start, int in_end,
                           const vector<int>& postorder, int po_start, int po_end) {
    if (po_start == po_end) return NULL;
    int root_val = postorder[po_end - 1];
    TreeNode* root = new TreeNode(root_val);
    int root_index = 0;
    for(int i=in_start; i<in_end; i++) {
        if(inorder[i] == root_val) {
            root_index = i;
            break;
        }
    }
    int left_size = root_index - in_start;
    root->left = buildTreeHelper2(inorder, in_start, root_index,
                                  postorder, po_start, po_start+left_size);
    root->right = buildTreeHelper2(inorder, root_index + 1, in_end,
                                   postorder, po_start+left_size, po_end-1);
    return root;
}
TreeNode* TreeNodeSolution::buildTree2(vector<int>& inorder, vector<int>& postorder) {
    return buildTreeHelper2(inorder, 0, inorder.size(), postorder, 0, postorder.size());
}

int TreeNodeSolution::maxDepth(TreeNode* root) {
    if (!root) return 0;
    int left_h = maxDepth(root->left);
    int right_h = maxDepth(root->right);
    return max(left_h, right_h) + 1;
}

int TreeNodeSolution::minDepth(TreeNode* root) {
    if (!root) return 0;
    if (!root->left) return minDepth(root->right) + 1;
    if (!root->right) return minDepth(root->left) + 1;
    return min(minDepth(root->right), minDepth(root->left)) + 1;
}
// 最大路径和
int TreeNodeSolution::pathSum(TreeNode* root){
    if (!root) return 0;
    int left_x = maxPathSum(root->left);
    int right_x = maxPathSum(root->right);
    int cur_max = left_x + right_x + root->val;
    max_sum = max(max_sum, cur_max);
    return max(left_x, right_x) + root->val;
}
int TreeNodeSolution::maxPathSum(TreeNode* root) {
    if (!root) return 0;
    max_sum = root->val;
    pathSum(root);
    return max_sum;
}

// 路径总和
void dfs(TreeNode* root, int targetSum, vector<int>& path, vector<vector<int>>& res) {
    if (!root) {
        return;
    }
    path.push_back(root->val);
    if (!root->left && !root->right && root->val == targetSum) {
        res.push_back(path);
    }
    dfs(root->left, targetSum - root->val, path, res);
    dfs(root->right, targetSum - root->val, path, res);
    path.pop_back();
}
vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
    vector<vector<int>> res;
    vector<int> path;
    dfs(root, targetSum, path, res);
    return res;
}
// 最大直径
int TreeNodeSolution::dfs(TreeNode* root){
    if (!root) return 0;
    int left_d = dfs(root->left);
    int right_d = dfs(root->right);
    int max_d = left_d + right_d;
    max_diameter = max(max_d, max_diameter);
    return max(left_d, right_d) + 1;
}
int TreeNodeSolution::diameterOfBinaryTree(TreeNode* root) {
    if (!root) return 0;
    max_diameter = 0;
    dfs(root);
    return max_diameter;
}

// 翻转二叉树
TreeNode* invertTree(TreeNode* root) {
    if (!root) return root;
    TreeNode* left = invertTree(root->left);
    TreeNode* right = invertTree(root->right);
    root->left = right;
    root->right = left;
    return root;
}
TreeNode* invertTree_way2(TreeNode* root) {
    if (!root) return root;
    std::queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int q_size = q.size();
        for(int i=0; i<q_size; i++) {
            TreeNode *Node = q.front();
            q.pop();
            std::swap(Node->left, Node->right);
            if (Node->left) { q.push(Node->left); }
            if (Node->right) { q.push(Node->right); }
        }
    }
    return root;
}
// 相同的树
bool isSameTree(TreeNode* p, TreeNode* q) {
    if (!p && !q) return true;
    if (!p || !q) return false;
    bool res = (p->val == q->val) &&  isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    return res;
}
// 另一颗树的子树
bool isSubtree(TreeNode* root, TreeNode* subRoot) {
    if (!root) return false;
    if (isSameTree(root, subRoot)) return true;
    return isSubtree(root->left, subRoot) or isSubtree(root->right, subRoot);
}
// 对称二叉树
bool check(TreeNode* left, TreeNode* right) {
    if (!left && !right) return true;
    if (!left || !right) return false;
    return left->val == right->val && check(left->left, right->right) && check(left->right, right->left);
}
bool isSymmetric(TreeNode* root) {
    if (!root) return true;
    return check(root->left, root->right);
}
// 236. 二叉树的最近公共祖先
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root) return NULL;
    if ((p == root) || (q == root)) return root;
    TreeNode* left_a = lowestCommonAncestor(root->left, p, q);
    TreeNode* right_a = lowestCommonAncestor(root->right, p, q);
    if (left_a && right_a) {
        return root;
    } else if (!left_a) {
        return right_a;
    } else {
        return left_a;
    }
}
// 二叉树序列化和反序列化
string serialize(TreeNode* root) {
    if (!root) return "NULL";
    return to_string(root->val) + "," + serialize(root->left) + "," + serialize(root->right);
}

vector<string> split(string data) {
    vector<string> res;
    stringstream ss(data);
    string token;
    while (getline(ss, token, ',')) {
        res.push_back(token);
    }
    return res;
}
TreeNode* helper(vector<string>& datalist) {
    string val = datalist.front();
    datalist.erase(datalist.begin());
    if (val == "NULL") {
        return NULL;
    }
    TreeNode* root = new TreeNode(stoi(val));
    root->left = helper(datalist);
    root->right = helper(datalist);
    return root;
}
// Decodes your encoded data to tree.
TreeNode* deserialize(string data) {
    std::vector<string> datalist = split(data);
    return helper(datalist);
}
// 114. 二叉树展开为链表
void flatten(TreeNode* root) {
    if (!root) return;
    flatten(root->left);
    flatten(root->right);
    TreeNode* temp_right = root->right;
    root->right = root->left;
    root->left = nullptr;
    while (root->right) {root = root->right;}
    root->right = temp_right;
}

// 98. 验证二叉搜索树
bool check(TreeNode* root, long long min_v, long long max_v) {
    if (!root) return true;
    if (root->val > min_v && root->val < max_v) {
        return check(root->left, min_v, root->val) && check(root->right, root->val, max_v);
    }
    return false;
}
bool isValidBST(TreeNode* root) {
    return check(root, numeric_limits<long long>::min(), numeric_limits<long long>::max());
}

//验证二叉搜索树的后序遍历
bool verify(vector<int>& postorder, int left, int right) {
    if (left >= right) return true;
    int index = left;
    while (postorder[index] < postorder[right]) {
        index++;
    }
    int mid = index;
    while (postorder[index] > postorder[right]) {
        index++;
    }
    return (index==right) && verify(postorder, left, mid - 1) && verify(postorder, mid, right-1);
}
bool verifyTreeOrder(vector<int>& postorder) {
    if (postorder.size() <= 2) return true;
    return verify(postorder, 0, postorder.size() - 1);
}

//450. 删除二叉搜索树中的节点
TreeNode* deleteNode(TreeNode* root, int key) {
    if(!root) return nullptr;
    if (root->val > key) {
        root->left = deleteNode(root->left, key);
    } else if (root->val < key){
        root->right = deleteNode(root->right, key);
    } else {
        if (!root->left) return root->right;
        if (!root->right) return root->left;
        TreeNode* cur = root->right;
        while (cur->left){
            cur = cur->left;
        }
        root->val = cur->val;
        root->right = deleteNode(root->right, cur->val);
    }
    return root;
}

// LCR 174. 寻找二叉搜索树中的第K大的数
int k;
int res;
void dfs_max(TreeNode* root) {
    if (!root) return;
    dfs_max(root->right);
    k--;
    if(k == 0){
        res = root->val;
        return;
    }
    dfs_max(root->left);
}
int findTargetNode(TreeNode* root, int cnt) {
    k = cnt;
    res = 0;
    dfs_max(root);
    return res;
}

//寻找二叉搜索树中的第K小的数
int kth;
void dfs_min(TreeNode* root) {
    if (!root) return;
    dfs_min(root->left);
    kth --;
    if (kth == 0){
        res = root->val;
        return;
    }
    dfs_min(root->right);
}
int kthSmallest(TreeNode* root, int k) {
    kth = k;
    res = 0;
    dfs_min(root);
    return res;
}

// 329. 矩阵中的最长递增路径
int longestIncreasingPath(vector<vector<int>>& matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
    int max_res = 0;
    // cache[i][j]表示从矩阵中位置(i, j)开始的最长递增路径的长度
    vector<vector<int>> cache(m, vector(n, 0));
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            max_res = max(max_res, dfs(i, j, matrix, cache));
        }
    }
    return max_res;
}
int dfs(int row, int col, vector<vector<int>>& matrix, vector<vector<int>>& cache) {
    if (cache[row][col] != 0) return cache[row][col];
    int max_len = 1;
    vector<std::pair<int, int>> direct = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
    for (const auto& [x, y]:direct) {
        int new_row = row + x, new_col = col + y;
        if (new_row>=0 && new_row < matrix.size() && new_col>=0 && new_col < matrix[0].size() && matrix[new_row][new_col] > matrix[row][col]) {
            max_len = max(max_len, dfs(new_row, new_col, matrix, cache) + 1);
        }
    }
    cache[row][col] = max_len;
    return max_len;
}

// 岛屿数量
void dfs1(vector<vector<char>>& grid, int row, int col) {
    int m = grid.size(), n = grid[0].size();
    if (row>=m || row <0 || col>=n || col <0 || grid[row][col] == '0') {
        return;
    }
    grid[row][col] = '0';
    vector<pair<int, int>> direct = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (const auto& [x, y]: direct) {
        dfs1(grid, row+x, col+y);
    }
}
int numIslands(vector<vector<char>>& grid) {
    int m = grid.size(), n = grid[0].size();
    int cnt = 0;
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++) {
            if (grid[i][j] == '1') {
                dfs1(grid, i, j);
                cnt++;
            }
        }
    }
    return cnt;
}
// 岛屿最大面积
int dfs2(vector<vector<int>>& grid, int row, int col) {
    int m = grid.size(), n = grid[0].size();
    int res = 0;
    if (row<0 || row>=m || col<0 || col>=n || grid[row][col] == 0) return 0;
    res = 1;
    grid[row][col] = 0;
    vector<pair<int, int>> direct = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
    for(const auto& [x, y]: direct) {
        int new_row = row + x, new_col = col + y;
        res += dfs2(grid, new_row, new_col);
    }
    return res;
}
int maxAreaOfIsland(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    int max_res = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            max_res = max(max_res, dfs2(grid, i, j));
        }
    }
    return max_res;
}
//129. 求根节点到叶节点数字之和
int dfs(TreeNode* root, int pre_total) {
    if(!root) return 0;
    int total = pre_total * 10 + root->val;
    if(!root->left && !root->right) {
        return total;
    }
    return dfs(root->left, total) + dfs(root->right, total);
}
int sumNumbers(TreeNode* root) {
    return dfs(root, 0);
}

bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    unordered_map<int, vector<int>> out_table;
    vector<int> in_table(numCourses, 0);
    for (const auto& i: prerequisites) {
        out_table[i[1]].push_back(i[0]);
        in_table[i[0]]++;
    }
    queue<int> q;
    for (int i = 0; i < in_table.size(); i++) {
        if (in_table[i] == 0) {
            q.push(i);
        }
    }
    int res = 0;
    while (!q.empty()) {
        int cur = q.front();
        res++;
        q.pop();
        for (auto& k : out_table[cur]) {
            in_table[k]--;
            if (in_table[k] == 0) {
                q.push(k);
            }
        }
    }
    return res == numCourses;
}



