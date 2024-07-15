#include <iostream>
#include "ListNode.h"
#include "TreeNode.h"
#include <vector>
using namespace std;


void printListNode(ListNode* head) {
    ListNode *cur = head;
    while (cur) {
        cout << cur->val << "->";
        cur = cur->next;
    }
    cout << "NULL" << endl;
}
void cleanMemory(ListNode* head) {
    ListNode* cur = head;
    while (cur) {
        ListNode* temp = cur;
        cur = cur->next;
        delete temp;
    }
    cout << "delete memory success!" << endl;
}

void testListNode() {
    ListNode *a = new ListNode(1);
    ListNode *b = new ListNode(2);
    ListNode *c = new ListNode(3);
    ListNode *d = new ListNode(4);
    a->next = b;
    b->next = c;
    c->next = d;
    std::cout << "Original list: ";
    printListNode(a);
    ListNodeSolution s;
//    ListNode *rev = s.reveseList(a);
//    ListNode *rev = s.reveseList2(a);
    ListNode *rev = s.swapPairs(a);

    std::cout << "After list: ";
    printListNode(rev);
    cleanMemory(rev);
}


//void printTreeNode(vector<int>& root_list) {
//    for (int x: root_list) {
//        cout << x << " ";
//    }
//    cout << "\n" << endl;
//}
//
//void testTreeNode() {
//    TreeNode *root = new TreeNode(0);
//    root->left = new TreeNode(1);
//    root->right = new TreeNode(2);
//    root->left->left = new TreeNode(3);
//    root->left->right = new TreeNode(4);
//    root->right->left = new TreeNode(5);
//    root->right->right = new TreeNode(6);
//    TreeNodeSolution s;
////    vector<int> res = s.preorderTraversal(root);
//    vector<int> res = s.inorderTraversal(root);
//    printTreeNode(res);
//}
//

int main() {
//    testListNode();
//    testTreeNode();
    vector<std::pair<int, int>> direct = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    for(int i = 0; i < direct.size(); i++) {
        cout << direct[i].first << "_" << direct[i].second << endl;
    }
}

