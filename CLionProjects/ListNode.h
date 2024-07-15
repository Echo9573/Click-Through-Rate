#include <iostream>

#ifndef ListNode_H
#define ListNode_H
struct ListNode {
    int val;
    ListNode *next;
    ListNode(): val(0), next(nullptr) {};
    ListNode(int x): val(x), next(nullptr) {};
};


struct Node {
    int val;
    Node *next;
    Node *random;
    Node(): val(0), next(nullptr), random(nullptr) {};
    Node(int x): val(x), next(nullptr), random(nullptr) {};
};



class ListNodeSolution {
public:
    ListNode* reveseList(ListNode* head);  //翻转链表1
    ListNode* reveseList2(ListNode* head);  //翻转链表1-way2
    ListNode* reverseBetween(ListNode* head, int left, int right); //92反转链表 II
    ListNode* reverseTopK(ListNode* head, int k);
    ListNode* reverseBetween2(ListNode* head, int left, int right); //92反转链表 II-way2
    ListNode* swapPairs(ListNode* head);  //24两两交换链表中的节点
    ListNode* deleteDuplicates(ListNode* head); //83删除链表中重复元素
    ListNode* deleteDuplicates2(ListNode* head); //82删除链表中重复元素2
    ListNode* reverseKGroup(ListNode* head, int k); //25K 个一组翻转链表
    ListNode* reverse_between(ListNode* a, ListNode* b);
    ListNode* oddEvenList(ListNode* head); // 328奇偶链表;
    bool isPalindrome(ListNode* head); //234回文链表;
    Node* copyRandomList(Node* head); //138随机链表的复制
    ListNode* rotateRight(ListNode* head, int k); // 61旋转链表
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2); //21合并两个有序链表
    ListNode* sortList(ListNode* head); //148排序链表
    ListNode* merge_sort(std::vector<ListNode*>& lists, int left, int right);
    ListNode* mergeKLists(std::vector<ListNode*>& lists); //23合并 K 个升序链表
    bool hasCycle(ListNode *head); //141环形链表
    ListNode *detectCycle(ListNode *head); // 142环形链表 II
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB); // 160相交链表
    ListNode *tailKNode(ListNode *head, int k); // 链表中倒数第k个节点
    void reorderList(ListNode* head); // 143重排链表
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2); // 2两数相加
    ListNode* addTwoNumbers2(ListNode* l1, ListNode* l2); //445两数相加 II
};



#endif

