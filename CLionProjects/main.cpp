#include <iostream>
#include "ListNode.h"
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

int main() {
    ListNode *a = new ListNode(1);
    ListNode *b = new ListNode(2);
    ListNode *c = new ListNode(3);
    ListNode *d = new ListNode(4);
    a->next = b;
    b->next = c;
    c->next = d;
    std::cout << "Original list: ";
    printListNode(a);
    Solution s;
//    ListNode *rev = s.reveseList(a);
//    ListNode *rev = s.reveseList2(a);
    ListNode *rev = s.swapPairs(a);

    std::cout << "After list: ";
    printListNode(rev);
    cleanMemory(rev);
}
