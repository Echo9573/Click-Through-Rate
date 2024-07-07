#include <iostream>
#include <unordered_map>
#include <vector>
#include "ListNode.h"
ListNode* Solution::reveseList(ListNode* head) {
    if (!head || !head->next) {
        return head;
    }
    ListNode* last = Solution::reveseList(head->next);
    head->next->next = head;
    head->next = NULL;
    return last;
}

ListNode* Solution::reveseList2(ListNode* head) {
    ListNode *prev = NULL;
    ListNode *cur = head;
    while (cur){
        ListNode *temp = cur->next;
        cur->next = prev;
        prev = cur;
        cur = temp;
    }
    return prev;
}

ListNode* Solution::reverseBetween(ListNode* head, int left, int right) {
    ListNode *dummy_node = new ListNode();
    dummy_node->next = head;
    ListNode *cur = dummy_node;
    int index = 1;
    while (cur && index < left) {
        cur = cur->next;
        index++;
    }
    ListNode *prev = cur;
    ListNode *cur0 = prev->next;
    while (cur0 && index <= right){
        ListNode *temp = cur0->next;
        cur0->next = prev;
        prev = cur0;
        cur0 = temp;
        index ++;
    }
    cur->next->next = cur0;
    cur->next = prev;
    return dummy_node->next;
}

ListNode* Solution::reverseTopK(ListNode* head, int k) {
    if (k == 1 || !head) {
        return head;
    }
    ListNode* last = reverseTopK(head->next, k - 1);
    ListNode* succ = head->next->next;
    head->next->next = head;
    head->next = succ;
    return last;
}

ListNode* Solution::reverseBetween2(ListNode* head, int left, int right) {
    int m = left;
    int n = right;
    if (n == 1) {
        return head;
    }
    if (m == 1) {
        return reverseTopK(head, n);
    }
    head->next = reverseBetween2(head->next, m - 1, n - 1);
    return head;
}

ListNode* Solution::swapPairs(ListNode* head) {
    if (!head or !head->next) {
        return head;
    }
    ListNode *dummy_node = new ListNode();
    dummy_node->next = head;
    ListNode *cur = dummy_node;
    while (cur->next && cur->next->next) {
        ListNode *node1 = cur->next;
        ListNode *node2 = cur->next->next;
        ListNode *temp = node2->next;
        cur->next = node2;
        node2->next = node1;
        node1->next = temp;
        cur = node1;
    }
    return dummy_node->next;
}

ListNode* Solution::deleteDuplicates(ListNode* head) {
    ListNode *cur = head;
    while (cur && cur->next) {
        if (cur->val == cur->next->val) {
            cur->next = cur->next->next;
        } else {
            cur = cur->next;
        }
    }
    return head;
}

ListNode* Solution::deleteDuplicates2(ListNode* head) {
    ListNode *dummy_node = new ListNode();
    dummy_node->next = head;
    ListNode *cur = dummy_node;
    while (cur->next && cur->next->next) {
        if (cur->next->val == cur->next->next->val) {
            ListNode *temp = cur->next;
            while (temp && temp->next && temp->val == temp->next->val) {
                temp = temp->next;
            }
            cur->next = temp->next;
        } else {
            cur = cur->next;
        }
    }
    return dummy_node->next;
}

ListNode* Solution::reverse_between(ListNode* a, ListNode* b) {
    ListNode* prev = new ListNode();
    ListNode* cur = a;
    while (cur != b) {
        ListNode* temp = cur->next;
        cur->next = prev;
        prev = cur;
        cur = temp;
    }
    return prev;
}
ListNode* Solution::reverseKGroup(ListNode* head, int k) {
    ListNode* a = head;
    ListNode* b = head;
    for(int i=0; i < k; i++) {
        if (!b) {
            return head;
        }
        b = b->next;
    }
    ListNode* new_head = reverse_between(a, b);
    a->next = reverseKGroup(b, k);
    return new_head;
}

ListNode* Solution::oddEvenList(ListNode* head) {
    if (!head || !head->next || !head->next->next) {
        return head;
    }
    ListNode* odd = head;
    ListNode* even_head = head->next;
    ListNode* even = head->next;
    ListNode* cur = even->next;
    bool is_odd = true;
    while (cur) {
        if (is_odd) {
            odd->next = cur;
            odd = odd->next;
        } else {
            even->next = cur;
            even = even->next;
        }
        cur = cur->next;
        is_odd = not is_odd;
    }
    even->next = NULL;
    odd->next = even_head;
    return head;
}

bool Solution::isPalindrome(ListNode* head) {
    if (!head || !head->next) {
        return true;
    }
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        fast = fast->next->next;
        slow = slow->next;
    }
    if (fast) {
        slow = slow->next;
    }
    ListNode *slow_rev = reveseList(slow);
    ListNode *begin = head;
    while (slow_rev) {
        if (slow_rev->val == begin->val) {
            slow_rev = slow_rev->next;
            begin = begin->next;
        } else {
            return false;
        }
    }
}

Node* Solution::copyRandomList(Node* head) {
    if (!head) {
        return head;
    }
    std::unordered_map<Node*, Node*> nodeDict;
    Node *cur = head;
    while (cur) {
        nodeDict[cur] = new Node(cur->val);
        cur = cur->next;
    }
    cur = head;
    while (cur) {
        if (cur->next) {
            nodeDict[cur]->next = nodeDict[cur->next];
        }
        if (cur->random) {
            nodeDict[cur]->random = nodeDict[cur->random];
        }
        cur = cur->next;
    }
    return nodeDict[head];
}

ListNode* Solution::rotateRight(ListNode* head, int k) {
    if (k == 0 || !head) {
        return head;
    }
    int lens = 1;
    ListNode *cur = head;
    while (cur->next) {
        cur = cur->next;
        lens ++;
    }
    int n = lens - (k % lens);
    cur->next = head;
    while (n > 0) {
        cur = cur->next;
        n -= 1;
    }
    ListNode *new_head = cur->next;
    cur->next = nullptr;
    return new_head;
}

ListNode* Solution::mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode *dummy_node = new ListNode();
    ListNode *cur = dummy_node;
    while (list1 && list2) {
        if (list1->val < list2->val) {
            cur->next = list1;
            list1 = list1->next;
        } else {
            cur->next = list2;
            list2 = list2->next;
        }
        cur = cur->next;
    }
    if (list1) {cur->next = list1;}
    if (list2) {cur->next = list2;}
    return dummy_node->next;
}

ListNode* Solution::sortList(ListNode* head) {
    if (!head || !head->next) {
        return head;
    }
    ListNode *slow = head;
    ListNode *fast = head->next;
    while (fast && fast->next) {
        fast = fast->next->next;
        slow = slow->next;
    }
    ListNode *right = slow->next;
    slow->next = NULL;
    ListNode *left = head;
    return mergeTwoLists(sortList(left), sortList(right));
}

ListNode* Solution::merge_sort(std::vector<ListNode*>& lists, int left, int right) {
    if (left == right) {return lists[left];}
    if (left > right) {return NULL;}
    int mid = left + (right - left) / 2;
    return mergeTwoLists(merge_sort(lists, left, mid), merge_sort(lists, mid + 1, right));
}

ListNode* Solution::mergeKLists(std::vector<ListNode*>& lists) {
    return merge_sort(lists, 0, lists.size() - 1);
}

bool Solution::hasCycle(ListNode *head) {
    if (!head || !head->next) {return false;}
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        fast = fast->next->next;
        slow = slow->next;
        if (fast == slow) {
            return true;
        }
    }
    return false;
}

ListNode* Solution::detectCycle(ListNode *head) {
    if (!head) {return head;}
    ListNode *slow = head;
    ListNode *fast = head;
    bool has_cycle = 0;
    while (fast && fast->next) {
        fast = fast->next->next;
        slow = slow->next;
        if (fast == slow) {
            has_cycle = true;
            break;
        }
    }
    if (has_cycle) {
        slow = head;
        while (slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    } else {
        return nullptr;
    }
}

ListNode* Solution::getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode *l1 = headA;
    ListNode *l2 = headB;
    while (l1 != l2) {
        if (!l1) {
            l1 = headB;
        } else {
            l1 = l1->next;
        }
        if (!l2) {
            l2 = headA;
        } else {
            l2 = l2->next;
        }
    }
    return l1;
}

ListNode* Solution::tailKNode(ListNode *head, int k) {
    ListNode *dummy_node = new ListNode();
    dummy_node->next = head;
    ListNode *fast = head;
    while (k) {
        fast = fast->next;
        k--;
    }
    ListNode *slow = head;
    while (fast) {
        fast = fast->next;
        slow = slow->next;
    }
    return slow;
}

void Solution::reorderList(ListNode* head) {
    std::vector<ListNode*> vec = {};
    ListNode *cur = head;
    while (cur) {
        vec.push_back(cur);
        cur = cur->next;
    }
    int left = 0;
    int right = vec.size() - 1;
    while (left < right) {
        vec[left]->next = vec[right];
        left ++;
        vec[right]->next = vec[left];
        right --;
    }
    vec[left]->next = nullptr; // 这个必须加上
}

ListNode* Solution::addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode *dummy_node = new ListNode();
    ListNode *cur = dummy_node;
    int carry = 0;
    int x1, x2;
    while (l1 or l2 or carry) {
        if (not l1) {
            x1 = 0;
        } else {
            x1 = l1->val;
            l1 = l1->next;
        }
        if (not l2) {
            x2 = 0;
        } else {
            x2 = l2->val;
            l2 = l2->next;
        }
        int sums = x1 + x2 + carry;
        carry = sums / 10;
        cur->next = new ListNode(sums % 10);
        cur = cur->next;
    }
    return dummy_node->next;
}

ListNode* Solution::addTwoNumbers2(ListNode* l1, ListNode* l2) {
    ListNode *l1_rev = reveseList(l1);
    ListNode *l2_rev = reveseList(l2);
    return reveseList(addTwoNumbers(l1_rev, l2_rev));
}