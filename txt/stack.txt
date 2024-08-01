class MinStack:
    def __init__(self):
        self.minstack = [] # 用辅助栈来记录当前栈中的最小值
        self.stack = []

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append(val)
            self.minstack.append(val)
        else:
            self.stack.append(val)
            self.minstack.append(min(val, self.minstack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.minstack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minstack[-1]
# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

class MyQueue:   # 栈实现队列

    def __init__(self):
        self.instack = []
        self.outstack = []


    def push(self, x: int) -> None:
        self.instack.append(x)

    def pop(self) -> int:
        # 秉承对于先进后出的栈，只能取栈顶的规则
        if len(self.outstack) == 0:
            while len(self.instack) != 0:
                self.outstack.append(self.instack[-1])
                self.instack.pop()
        return self.outstack.pop()

    def peek(self) -> int:
        # 秉承对于先进后出的栈，只能取栈顶的规则
        if len(self.outstack) == 0:
            while len(self.instack) != 0:
                self.outstack.append(self.instack[-1])
                self.instack.pop()
        top = self.outstack[-1]
        return top

    def empty(self) -> bool:
        return len(self.instack) == 0 and len(self.outstack) == 0



# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()

class Solution:
    def isValid(self, s):  #  有效的括号
        dic = {"(": ")", "{": "}", "[": "]"}
        stack = []
        for i in s:
            if i in dic.keys():
                stack.append(dic[i])
            else:
                if stack and i == stack[-1]:
                    stack.pop()
                else:
                    return False
        if len(stack) > 0:
            return False
        else:
            return True


    def decodeString(self, s):  # 字符串解码
        stack1 = []
        stack2 = []
        num = 0  # 存储当前数字
        res = ""  # 存储待解码的字符串
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == "[":
                stack1.append(res)
                stack2.append(num)
                res = ""
                num = 0
            elif ch == "]":
                cur_res = stack1.pop()
                cur_num = stack2.pop()
                res = cur_res + res * cur_num
            else:
                res += ch
        return res

    def compress(self, chars):  # 443压缩字符串方法2
        """
        :type chars: List[str]
        :rtype: int
        """
        write, start, read = 0, 0, 0
        n = len(chars)
        for read in range(n):
            if read == n - 1 or chars[read] != chars[read + 1]:
                chars[write] = chars[read]
                write += 1
                ls = read - start + 1
                if ls > 1:
                    numstr = list(str(ls))
                    for i in numstr:
                        chars[write] = i
                        write += 1
                start = read + 1
        return write

    def longestValidParentheses(self, s: str) -> int:  # 最长有效括号
        # 方法一：栈
        # way1: 栈
        stack = [-1]
        res = 0
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if stack:
                    res = max(res, i - stack[-1])
                else:
                    stack.append(i)
        return res


