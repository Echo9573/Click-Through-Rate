class Solution:
    def addStrings(self, num1, num2):   # 字符串相加，返回字符串
        # 字符串加法（模拟加法运算）
        d1 = len(num1) - 1
        d2 = len(num2) - 1
        carry = 0  # 记录是否进位
        res = []
        while carry > 0 or d1 >= 0 or d2 >= 0:
            x = int(num1[d1]) if d1 >= 0 else 0
            y = int(num2[d2]) if d2 >= 0 else 0
            d1 -= 1
            d2 -= 1
            cur_sum = x + y + carry
            res.append(str(cur_sum % 10))
            carry = cur_sum // 10
        return "".join(res)[::-1]

    def subStrings(self, num1, num2):   # 字符串减法，返回字符串
        # 备注：num1 是正数， nums2是负数
        if len(num1) < len(num2) or (len(num1) == len(num2) and num1 < num2):
            num1, num2 = num2, num1
            sign = "-"
        else:
            sign = ""
        # 调整成num1 的绝对值 大于 num2的绝对值，方便下面进行temp_diff处理
        res = []
        borrow = 0
        p1, p2 = len(num1) - 1, len(num2) - 1

        while p1 >= 0 or p2 >= 0:
            x1 = ord(num1[p1]) - ord('0') if p1 >= 0 else 0
            x2 = ord(num2[p2]) - ord('0') if p2 >= 0 else 0
            temp_diff = x1 - x2 - borrow

            if temp_diff < 0:
                temp_diff += 10
                borrow = 1
            else:
                borrow = 0

            res.append(temp_diff)
            p1 -= 1
            p2 -= 1
        print(res)

        while len(res) > 1 and res[-1] == 0:
            res.pop()
        return sign + ''.join(str(x) for x in res[::-1])

    def multiply(self, num1: str, num2: str) -> str:  # 字符串相乘
        if num1 == '0' or num2 == '0':
            return "0"
        m, n = len(num1), len(num2)
        res = [0] * (m + n)  # 性质：两数乘积的结果长度为m + n - 1 或者 m + n
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                product = int(num1[i]) * int(num2[j])
                p1, p2 = i + j, i + j + 1
                sum = product + res[p2]
                res[p1] += sum // 10
                res[p2] = sum % 10
        if res[0] == 0:
            res.pop(0)
        return "".join([str(i) for i in res])

    def compareVersion(self, version1, version2):  # 比较版本号
        from itertools import zip_longest
        for v1, v2 in zip_longest(version1.split("."), version2.split("."), fillvalue=0):
            x, y = int(v1), int(v2)
            if x != y:
                return 1 if x > y else -1
        return 0

    def myAtoi(self, s):  # 字符串转换整数 (atoi)
        # 判断是否空格、符号(+/-)、是否字符串字符（isdigit())、转成数字后是否越界
        num_str = ""  # 有效的用于转换数字的字符串
        start = 0  # 有效的用于转换数字的字符串起点
        sign = 1
        s = s.strip()
        if not s:
            return 0
        if s[0] == '-':
            sign = -1
            start = 1
        elif s[0] == '+':
            sign = 1
            start = 1
        elif not s[0].isdigit():
            return 0

        for i in range(start, len(s)):
            if s[i].isdigit():
                num_str += s[i]
            else:
                break
        if len(num_str) == 0:
            return 0
        num = int(num_str)
        if sign == -1:
            num = -num
            return max(num, -2 ** 31)
        else:
            return min(num, 2 ** 31 - 1)

    def validIPAddress(self, queryIP):  # 验证IP地址（是否合法的IPV4\IPV6)
        # 判断ipv4
        path = queryIP.split(".")
        if len(path) == 4:
            for sub in path:
                if not sub or not sub.isdecimal():
                    return "Neither"
                if sub[0] == "0" and len(sub) != 1:
                    return "Neither"
                if int(sub) > 255:
                    return "Neither"
            return "IPv4"

        # 判断ipv6
        path = queryIP.split(':')
        if len(path) == 8:
            valid = "0123456789abcdefABCDEF"
            for sub in path:
                if not sub:
                    return "Neither"
                if len(sub) > 4:
                    return "Neither"
                for i in sub:
                    if i not in valid:
                        return "Neither"
            return "IPv6"
        return "Neither"

    def calculate(self, s) : # 基本计算器 II
        # 带括号的加减
        ops = [1] # 栈存储操作数用以控制sign这个操作符
        sign = 1
        res = 0 #记录当前的结果
        i = 0
        while i < len(s):
            if s[i] == " ":
                i += 1
            elif s[i] == "+":
                sign = ops[-1]
                i += 1
            elif s[i] == "-":
                sign = -ops[-1]
                i += 1
            elif s[i] == "(":
                ops.append(sign)
                i += 1
            elif s[i] == ")":
                ops.pop()
                i += 1
            else:
                num = 0
                while i < len(s) and s[i].isdigit():
                    num = num * 10 + ord(s[i]) - ord('0')
                    i += 1
                res += num * sign
        return res

    def calculate(self, s):
        # 不带括号的 加减乘除
        stack = []
        index = 0
        op = "+" # 这里必须初始化为+，要不然第一个字符就不会被加进去。
        while index < len(s):
            if s[index] == " ":
                index += 1
                continue
            if s[index].isdigit():
                num = ord(s[index]) - ord('0')
                while index + 1 < len(s) and s[index + 1].isdigit():
                    index += 1
                    num = num * 10 + ord(s[index]) - ord('0')
                if op == "+":
                    stack.append(num)
                elif op == "-":
                    stack.append(-num)
                elif op == "*":
                    top = stack.pop()
                    stack.append(top * num)
                elif op == "/":
                    top = stack.pop()
                    stack.append(int(top / num))
            elif s[index] in "+-*/":
                op = s[index]
            index += 1
        return sum(stack)

    # 带括号的
    # 计算器V3 ——加减乘除和括号
    def calculate(self, s):
        def helper(index):
            stack = []
            op = "+"
            num = 0
            while index < len(s):
                if s[index].isdigit():
                    num = num * 10 + int(s[index])
                if s[index] in "+-*/" or index == len(s) - 1 or s[index] == ")":
                    if op == "+":
                        stack.append(num)
                    elif op == "-":
                        stack.append(-num)
                    elif op == "*":
                        top = stack.pop()
                        stack.append(top * num)
                    elif op == "/":
                        top = stack.pop()
                        stack.append(int(top / num))
                    op = s[index]
                    num = 0
                    if s[index] == ")":
                        break
                if s[index] == "(":
                    num, index = helper(index + 1)
                index += 1
            return sum(stack), index

        result, _ = helper(0)
        return result

    def longestCommonPrefix(self, strs):  # 最长公共前缀
        if len(strs) == 0:
            return ""
        if len(strs) == 1:
            return strs[0]
        minlen = float('inf')
        res = []
        for str in strs:
            minlen = min(len(str), minlen)
        for i in range(minlen):
            cur = strs[0][i]
            for str in strs:
                if str[i] != cur:
                    return strs[0][:i]
        return strs[0][:minlen]  # 注意最后返回这个



