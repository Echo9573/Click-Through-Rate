class Solution:
    # 回溯算法框架
    # def trackback(路径，选择列表):
    #     if 满足结束条件：
    #         result.add(路径)
    #         return
    #     for 选择 in 选择列表：
    #         剔除不满足条件的选择
    #         做选择
    #         backtrack(路径，选择列表)
    #         撤销选择
    def permute(self, nums):  # 46全排列(nums中没有重复数字)
        def traceback(path, num):
            if len(path) == self.n:
                # 备注，这里如果append track的话，实际上是在self.res中添加了对track的引用，而不是track的副本。
                # 因此，当你在后面的代码中修改track时，self.res中的所有列表也会被修改。
                self.res.append(path[:])
            for i in range(len(num)):
                # if nums[i] in track:
                #     continue
                traceback(path + [num[i]], num[:i] + num[i + 1:])
        self.res = []
        self.n = len(nums)
        traceback([], nums)
        return self.res
    


    def permuteUnique(self, nums):  # 47全排列2(nums中可能存在重复数字), 返回所有不重复的全排列，时间复杂度O(n*n!)
        def traceback(path, num):
            if len(path) == self.n:
                self.res.append(path[:])
            for i in range(len(num)):
                if i > 0 and num[i] == num[i - 1]:
                    continue
                traceback(path + [num[i]], num[:i] + num[i + 1:])
        self.res = []
        nums.sort()
        self.n = len(nums)
        traceback([], nums)
        return self.res


    def subsets(self, nums):  # 返回数组中素有可能的子集
        def traceback(path, startindex):
            self.res.append(path[:])
            for i in range(startindex, self.n):
                traceback(path + [nums[i]], i + 1)
        self.res = []
        self.n = len(nums)
        traceback([], 0)
        return self.res


    def combinationSum(self, candidates, target):  # 组合总和，候选集中，所有和为target的组合（候选集中的数字可以重复获取）
        def traceback(total, path, startindex):
            if total > target:
                return
            if total == target:
                self.res.append(path[:])
            for i in range(startindex, len(candidates)):
                traceback(total + candidates[i], path + [candidates[i]], i)

        self.res = []
        traceback(0, [], 0)
        return self.res


    def combinationSum2(self, candidates, target):  # 组合总和，候选集中，所有和为target的组合（候选集中的数字一次组合只能用一次）
        def traceback(total, path, startindex):
            if total > target:
                return
            if total == target:
                self.res.append(path[:])
            for i in range(startindex, len(candidates)):
                if i > startindex and candidates[i] == candidates[i - 1]:
                    continue
                traceback(total + candidates[i], path + [candidates[i]], i + 1)

        self.res = []
        candidates.sort()
        traceback(0, [], 0)
        return self.res


    def restoreIpAddresses(self, s):  # 复原IP地址 判断是否有可能是有效的IP地址
        def trackback(index, track):
            if len(track) == 4 and index == self.n:
                self.res.append(".".join(track))
            for i in range(index, self.n):
                cur = s[index: i + 1]
                if int(cur) > 255:
                    continue
                if int(cur) > 0 and cur[0] == "0":
                    continue
                if int(cur) == 0 and len(cur) > 1:
                    continue
                track.append(cur)  # 注意这里是加入cur 这里需要是i+1，不能是index+1，需要对index做更新
                trackback(i + 1, track)
                track.pop()

        self.res = []
        self.n = len(s)
        track = []
        trackback(0, track)
        return self.res

    def generateParenthesis(self, n): # 22括号生成
        def traceback(left, right, track):
            # 终止条件
            if len(track) == 2 * self.n:
                self.res.append(track[:])
            # 跟踪到目前为止放置的左括号和右括号的数目来做到这一点
            # 如果左括号数量不大于 n，我们可以放一个左括号.
            # 如果右括号数量小于左括号的数量，我们可以放一个右括号
            if left < n:
                traceback(left + 1, right, track + "(")
            if right < left:
                traceback(left, right + 1, track + ")")

        self.n = n
        self.res = []
        track = ""
        traceback(0, 0, track)
        return self.res


    def exist(self, board, word):  # 79单词搜索
        # 定义表示从board[i][j]开始搜索，能不能搜到word[index]以及index位置之后的后缀子串（上下左右）
        # 如果能搜到，返回True， 否则返回False
        def backtrack(i, j, index):
            # 结束条件，index到达末尾了，直接判断当前位置是不是等于该位置，不需要再下钻搜索了
            if index == len(word) - 1:
                return board[i][j] == word[index]
            if board[i][j] == word[index]:
                # 记得增加撤销选择的机制
                temp, board[i][j] = board[i][j], "_"
                for direct in self.directs:
                    new_i = i + direct[0]
                    new_j = j + direct[1]
                    if 0 <= new_i < self.row and 0 <= new_j < self.col:
                        if backtrack(new_i, new_j, index + 1):
                            return True # 备注：只有true时才返回
                board[i][j] = temp
            else:
                return False

        self.directs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 每个位置，四种选择
        if len(board) == 0:
            return False
        self.row, self.col = len(board), len(board[0])
        for i in range(self.row):
            for j in range(self.col):
                if backtrack(i, j, 0): # 备注：只有true时才返回
                    return True
        return False

    def judgePoint24(self, cards):   # 679.24 点游戏(hard)
        def backtrack(cards):
            if not cards:
                return False
            if len(cards) == 1:
                return abs(cards[0] - self.target) < self.diff
            for i, x in enumerate(cards):
                for j, y in enumerate(cards):
                    if i != j:
                        # 剩下的没被选中的单独放在一个列表里
                        tempcard = []
                        for k, z in enumerate(cards):
                            if k != i and k != j:
                                tempcard.append(z)
                        for act in self.choose:
                            # 加法和乘法满足交换律
                            if act in ["+", "*"] and i > j:
                                continue
                            if act == "+":
                                tempcard.append(x + y)
                            if act == "-":
                                tempcard.append(x - y)
                            if act == "*":
                                tempcard.append(x * y)
                            if act == "/":
                                # 判断不合法的除数
                                if abs(y) < self.diff:
                                    continue
                                tempcard.append(x / y)
                            if backtrack(tempcard):
                                return True
                            tempcard.pop()  # 撤销选择
            return False

        self.target = 24
        self.diff = 1e-6  # 因为有精度的影响
        self.choose = ["+", "-", "*", "/"]
        return backtrack(cards)


    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        def valid(row, col, k, board):
            # 行判断
            for i in range(9):
                if board[i][col] == str(k):
                    return False
            # 列判断
            for j in range(9):
                if board[row][j] == str(k):
                    return False
            # 九宫格判断
            start_row = (row // 3) * 3
            start_col = (col // 3) * 3
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    if board[i][j] == str(k):
                        return False
            return True

        def backtrack(board):
            for i in range(self.row):
                for j in range(self.col):
                    if board[i][j] != ".":
                        continue
                    # 做选择
                    for k in range(1, 10):
                        if valid(i, j, k, board):
                            board[i][j] = str(k)
                            # 终止条件，一旦遇到满足条件的返回
                            if backtrack(board):
                                return True
                            board[i][j] = '.'
                    # 这里的return False 不能放在for i 循环外侧，会提示超时，不能省略
                    return False
            # 备注：这一行必须加上，否则有问题，判断第一行第一列不满足直接就返回了，不再检查其他行列
            return True

        # 比较麻烦，每一行、每一列、每个数字都需要一个for循环，去判断和检验
        self.row, self.col = len(board), len(board[0])
        backtrack(board)












