class Solution:
    def nextPermutation(self, nums):  # 数组中，下一个更大数，原地修改
        """
        Do not return anything, modify nums in-place instead.
        """
        # 将后面的大数和前面的小数交换
        # 希望增加的幅度尽可能小
        #     - 尽可能靠右的低位进行交换（从后向前）
        #     - 尽可能小的大数和前面的小数做交换
        #     - 大数换到前面后，需要将大数后面的所有数升序

        # 1、从后向前 查找第一个 相邻升序 的元素对 (i,j)，满足 A[i] < A[j]。此时 [j,end) 必然是降序
        # 2、在 [j,end) 从后向前 查找第一个满足 A[i] < A[k] 的 k。A[i]、A[k] 分别就是上文所说的「小数」、「大数」
        # 3、将 A[i] 与 A[k] 交换
        # 4、可以断定这时 [j,end) 必然是降序，逆置 [j,end)，使其升序
        # 5、如果在步骤 1 找不到符合的相邻元素对，说明当前 [begin,end) 为一个降序顺序，则直接跳到步骤 4
        # 方法2：while循环
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i < 0:
            nums[:] = nums[::-1]
            return nums
        j = len(nums) - 1
        while j >= 0 and nums[i] >= nums[j]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1:] = nums[i + 1:][::-1]

        # res = int("".join(nums))
        return nums


    def rand10(self): # 用rand7实现rand10
        import random
        def rand7():
            return random.randint(1, 7)
        # # 方法1：古典概型
        # while True:
        #     m = rand7()
        #     if m <= 6:
        #         break
        # while True:
        #     n = rand7()
        #     if n <= 5:
        #         break
        # # 备注：判断奇数和偶数的方法还可以用m & 2 == 1:奇数，否则偶数
        # # 奇数/偶数的概率：1/2; 取到每个数的概率是1/5, 则最终每个数的概率是1/10
        # # 为了得到 11，我们可以在 first 是偶数且 second 是 5 的情况下返回 11(概率也是1/10)
        # return n if (m % 2 == 1) else n + 5
        # 方法2：拒绝采样
        while True:
            row = rand7()
            col = rand7()
            cur = (row - 1) * 7 + col
            if cur <= 40:
                return 1 + (cur - 1) % 10

    def trap(self, height):  # 接雨水
        if len(height) <= 2:
            return 0
        n = len(height)
        res = 0
        l_max = [0] * n
        r_max = [0] * n
        l_max[0] = height[0]
        r_max[n - 1] = height[n - 1]
        for i in range(1, n):
            l_max[i] = max(height[i], l_max[i - 1])
        for i in range(n - 2, -1, -1):
            r_max[i] = max(height[i], r_max[i + 1])
        for i in range(1, n - 1):
            res = res + min(l_max[i], r_max[i]) - height[i]
        return res

    def longestConsecutive(self, nums):  # 128最长连续序列  # 数字连续的最长序列（不要求序列元素在原数组中连续）的长度
        # 哈希表，注意这里的时间复杂度是o(n)
        longest_res = 0
        nums_set = set(nums)  # 转成set 为了后续方便查找元素
        for num in nums:
            if num - 1 not in nums_set:  # 说明是一个新的序列的开始, 注意，这里判断一定要在set中去做，否则超时
                cur = num
                cur_res = 1
                while cur + 1 in nums_set:
                    cur += 1
                    cur_res += 1
                longest_res = max(cur_res, longest_res)
        return longest_res

    def longestIncreasingPath(self, matrix):  # 矩阵中最长的递增路径
        def dfs(row, col):
            if cache[row][col] != 0:
                return cache[row][col]
            max_len = 1
            all_direct = [(-1, 0), (1, 0), (0, 1), (0, -1)]
            for x, y in all_direct:
                new_row, new_col = row + x, col + y
                if 0 <= new_row < m and 0 <= new_col < n and matrix[new_row][new_col] > matrix[row][col]:
                    max_len = max(max_len, dfs(new_row, new_col)) + 1
            cache[row][col] = max_len
            return max_len

        result = 0
        m, n = len(matrix), len(matrix[0])
        cache = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                result = max(result, dfs(i, j))
        return result


