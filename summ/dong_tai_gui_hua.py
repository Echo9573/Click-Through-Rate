class Solution(object):
    def coinChange(self, amount, coins):  # 零钱兑换：最少硬币数（way1:广度优先搜索，找最短路径）；way2:动态规划O(amount * size)\O(amount)
        # dp[c]:凑成金额c的最少得硬币数量
        dp = [amount + 1 for _ in range(amount + 1)]
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i < coin:
                    continue
                dp[i] = min(dp[i], dp[i - coin] + 1)
        if dp[-1] != amount + 1:
            return dp[-1]
        else:
            return -1

    def num_coinChange(self, amount, coins):  # 零钱兑换2总兑换方案数:动态规划 O(amount * size)\O(amount)
        # dp[c]:凑成金额c的方案总数
        dp = [amount + 1 for _ in range(amount + 1)]
        dp[0] = 0
        for coin in coins:  # ***
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        return dp[amount]

    def climbStairs(self, n): # 方案数
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]

        # # 状态压缩
        # if n == 0 or n == 1: return 1
        # pre, cur = 1, 1
        # for i in range(2, n + 1):
        #     pre, cur = cur, pre + cur
        # return cur

    def climbStairs_k(self, n, k): # 方案数(每次最多能爬k)
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            for s in range(1, k + 1):
                dp[i] += dp[i - k]
        return dp[-1]

    def maxProfit(self, prices): # 股票买卖，只能买一次，选择一天买，在选择一天卖
        minPrice = float('inf')
        maxProfit = 0
        for i in range(len(prices)):
            if prices[i] < minPrice:
                minPrice = prices[i]
            else:
                maxProfit = max(maxProfit, prices[i] - minPrice)
        return maxProfit

    def maxProfit2(self, prices): # 股票买卖，可以买卖多次，每天都可以买/卖，最多有1支股票在手
        res = 0
        for i in range(1, len(prices)):
            res += max(0, prices[i] - prices[i - 1])
        return res

    def maxProfit3(self, prices, K): # 股票买卖，最多买卖K次，最多同时有1支股票在手
        # dp[i][k][0]: 第i天，最大的交易次数上线限制为k，当前手上没有股票的收益, 最终返回dp[N][K][0]
        # 状态转移方程：
        # dp[i][k][0] = max(dp[i - 1][k][1] + prices[i - 1], dp[i - 1][k][0])
        # dp[i][k][1] = max(dp[i - 1][k - 1][0] - prices[i - 1], dp[i - 1][k][1])
        # 第i天不持有的最大利润 = max(i-1天持有但是第i天卖了，第i-1天不持有）
        # 第i天持有的最大利润 = max(i-1天不持有但是第i天买了，第i-1天持有）
        dp = [[[0] * 2 for _ in range(K + 1)] for _ in range(len(prices) + 1)]
        for i in range(len(prices) + 1):
            for k in range(K + 1):
                if i == 0 or k == 0:  # k==0意味着不允许交易 i==0
                    dp[i][k][0] = 0
                    dp[i][k][1] = -prices[0]  # float('-inf') 不允许的情况下，交易不允许发生，因此负无穷
                else:
                    dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i - 1])
                    dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i - 1])
        return dp[-1][-1][0]

    def maxProfit4(self, prices, K): # 股票买卖，有冷冻期1天，无限多次买卖，最多同时有1支股票在手
        # dp[i][0]: 第i天，当前手上没有股票的收益, 最终返回dp[N][0]
        # 状态转移方程：
        # dp[i][0] = max(dp[i - 1][1] + prices[i - 1], dp[i - 1][0])
        # 因为有冷冻期dp[i][1] = max(dp[i - 2][0] - prices[i - 1], dp[i - 1][1])
        dp = [[0] * 2 for _ in range(len(prices) + 1)]
        for i in range(len(prices) + 1):
            if i == 0 or i == 1:
                dp[i][1] = -prices[0]
            else:
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1])
                dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i - 1])
        return dp[-1][0]

    def maxProfit5(self, prices, fee): # 股票买卖，无限多次买卖，每次交易有手续费,最多同时有1支股票在手
        # dp[i][0]: 第i天，当前手上没有股票的收益, 最终返回dp[N][0]
        # 因为有冷冻期，每次买入持有时，减去fee
        dp = [[0] * 2 for _ in range(len(prices) + 1)]
        for i in range(len(prices) + 1):
            if i == 0:
                dp[i][1] = -prices[0] - fee
            else:
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1] )
                dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1] - fee)
        return dp[-1][0]

    def LengthOfLIS_way1(self, nums):  # 最长递增子序列的长度 时间复杂度O(n**2)
        # 方法1：动态规划
        # dp[i] : nums[i]结尾的最长的递增子序列的长度
        # dp[i] = max(dp[j]) + 1 (0<=j <i, 且nums[j] < nums[i])
        dp = len(nums) * [1]
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[j] + 1, dp[i])
        return max(dp)

    def LengthOfLIS_way2(self, nums):  # 最长递增子序列的长度 时间复杂度O(n*log(n))
        # 方法1：贪心 + 二分搜索
        # d[i] 表示长度为i的最长上升子序列的末尾元素的最小值（di是单调递增的）
        def low_bound(nums, k):
            l, r = 0, len(nums) - 1
            loc = r
            while l <= r:
                mid = (l + r) // 2
                if nums[mid] >= k:
                    loc = mid
                    r = mid - 1
                else:
                    l = mid + 1
            return loc

        d = []
        for num in nums:
            if not d or num > d[-1]:
                d.append(num)
            else:
                loc = low_bound(d, num) # 找当前d中第一个大于或等于num的数字(下界)
                d[loc] = num
        return len(d)

    def NumberOfLIS(self, nums):  # 最长递增子序列（LIS）的个数 时间复杂度O(n**2)；空间复杂度O(n)
        # dp[i] 表示以 nums[i] 结尾的最长递增子序列的长度。
        # coun[i] 表示以 nums[i] 结尾的最长递增子序列的个数。
        lens = len(nums)
        dp = [1 for _ in range(lens)]
        coun = [1 for _ in range(lens)]
        for i in range(lens):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] + 1 > dp[i]:  # 更长的递增子序列
                        dp[i] = dp[j] + 1
                        coun[i] = coun[j]
                    elif dp[j] + 1 == dp[i]:  # 两种方案了
                        coun[i] += coun[j]
        max_len = max(dp)
        res = 0
        for i in range(lens):
            if max_len == dp[i]:
                res += coun[i]
        return res

    def longestPalindrome(self, s):  # 最长回文子串
        # # dp[i][j]表示字符串s在[i, j]范围内是否是一个回文串
        # # 状态转移方程 dp[i][j] = dp[i+1][j-1] and (s[i] == s[j])
        n = len(s)
        if n <= 1:
            return s
        dp = [[0 for _ in range(n)] for _ in range(n)]
        start, max_len = 0, 1
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j]:
                    if j - i <= 2:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                if dp[i][j] and (j - i + 1) > max_len:
                    max_len = j - i + 1
                    start = i
        return s[start: start + max_len]

    def longestPalindromeSubseq(self, s):
        # way1: 二维数组
        # dp[i][j]表示为：字符串s在区间[i, j]范围内的最长回文子序列长度
        n = len(s)
        if n <= 1:
            return n
        dp = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])
        return dp[0][n - 1]

        # way2: 状态压缩

    def findLength(self, nums1, nums2):  # 最长重复子数组 的 长度 ok
        # dp[i][j] 表示nums1的前i个元素（下标为i - 1)和nums2的前j个元素（下标为j - 1)的最长重复子数组长度
        max_res = 0
        dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]
        for i in range(1, len(nums1) + 1):
            for j in range(1, len(nums2) + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_res:
                    max_res = dp[i][j]
        return max_res

    def maxSubArray(self, nums):  # 最大子数组的和
        # dp[i]: 所有以nums[i]结尾的子数组的最大的和
        dp = [0 for _ in range(len(nums))]
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        return max(dp)

    def minPathSum(self, grid):  # 二维网格最短路径和ok
        # dp[i][j]: 以grid[i][j]为结尾的最短路径
        m, n = len(grid), len(grid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[-1][-1]

    def minTrianglePath(self, triangle):  # 三角形最小路径和ok
        # dp[i][j] 表示为：从顶部走到第 i 行（从 0 开始编号）、第 j 列的位置时的最小路径和
        s = len(triangle)
        dp = [[0 for _ in range(s)] for _ in range(s)]
        dp[0][0] = triangle[0][0]
        for i in range(1, s):
            dp[i][0] = dp[i - 1][0] + triangle[i][0]  # 每行最左边
            dp[i][i] = dp[i - 1][i - 1] + triangle[i][i]  # 每行最右边
            for j in range(1, i):
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
        return min(dp[-1])

    def uniquePaths(self, m, n):  # 不同路径：二维数组中，从左上角到达右下角的路径数量
        # dp[i][j]:二维数组中，从左上角到位置(i, j)的路径数量
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
        return dp[-1][-1]

    def uniquePathsV2(self, obstacleGrid):  # 不同路径：二维数组中，从左上角到达右下角的路径数量（路上会有障碍物）
        # dp[i][j]:二维数组中，从左上角到位置(i, j)的路径数量
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
        return dp[-1][-1]



    def longestCommonSubsequence(self, text1, text2):  # 最长公共子序列ok
        # dp[i][j]表示text1的前i个元素和text2的前j个元素的最长公共子序列的长度
        m, n = len(text1), len(text2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return dp[m][n]

    def minDistance(self, word1, word2):  # 编辑距离ok
        # dp[i][j] 表示word1的前i个元素和word2的前j个元素，最近编辑距离为dp[i][j]。
        m, n = len(word1), len(word2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[m][n]

    def longestValidParentheses(self, s):  # 最长有效括号：方法1：栈；方法2：动态规划（好麻烦）
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

    def maximalSquare(self, matrix):  # 最大正方形ok
        # dp[i][j] 表示，以矩阵位置(i,j)为右下角，且值包含1的正方形的最大的边长
        import numpy as np
        m, n = len(matrix), len(matrix[0])
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        res = int(np.max(np.array(dp)))
        return res * res

    def rob(self, nums):  # 打家劫舍，相邻房间不能偷，会报警
        # way1: dp[i]表示为：前i间房屋所能偷窃到的最高金额【因为有什么都不做的这个选项，所以加1】
        dp = [0 for _ in range(len(nums) + 1)]
        dp[0] = 0
        dp[1] = nums[0]
        for i in range(2, len(nums) + 1):
            dp[i] = max(dp[i - 2] + nums[i - 1], dp[i - 1])
        return max(dp)

        # # way2: 状态压缩 YYDS
        # pre, cur = 0, 0
        # for i in nums:
        #     pre, cur = cur, max(pre + i, cur)
        # return cur

    def rob2(self, nums):  # 打家劫舍2，房屋围成圈
        if len(nums) == 1:
            return nums[0]
        # 分别求解[0, size - 2]和[1, size -1]两种情况的最大值
        res1 = self.rob(nums[0:len(nums) - 1])
        res2 = self.rob(nums[1:])
        return max(res1, res2)

    def canJump(self, nums):  # 跳跃游戏
        # dp[i]: 从位置0出发，经过j <= i, 可以跳出的最远距离
        # 状态转移方程
        # ①能通过0~i-1个位置到达i，dp[i - 1] >= i: dp[i] = max(dp[i - 1], i + nums[i])
        # ②不能通过0~i-1个位置到达i，dp[i - 1] <= i: dp[i] = dp[i - 1]【其实这里直接返回false就好】
        dp = [0 for _ in range(len(nums))]
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            if dp[i - 1] >= i:
                dp[i] = max(dp[i - 1], i + nums[i])
            else:
                return False
        return dp[-1] >= len(nums) - 1




















































