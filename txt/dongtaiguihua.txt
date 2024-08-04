class Solution(object):

    
    def canPartition(self, nums: List[int]) -> bool: # 416. 分割等和子集
        # dp[w] = 从数组中选择一些元素，放入最多能装元素和为w的背包中，得到的元素和最大为多少。
        n = len(nums)
        target = sum(nums) // 2
        if sum(nums) % 2 != 0:
            return False
        W = target
        dp = [0 for _ in range(W + 1)]
        for i in range(1, n + 1):
            for w in range(W, nums[i - 1] - 1, -1):
                dp[w] = max(dp[w], dp[w - nums[i - 1]] + nums[i - 1])
        return dp[W] == target

    def findTargetSumWays(self, nums, S):  # 494. 目标和
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        #原问题可以化成：nums中存在几个子集A，使得A的和为(S + sum(nums))/2
        # sum(A) = (S + sum(nums))/2
        # dp[i][j]只使用nums中的前i个数据,想凑出总和j,有dp[i][j]种凑法
        n = len(nums)
        if (sum(nums)<S) or ((S + sum(nums))%2==1):
            return 0
        amount = (S + sum(nums))/2
        import numpy as np 
        dp = np.zeros([n+1, amount+1])
        for i in range(n+1):
            dp[i][0] = 1

        for i in range(1, n+1):
            for j in range(0, amount+1):
                if (j - nums[i-1]) < 0:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] +  dp[i-1][j-nums[i-1]]
        return int(dp[-1][-1])
        
        # dp[j] 的含义是：使用数组 nums 中的元素，组成和为 j 的方案数 # 注意：本题和零钱兑换2非常相近
        # import numpy as np
        # n = len(nums)
        # if (sum(nums) < S) or ((S + sum(nums)) % 2 == 1):
        #     return 0
        # amount = (S + sum(nums))/2
        # if amount < 0:
        #     return 0
        # dp = [0] * (amount + 1)
        # dp[0] = 1
        # for i in range(1, n+1):
        #     for j in range(amount, nums[i-1] -1, -1):
        #         dp[j] = dp[j] + dp[j-nums[i-1]]
        # return dp[-1]
    def coinChange(self, amount, coins):  # 零钱兑换：最少硬币数（way1:广度优先搜索，找最短路径）；way2:动态规划O(amount * size)\O(amount)
        # dp[i]:凑成金额i的最少得硬币数量
        # way1:先遍历钱币，再遍历金额：
        # dp = [0] + [float('inf')] * amount
        # for coin in coins:
        #     for i in range(coin, amount + 1):
        #         dp[i] = min(dp[i], dp[i - coin] + 1)
        # return dp[-1] if dp[-1] != float('inf') else -1

        # way2:先遍历金额，再遍历钱币：
        # dp = [0] + [float('inf')] * amount
        # for i in range(1, amount + 1):
        #     for coin in coins:
        #         if i >= coin:
        #             dp[i] = min(dp[i], dp[i - coin] + 1)
        # return dp[-1] if dp[-1] != float('inf') else -1

        # way3:二维数组：使用前i种钱币，凑金额j的最少钱币数
        dp = [[float('inf')] * (amount + 1) for _ in range(len(coins) + 1)]
        for i in range(len(coins) + 1):
            dp[i][0] = 0
        for i in range(1, len(coins) + 1):
            for j in range(1, amount + 1):
                dp[i][j] = dp[i - 1][j]
                if j - coins[i - 1] >= 0:
                    dp[i][j] = min(dp[i][j], dp[i][j - coins[i - 1]] + 1)
        return dp[-1][-1] if dp[-1][-1] != float('inf') else -1


    def num_coinChange(self, amount, coins):  # 零钱兑换2总兑换方案数:动态规划 O(amount * size)\O(amount)
        # 本题【必须】应该先遍历物品再遍历背包，如果相反的话，则是排列数
        # 总体思想：先物品后背包
        # dp[i]：凑金额i的方案数 dp[i] = dp[i] + dp[i - coin]
        # dp = [0] * (amount + 1)
        # dp[0] = 1
        # for coin in coins:
        #     for i in range(coin, amount + 1):
        #         dp[i] = dp[i] + dp[i - coin]
        # return dp[-1]

        # dp[i][j]：使用前i种钱币，凑金额j的方案数。
        dp = [[0 for _ in range(amount + 1)] for _ in range(len(coins) + 1)]
        for i in range(len(coins) + 1):
            dp[i][0] = 1
        for i in range(1, len(coins) + 1):
            for j in range(1, amount + 1):
                dp[i][j] = dp[i - 1][j] # 不使用当前硬币
                if j - coins[i - 1] >= 0: # 使用当前硬币
                    dp[i][j] += dp[i][j - coins[i - 1]]
        return dp[-1][-1]

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
    
    # 最多可以进行两次买卖.
    # 确定当前天数下，方法1：一共有几种状态（4种）buy1, sell1, buy2, sell2, 然后确定已知i-1天的状态下，当前是如何通过状态方程得到第i天的状态的；方法2：复用最多交易K次的代码（dp[n][k][0]:第n天，至今最多进行了k次交易，当前手上没有股票的收益，dp[n][k][1]:第n天，至今最多进行了k次交易，当前手上有股票的收益）
    # 方法2：复用最多交易K次的代码
        # 一共有N * K * 2种状态，有以下两类
        # dp[n][k][0]:第n天，至今最多进行了k次交易，当前手上没有股票的收益，最终返回dp[N][K][0]
        # 对应的状态转移方程是：dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
        # dp[n][k][1]:第n天，至今最多进行了k次交易，当前手上有股票的收益
        # 对应的状态转移方程是：dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i])
        # 注意，因为k可以为0，所以是k + 1, 因为i可以是0，所以是n + 1
        # dp = [[[0 for _ in range(len(prices) + 1)] for _ in range(K + 1)] for _ in range(2)]
    def maxProfit23(self, prices: List[int]) -> int:  # 最多可以进行两次买卖. 复用最多交易K次的代码
        K = 2
        dp = [[[0] * 2 for _ in range(K + 1)] for _ in range(len(prices) + 1)]

        for i in range(0, len(prices) + 1):
            for k in range(K + 1):
                if i == 0 or k == 0:
                    dp[i][k][0] = 0
                    dp[i][k][1] = -prices[0]  # 这里是负无穷或者-prices[0]，用以迭代计算dp[1][1][0]
                else:
                    dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i - 1])
                    dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i - 1])
        return dp[-1][-1][0]

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

    def maxProfit5(self, prices, fee): # 714. 买卖股票的最佳时机含手续费，无限多次买卖，每次交易有手续费,最多同时有1支股票在手
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

    def maxEnvelopes(self, envelopes):  # 354. 俄罗斯套娃信封问题
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        import numpy as np
        if len(envelopes)<1:
            return 0
        envelopes.sort(key = lambda x: (x[0], -x[1]))
        y = np.array(envelopes)[:,1]
        return self.LengthOfLIS_way2(y)

    def longestPalindrome(self, s):  # 5. 最长回文子串
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

    def longestPalindromeSubseq(self, s):  # 516. 最长回文子序列
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

    def minPathSum(self, grid):  # 64. 最小路径和 二维网格最短路径和ok
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

    def uniquePathsV2(self, obstacleGrid):  # 不同路径2：二维数组中，从左上角到达右下角的路径数量（路上会有障碍物）
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
    
    def jump(self, nums: List[int]) -> int:  # 45. 跳跃游戏 II
        # # 方法1：动态规划
        # # dp[i]:到达nums索引i所需要的最少的跳跃步数
        # dp = [float('inf') for _ in range(len(nums))]
        # dp[0] = 0 # 注意这个
        # for i in range(1, len(nums)):
        #     for j in range(i):
        #         if nums[j] + j >= i:
        #             dp[i] = min(dp[i], dp[j] + 1)
        #             break # 一旦找到，就跳出，因为dp数组单调增
        # return dp[-1]

        # 方法2：贪心
        # 对于每一个位置 i 来说，所能跳到的所有位置都可以作为下一个起跳点，为了尽可能使用最少的跳跃次数，所以我们应该使得下一次起跳所能达到的位置尽可能的远
        # 就是每次在「可跳范围」内选择可以使下一次跳的更远的位置。这样才能获得最少跳跃次数
        # 如果索引i到达了end边界，则把end置为当前能跳到的最大的距离
        max_pos, end = 0, 0
        steps = 0
        for i in range(len(nums) - 1):
            max_pos = max(max_pos, nums[i] + i)
            if i == end:
                end = max_pos
                steps += 1
        return steps
    
    def maxProduct(self, nums):  # 0152. 乘积最大子数组
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [0 for i in range(len(nums))]
        dp[0] = nums[0]
        dp_min = [0 for i in range(len(nums))]
        dp_min[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i - 1] * nums[i], nums[i], dp_min[i - 1] * nums[i])
            dp_min[i] = min(dp[i - 1] * nums[i], nums[i], dp_min[i - 1] * nums[i])
        return max(dp)
    
    def numDecodings(self, s):  # 0091. 解码方法
        """
        :type s: str
        :rtype: int
        """
        # dp[i]:前i个字符串（s下标i-1)可以有多少种解法
        dp = [0 for i in range(len(s) + 1)]
        dp[0] = 1
        for i in range(1, len(s) + 1):
            if s[i - 1] != "0":
                dp[i] += dp[i - 1]
            if (i > 1) and (int(s[i - 2:i]) <= 26) and (s[i - 2] != "0"):
                dp[i] += dp[i - 2]
        return dp[-1]
    
    def isMatch(self, s, p):  # 10. 正则表达式匹配（.and*）
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        # dp[i][j]:字符串s的前i个字符（下标i-1）,和字符串p的前j个字符（下标j-1）是否匹配
        dp = [[False for j in range(len(p) + 1)] for i in range(len(s) + 1)]
        dp[0][0] = True
        for j in range(1, len(p) + 1):
            if p[j - 1] == "*":
                dp[0][j] = dp[0][j - 2] # 
        for i in range(1, len(s) + 1):
            for j in range(1, len(p) + 1):
                if s[i - 1] == p[j - 1] or p[j - 1] == ".":
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == "*":
                    if p[j - 2] != "." and s[i - 1] != p[j - 2]:
                        dp[i][j] = dp[i][j - 2]
                    else:
                        dp[i][j] = dp[i][j - 1] or dp[i][j - 2] or dp[i - 1][j]
        return dp[-1][-1]

    
    def isMatch2(self, s, p):  # 44. 通配符匹配（?and*)
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        # dp[i][j]:字符串s的前i个字符（下标i-1）,和字符串p的前j个字符（下标j-1）是否匹配
        dp = [[False for j in range(len(p) + 1)] for i in range(len(s) + 1)]
        dp[0][0] = True
        # 默认状态下，两个空字符串是匹配的，即 dp[0][0] = True。
        # 当字符串 s 为空，字符串 p 开始字符为若干个 * 时，两个字符串是匹配的，即 p[j - 1] == '*' 时，dp[0][j] = True。
        for j in range(1, len(p) + 1):
            if p[j - 1] != "*":
                break
            dp[0][j] = True
        for i in range(1, len(s) + 1):
            for j in range(1, len(p) + 1):
                if s[i - 1] == p[j - 1] or p[j - 1] == "?":
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1]  == "*":
                    dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
        return dp[-1][-1]

    def checkValidString(self, s: str) -> bool:  # 678 有效的括号字符串，用的贪心算法
        minCount, maxCount = 0, 0
        n = len(s)
        for i in range(n):
            c = s[i]
            if c == "(":
                minCount += 1
                maxCount += 1
            elif c == ")":
                minCount = max(minCount - 1, 0)
                maxCount -= 1
                if maxCount < 0:
                    return False
            else:
                minCount = max(minCount - 1, 0)
                maxCount += 1
        return minCount == 0
    
    def wordBreak(self, s, wordDict):  # 139. 单词拆分
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        # dp[i]:长度为i的字符串s[0:i]能否拆分成单词。
        dp = [False for _ in range(len(s) + 1)]
        dp[0] = True # 长度为 0 的字符串 s[0: i] 可以拆分为单词，即 dp[0] = True 这个不是很理解
        for i in range(len(s) + 1):
            for j in range(i):
                if dp[j] and s[j : i] in wordDict:
                    dp[i] = True
        return dp[-1]

    def superEggDrop(self, k, n): # 887. 鸡蛋掉落
        # https://leetcode.cn/problems/super-egg-drop/submissions/533789874/
        # 方法3 OK
        dp = [[0 for _ in range(n + 1)] for i in range(k + 1)]
        dp[1][1] = 1

        for i in range(1, k + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1] + 1
                if i == k and dp[i][j] >= n:
                    return j
        return n

    def isInterleave(self, s1, s2, s3):  # 97. 交错字符串
        m, n, t = len(s1), len(s2), len(s3)
        if m + n != t:
            return False
        # 方法一
        # dp[i][j]: s1的前i个元素 和  s2的前j个元素是否能交错成s3的前i + j个元素(注意这里0有意义，因此长度是m+1)
        dp = [[False for _ in range(n + 1)] for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(0, m + 1):
            for j in range(0, n + 1):
                p = i + j - 1
                if i > 0:
                    dp[i][j] = dp[i][j] or ((s1[i - 1] == s3[p]) and dp[i - 1][j])
                if j > 0:
                    dp[i][j] = dp[i][j] or ((s2[j - 1] == s3[p]) and dp[i][j - 1]) 
        return dp[-1][-1]
    



















































