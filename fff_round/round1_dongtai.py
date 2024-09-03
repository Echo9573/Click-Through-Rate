class Dongtaiguihua:
; ================================================================================================================================
# 动态规划
# 1 494. 目标和
# 2 416. 分割等和子集
# 3 322. 零钱兑换
# 4 518. 零钱兑换 II
# 5 70. 爬楼梯（题目延伸）
# 6 509. 斐波那契数
# 7 121.买卖股票的最佳时机
# 8 122.买卖股票的最佳时机 II
# 9 123.买卖股票的最佳时机 II
# 10 188. 买卖股票的最佳时机 IV
# 11 309. 买卖股票的最佳时机含冷冻期
# 12 714. 买卖股票的最佳时机含手续费
# 13 300. 最长递增子序列(长度）
# 14 673. 最长递增子序列的个数
# 15 354. 俄罗斯套娃信封问题
# 16 5. 最长回文子串
# 17 516. 最长回文子序列
# 18 718. 最长重复子数组
# 19 53. 最大子数组和
# 20 1143. 最长公共子序列
# 21 72. 编辑距离
# 22 32. 最长有效括号
# 23 221. 最大正方形
# 24 64. 最小路径和
# 25 120. 三角形最小路径和
# 26 62. 不同路径
# 27 63. 不同路径 II
# 28 0152. 乘积最大子数组
# 29 198. 打家劫舍
# 30 0213. 打家劫舍 II
# 31 0091. 解码方法
# 32 10. 正则表达式匹配（.and*)
# 33 44. 通配符匹配（?and*)
# 34 678. 有效的括号字符串
# 35 139. 单词拆分
# 36 97. 交错字符串
# 37 55. 跳跃游戏
# 38 45. 跳跃游戏 II
# 39 1306. 跳跃游戏 III
# 40 1345. 跳跃游戏IV
# 41 1696. 跳跃游戏VI
; ===================================
# 1 494. 目标和
def findTargetSumWays(self, nums, target):
    # s+ - s- = sum(nums)
    # s+ + s- = target
    # 2 * s+ = sum(nums) + target
    if (sum(nums) < target) or (sum(nums) < -target) or (sum(nums) + target) % 2 != 0:
        return 0
    amount = (sum(nums) + target) // 2
    dp = [0 for i in range(amount + 1)]
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(amount, nums[i - 1] - 1, -1):
            dp[j] += dp[j - nums[i - 1]]  # !!!这里不能加1
    return dp[-1]
# 2 416. 分割等和子集
def canPartition(nums):
    if sum(nums) % 2 != 0:
        return False
    amount = sum(nums) // 2
    n = len(nums)
    dp = [0 for i in range(amount + 1)]
    for i in range(1, n + 1):
        for j in range(amount, nums[i - 1] - 1, -1):
            dp[j] = max(dp[j], dp[j - nums[i - 1]] + nums[i - 1])
    if dp[-1] == amount:
        return True
    else:
        return False
# 3 322. 零钱兑换
def coinChange(coins, amount):
    dp = [amount + 1 for _ in range(amount + 1)]
    dp[0] = 0
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] = min(dp[j], dp[j - coin] + 1)
    if dp[-1] == amount + 1:
        return -1
    return dp[-1]
# 4 518. 零钱兑换 II
def coinChange2(coins, amount):
    dp = [0 for _ in range(amount + 1)]
    dp[0] = 1
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] += dp[j - coin]
    return dp[-1]
# 5 70. 爬楼梯（题目延伸）
def climbStairs(n):
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[-1]
# 6 509. 斐波那契数
def fibo(n):
    if n == 0 or n == 1:
        return 1
    pre, cur = 1, 1
    for i in range(2, n + 1):
        pre, cur = cur, pre + cur
    return cur
def maxprofit(prices):
    maxprofit = 0
    minprice = float('inf')
    for i in range(len(prices)):
        if price[i] < minprice:
            minprice = price[i]
        else:
            maxprofit = max(maxprofit, price[i] - minprice)
    return maxprofit
def maxprofit2(prices):
    maxprofit = 0
    for i in range(1, len(prices)):
        if price[i] - price[i - 1] > 0:
            maxprofit += price[i] - price[i - 1]
    return maxprofit
def maxprofit3(prices, k):
    N = len(price)
    dp = [[[0] * 2 for _ in range(K + 1)] for _ in range(N + 1)]
    for i in range(N + 1):
        for k in range(K + 1):
            if i == 0 or k == 0:
                dp[i][k][0] = 0
                dp[i][k][1] = -price[0]
            else:
                dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + price[i-1])
                dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - price[i-1])
    return dp[-1][-1][0]
def maxProfit4_freeze(prices): # 股票买卖，有冷冻期1天，无限多次买卖，最多同时有1支股票在手
    dp = [[0] * 2 for _ in range(len(prices) + 1)]
    for i in range(len(prices) + 1):
        if i == 0 or i == 1:
            dp[i][1] = -prices[0]
        else:
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1])
            dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i - 1])
    return dp[-1][0]
def maxProfit5_fee(prices): # 股票买卖，有手续费
    dp = [[0] * 2 for _ in range(len(prices) + 1)]
    for i in range(len(prices) + 1):
        if i == 0 or i == 1:
            dp[i][1] = -prices[0] - fee
        else:
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1] - fee)
    return dp[-1][0]

# 13 300. 最长递增子序列(长度）

# 14 673. 最长递增子序列的个数

# 15 354. 俄罗斯套娃信封问题

# 16 5. 最长回文子串

# 17 516. 最长回文子序列

# 18 718. 最长重复子数组

# 19 53. 最大子数组和

# 20 1143. 最长公共子序列

# 21 72. 编辑距离

# 22 32. 最长有效括号

# 23 221. 最大正方形

# 24 64. 最小路径和

# 25 120. 三角形最小路径和

            


            

 
