#include <vector>
using namespace std;
// # 1 494. 目标和
int findTargetSumWays(vector<int>& nums, int target) {
    int sum = 0;
    for (int num : nums) sum += num;
    if (((sum + target) % 2 != 0) || (sum < target) || (sum < -target)) {
        return 0;
    }
    int amount = (sum + target) / 2;
    int n = nums.size();
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;
    for (int i=1; i < n + 1; i++) {
        for (int j=amount; j >= nums[i - 1]; j--) {
            dp[j] += dp[j - nums[i - 1]];
        }
    }
    return dp.back();
}
// # 2 416. 分割等和子集
bool canPartition(vector<int>& nums) {
    int sum = 0;
    for (int num : nums) sum += num;
    if (sum % 2 != 0) return false;
    int amount = sum / 2;
    int n = nums.size();
    vector<int> dp(amount + 1, 0);
    for (int i=1; i < n + 1; i++) {
        for (int j=amount; j >= nums[i - 1]; j--) {
            dp[j] = max(dp[j], dp[j - nums[i - 1]] + nums[i - 1]);
        }
    }
    return dp.back() == amount;
}
// # 3 322. 零钱兑换
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    for (int coin : coins) {
        for (int j = coin; j < amount + 1; j++) {
            dp[j] = min(dp[j], dp[j - coin] + 1);
        }
    }
    if (dp.back() == amount + 1) return -1;
    return dp.back();
}
// # 4 518. 零钱兑换 II
int coinChange2(vector<int>&coins, int amount) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;
    for (int coin : coins) {
        for (int j=coin; j < amount + 1; j++) {
            dp[j] += dp[j - coin];
        }
    }
    return dp.back();
}
