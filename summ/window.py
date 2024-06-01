class Solution:
    def subarraySum(self, nums, k):  # 和为 K 的子数组
        # 前缀和+哈希表
        # pre[i]为[0, i]中所有数的和，那么[j, i]里所有数的和为k
        # 那么pre[j - 1] == pre[i] - k
        cnt = 0
        presum = 0
        sum_dict = {0: 1}
        for i in range(len(nums)):
            presum += nums[i]
            if presum - k in sum_dict:
                cnt += sum_dict[presum - k]
            sum_dict[presum] = sum_dict.get(presum, 0) + 1
        return cnt

    def shortestSubarray(self, nums, k):  # 和至少为K的最短子数组 ***
        # 方法：前缀和+单调队列
        # presum[i]就表示nums中前i个元素的和，presum[0]就是0，表示没有元素的情况
        # 对于某个区间 [j,i) 来说，如果 presum[i]−pre sum[j]≥k，那么大于 i 的索引值就不用再进行枚举了，不可能比 i−j 的差值更优了。此时我们应该尽可能的向右移动 j，从而使得 i−j 更小。
        # 对于某个区间 [j,i) 来说，如果 presum[j]≥presum[i]，对于任何大于等于 i 的索引值r 来说，presum[r]−presum[i] 一定比presum[i]−presum[j] 更小且长度更小，此时presum[j] 可以直接忽略掉
        presum = [0 for _ in range(len(nums) + 1)]
        for i in range(len(nums)):
            presum[i + 1] = presum[i] + nums[i]
        from collections import deque
        q = deque()
        res = len(nums) + 1
        for i, cursum in enumerate(presum):
            while q and cursum - presum[q[0]] >= k:
                res = min(res, i - q.popleft())
            while q and presum[q[-1]] > cursum:
                q.pop()
            q.append(i)
        return res if res <= len(nums) else -1

    def minSubArrayLen(self, target, nums):  # 长度最小的子数组 找出该数组中满足其总和大于等于target的长度最小的子数组
        # 不定长度的滑动窗口求和
        n = len(nums)
        if n == 0:
            return 0
        res = n + 1
        start, end = 0, 0
        total_sum = 0
        while end < n:
            total_sum += nums[end]
            while total_sum >= target:
                res = min(res, end - start + 1)
                total_sum -= nums[start]
                start += 1
            end += 1
        return 0 if res == n + 1 else res

    def maxSlidingWindow(self, nums, k):  # 滑动窗口的最大值 时间复杂度O(n)
        window, res = [], []
        # 队列中存的是index
        for i, v in enumerate(nums):
            if (i >= k) and (i - k >= window[0]): # 当最左侧是最大值，窗口右移的时候，要弹出
                window.pop(0)
            while window and nums[window[-1]] < v: #维护一个单调栈
                window.pop()
            window.append(i)
            if i >= k - 1: # 开始存结果
                res.append(nums[window[0]])
        return res


    def lengthOfLongestSubstring(self, s):   # 无重复字符的最长子串
        window = {}  # 存储当前窗口内元素的计数情况
        res = 0
        left, right = 0, 0
        while right < len(s):
            # 加元素
            cur = s[right]
            window[cur] = window.get(cur, 0) + 1
            right += 1

            # 减元素，注意，这里是while，不是if，如果不满足，left一直右移
            while window[cur] > 1:
                print(cur, window[cur])
                d = s[left]
                window[d] -= 1
                left += 1

            res = max(res, right - left)
        return res

    def removeDuplicates(self, nums):  # 删除有序数组中的重复项
        slow, fast = 0, 1
        while fast < len(nums):
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        return slow + 1


    def minCoverSubstring(self, s, t):  # 最小覆盖子串 返回s中包含t的所有字符的最小子串
        need = {}
        for i in range(len(t)):
            need[t[i]] = need.get(t[i], 0) + 1
        left, right = 0, 0
        window = {}
        # 记录最小子串的长度
        maxlen = float('inf')
        valid = 0
        start = 0
        while right < len(s):
            x = s[right]
            right += 1
            # 对window做更新，增加元素
            if need.get(x, 0) > 0:  # need[x]:
                window[x] = window.get(x, 0) + 1
                if window[x] == need[x]:
                    valid += 1
            # 对window做更新，收缩窗口
            while valid == len(need.keys()):
                if (right - left) < maxlen:
                    maxlen = right - left
                    start = left  # 需要加个变量存储，否则如果直接存left的话，当长度为1时会有遗漏。
                d = s[left]
                left += 1
                if need.get(d, 0) > 0:
                    # 先判断valid收缩再去减window
                    if window[d] == need[d]:
                        valid -= 1
                    window[d] -= 1
        if maxlen == float('inf'):
            return ""
        else:
            return s[start: start + maxlen]

    def longestOne(self, nums, k):  # 最大连续1的个数 包含0和1的数组，可以允许翻转最多K个0，变成1，则最大的连续的1的个数
        # 滑动窗口（不定窗口）
        left, right = 0, 0
        max_res = 0
        window = {}
        while right < len(nums):
            x = nums[right]
            window[x] = window.get(x, 0) + 1
            right += 1
            while window.get(0, 0) > k:
                if nums[left] == 0:
                    window[0] -= 1
                left += 1
            max_res = max(max_res, right - left)
        return max_res






