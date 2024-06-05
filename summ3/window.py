class Solution:


    def minSubArrayLen(self, target, nums):  # 209长度最小的子数组(数组中全是正数) 找出该数组中满足其总和大于等于target的长度最小的子数组：时间复杂度o(n)
        # 不定长度的滑动窗口求和
        if len(nums) == 0:
            return 0
        minlen = float('inf')
        left, right = 0, 0
        total_sum = 0
        while right < len(nums):
            cur = nums[right]
            right += 1
            total_sum += cur
            while total_sum >= target:
                d = nums[left]
                left += 1
                minlen = min(minlen, right - left + 1)
                total_sum -= d
        if minlen == float('inf'):
            return 0
        else:
            return minlen

    def shortestSubarray(self, nums, k):  # 862和至少为K的最短子数组 ***类似长度最小的子数组(数组中存在正数和负数)，因此要用前缀和+单调队列进行解决。
        # 时间复杂度O(n)
        # 方法：前缀和+单调队列
        # presum[i]就表示nums中前i个元素的和，presum[0]就是0，表示没有元素的情况
        # 对于某个区间 [j,i) 来说，如果 presum[i]−pre sum[j]≥k，那么大于 i 的索引值就不用再进行枚举了，不可能比 i−j 的差值更优了。此时我们应该尽可能的向右移动 j，从而使得 i−j 更小。
        # 对于某个区间 [j,i) 来说，如果 presum[j]≥presum[i]，对于任何大于等于 i 的索引值r来说，presum[r]−presum[i] 一定比presum[i]−presum[j] 更小且长度更小，此时presum[j] 可以直接忽略掉
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

    def maxSlidingWindow(self, nums, k):  # 滑动窗口的最大值 时间复杂度O(n)
        window, res = [], []
        # 队列中存的是index
        for i in range(len(nums)):
            if (i >= k) and (i - k >= window[0]): # 已经遍历超过k个数据了 & 滑动窗口的第一个元素已经不在窗口范围内
                window.pop(0)
            while window and nums[window[-1]] < nums[i]: #维护一个单调栈
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

            # 减元素，注意，这里是while，不是if，因为这里left不一定是那个重复的元素，因此这里是while
            while window[cur] > 1:
                d = s[left]
                window[d] -= 1
                left += 1

            res = max(res, right - left)
        return res


    def minCoverSubstring(self, s, t):  # 最小覆盖子串 返回s中包含t的所有字符的最小子串
        need = {}
        for i in t:
            need[i] = need.get(i, 0) + 1
        window = {}
        left, right = 0, 0
        valid = 0
        minlen = float("inf")  # 记录最小子串的长度
        start = 0  # 记录输出的结果的起始位置的索引
        while right < len(s):
            # add
            cur = s[right]
            right += 1
            window[cur] = window.get(cur, 0) + 1
            if window[cur] == need.get(cur, 0):
                valid += 1
            # 减
            while valid == len(need.keys()):
                if right - left < minlen:
                    minlen = right - left
                    start = left  # 需要加个变量存储，否则如果直接存left的话，当长度为1时会有遗漏。
                d = s[left]
                left += 1
                if need.get(d, 0) > 0:
                    if window[d] == need[d]:  # 这个一定要加
                        valid -= 1
                    window[d] -= 1
        if minlen == float('inf'):  # 需要加这个判断
            return ""
        else:
            return s[start: start + minlen]

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






