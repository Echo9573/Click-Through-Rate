class Solution:
    def twoSum(self, nums, target):  # 两数之和：和为目标值target的两个数字的下标
        # 方法1：遍历一次，哈希表，时间复杂度O(n)：注意，存到dict里面的key：target-nums[i], value是当前index——i
        res_dict = {}
        for i in range(len(nums)):
            if nums[i] in res_dict.keys():
                return [i, res_dict[nums[i]]]
            else:
                res_dict[target - nums[i]] = i
        return []
        # 方法2：双指针 时间复杂度O(n*logn)

    def threeSum(self, nums):  # 三数之和，返回和为0且不重复的三元组
        # 方法1：排序 + 双指针
        if len(nums) < 3:
            return []
        res = []
        nums.sort()
        for i in range(len(nums)):
            target = - nums[i]
            left = i + 1
            right = len(nums) - 1
            temp = []
            if i - 1 >= 0 and nums[i] == nums[i - 1]:
                continue
            while left < right:
                sums = nums[left] + nums[right]
                if sums == target:  # 等价于nums[left] + nums[right] = -nums[i]
                    temp.append([nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while nums[left] == nums[left - 1] and left < right:
                        left += 1
                    while nums[right] == nums[right + 1] and left < right:
                        right -= 1
                elif sums < target:
                    left += 1
                else:
                    right -= 1
            # print(nums[i], temp)
            for x in temp:
                res.append([nums[i]] + x)
        return res

    def  threeSumClosest(self, nums, target):  # 最接近的三数之和
        res = float('inf')
        nums.sort()
        for i in range(len(nums)):
            left = i + 1
            right = len(nums) - 1
            while left < right:
                cursum = nums[left] + nums[right] + nums[i]
                # 用abs来判断距离目标值最近 这一逻辑
                if abs(cursum - target) < abs(res - target):
                    res = cursum
                if cursum < target:
                    left += 1
                else:
                    right -= 1
        return res

    def moveZeroes(self, nums):  # 283移动0 ： 将所有的0移动到数组的末尾，保持非零元素的相对顺序
        slow, fast = 0, 0
        while fast < len(nums):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
            fast += 1
        return nums

    def moveOdd(self, actions):  # 调整数组顺序使奇数位于偶数前面
        slow, fast = 0, 0
        while fast < len(actions):
            if actions[fast] % 2 == 1:
                actions[fast], actions[slow] = actions[slow], actions[fast]
                slow += 1
            fast += 1
        return actions

    def sortColors(self, nums):  # 颜色分类，原地对包含0、1、2的数组进行排序。
        # 双指针 + 快排思想：将序列分成——比基准数小，等于基准数，大于基准数
        left, mid, right = 0, 0, len(nums) - 1
        while mid <= right:
            if nums[mid] == 0:
                nums[mid], nums[left] = nums[left], nums[mid]
                left += 1
                mid += 1
            elif nums[mid] == 2:
                nums[mid], nums[right] = nums[right], nums[mid]
                right -= 1
            else:
                mid += 1

    def removeDuplicates(self, nums):  # 26删除有序数组中的重复项
        slow, fast = 0, 0
        while fast < len(nums):
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        return slow + 1


    def merge(self, nums1, m, nums2, n):  # 88合并两个有序数组nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
        i, j = m - 1, n - 1
        index = m + n - 1
        while i >= 0 and j >= 0:
            if nums1[i] >= nums2[j]:
                nums1[index] = nums1[i]
                i -= 1
            else:
                nums1[index] = nums2[j]
                j -= 1
            index -= 1
        if j >= 0:
            nums1[: j + 1] = nums2[:j + 1]
        return nums1

    def maxArea(self, height):  # 11盛最多水的容器
        left, right = 0, len(height) - 1
        area = 0
        while left < right:
            temp = min(height[left], height[right]) * (right - left)
            area = max(temp, area)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return area


     def triangleNumber(self, nums):  # 有效三角形的个数
         nums.sort()
         res = 0
         for i in range(2, len(nums)):
             left, right = 0, i - 1
             while left < right:
                 if nums[left] + nums[right] <= nums[i]:
                     left += 1
                 else:
                     res += (right - left)
                     right -= 1
         return res

    def reverse(self, chars, left, right):  # 必须增加这个翻转函数
        if left <= right:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1
    def compress(self, chars: List[str]) -> int:  # 443压缩字符串
        # 有三个指针，分别标记，当前写的位置，当前读的新字符的起点，读指针
        write, start, read = 0, 0, 0
        n = len(chars)
        for read in range(n):
            # 有重复的一直在读，并处理
            if read == n - 1 or chars[read] != chars[read + 1]:
                chars[write] = chars[read]
                write += 1
                ls = read - start + 1
                if ls > 1:  # 这里需要有这样一个判断
                    left = write
                    while ls > 0:
                        chars[write] = str(ls % 10)
                        write += 1
                        ls = ls // 10
                    self.reverse(chars, left, write - 1)
                start = read + 1  # 这个不能在ls > 1的逻辑里面
            # print(chars)
        return write









