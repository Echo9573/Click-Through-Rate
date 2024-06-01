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

    def moveZeroes(self, nums):  # 移动0 ： 将所有的0移动到数组的末尾，保持非零元素的相对顺序
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
            # print(mid, left, right, nums)
            if mid < left:  # 注意要加这个判断，否则，最后mid会到最左边了
                mid += 1
            elif nums[mid] == 0:
                nums[mid], nums[left] = nums[left], nums[mid]
                left += 1
            elif nums[mid] == 2:
                nums[mid], nums[right] = nums[right], nums[mid]
                right -= 1
            else:
                mid += 1

    def removeDuplicates(self, nums):  # 删除有序数组中的重复项
        slow, fast = 0, 1
        while fast < len(nums):
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        return slow + 1

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









