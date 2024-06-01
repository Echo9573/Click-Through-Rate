class Solution:
    def search(self, nums, target):  # 已排序数组中，二分搜索查找target
        # # [left, right]
        # left, right = 0, len(nums) - 1
        # while left <= right:
        #     mid = left + (right - left) // 2
        #     if nums[mid] == target:
        #         return mid
        #     elif nums[mid] > target:
        #         right = mid - 1
        #     else:
        #         left = mid + 1
        # return -1

        # [left, right)
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid # 注意这里因为是右边开区间，这里必须是mid，而不是mid - 1
            else:
                left = mid + 1
        return -1

    def low_bound(self, nums, k):  # 找当前nums中第一个大于或等于k的数字的索引(下界)
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

    def right_loc(self, nums, target):
        # 右边界
        left, right = 0, len(nums) - 1
        while left <= right:  # 闭区间
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        # 因为退出条件是left = right + 1，target 比所有元素都小时，right会被减到-1
        if right < 0 or nums[right] != target:
            return -1
        return right

    def left_loc(self, nums, target):
        # 左边界
        left, right = 0, len(nums) - 1
        while left <= right:  # 闭区间
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        # 因为退出条件是left = right + 1，所以当target 比 nums中所有的数都大时，left >=len(nums)
        if left >= len(nums) or nums[left] != target:
            return -1
        return left

    def searchRange(self, nums, target):  # 在排序数组中查找元素的第一个和最后一个位置
        if len(nums) == 0:
            return [-1, -1]
        right_ = self.right_loc(nums, target)
        left_ = self.left_loc(nums, target)
        return [left_, right_]

    def rotateMin(self, nums):  # 寻找旋转排序数组中的最小值(向右旋转）:数字没有重复
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]

    def rotateMin_v2(self, nums):  # 寻找旋转排序数组中的最小值(向右旋转）:数字可能重复
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            elif nums[mid] < nums[right]:
                right = mid
            else:
                # nums[mid]==nums[right]，无法判断在 mid 的哪一侧
                # 可以采用 right = right - 1 逐步缩小区域
                right = right - 1
        return nums[left]

    def search_rotate(self, nums, target):  # 搜索旋转排序数组中target的索引
        left, right = 0, len(nums) - 1
        # 解题思路，先找到当前mid属于左半部分还是右半部分，然后确定左右区间
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] >= nums[left]:
                # target在左半部分中间(left < target_i < mid)
                if nums[mid] > target >= nums[left]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # target在左半部分中间(mid < target_i < right)
                if nums[right] >= target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

    def findPeakElement(self, nums):
        # 因为nums[-1] = nums[n] = -∞，所以下面解法可以解决。
        # 如果nums[mid] 小于nums[mid + 1]，说明峰值在mid右侧，因此将left更新为mid + 1。
        # 否则，说明峰值在mid左侧（或mid本身就是峰值），将right更新为mid。
        left = 0
        right = len(nums) - 1  # 注意这里是闭区间
        # 这里是left<right，猜测是因为这里终止条件是left=right,但是nums[n] = -∞不可能
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left

    def search_left_bound(self, arr, target):  # 找到最后一个不大于目标值的元素
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right if right >= 0 else -1

    def searchMatrix(self, matrix, target):  #  限制条件：从左到右升序，每行的第一个整数大于前一行的最后一个整数
        # 先对矩阵的第一列开始搜索，找到最后一个不大于目标值的元素，然后对这个元素所在的行进行二分查找。
        first_row = [matrix[i][0] for i in range(len(matrix))]
        row_index = self.search_left_bound(first_row, target)
        index = self.search_left_bound(matrix[row_index], target)
        if matrix[row_index][index] == target:
            return True
        else:
            return False

    def searchMatrix2(self, matrix, target):  # 无限制条件，从左到右，从上到下都是升序
        # Z字型解法 时间复杂度O(m + n)，空间复杂度O(1)
        m, n = len(matrix), len(matrix[0])
        x, y = 0, n - 1
        while x < m and y >= 0:
            if matrix[x][y] == target:
                return True
            if matrix[x][y] > target:
                y -= 1
            else:
                x += 1
        return False

    def mySqrt(self, x):  # x的平方根
        left, right = 0, x
        res = 0
        while left <= right:
            mid = (left + right) // 2
            if mid * mid <= x:
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    def myPow(self, x, n):  # pow(x, n)
        # 递归
        def helper(n):
            if n == 0:
                return 1.0
            y = helper(n // 2)
            if n % 2 == 0:
                return y * y
            else:
                return y * y * x

        if n >= 0:
            return helper(n)
        else:
            return 1.0 / helper(-n)

    def findDuplicate(self, nums):  # 找出nums([1, n])中唯一一个重复的元素
        # 二分搜索: 利用性质，位置i如果有重复，则nums中小于等于i的元素个数大于i
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) >> 1
            cnt = 0
            for num in nums:
                cnt += int(num <= mid)
            if cnt <= mid:
                left = mid + 1
            else:
                right = mid  # mid 有可能是我们需要的值，因此，这里搜索范围right=mid
        return left









