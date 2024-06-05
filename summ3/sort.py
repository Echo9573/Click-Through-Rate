class Solution(object):
    # way1:快排2 使用交换的方式 交换基准数左右的值更快一些——超时
    # 哨兵划分：以第 1 位元素 nums[low] 为基准数，然后将比基准数小的元素移动到基准数左侧，将比基准数大的元素移动到基准数右侧，最后将基准数放到正确位置上
    def partition(self, nums, low, high):
        # 以第 1 位元素为基准数
        pivot = nums[low]
        i, j = low, high
        while i < j:
            # 从右向左找到第 1 个小于基准数的元素
            while i < j and nums[j] >= pivot:
                j -= 1
            # 从左向右找到第 1 个大于基准数的元素
            while i < j and nums[i] <= pivot:
                i += 1
            # 交换元素
            nums[i], nums[j] = nums[j], nums[i]
        # 将基准数放到正确位置上
        nums[j], nums[low] = nums[low], nums[j]
        return j
    def quickSort(self, nums, low, high):
        if low < high:
            # 按照基准数的位置，将数组划分为左右两个子数组
            pivot_i = self.partition(nums, low, high)
            # 对左右两个子数组分别进行递归快速排序
            self.quickSort(nums, low, pivot_i - 1)
            self.quickSort(nums, pivot_i + 1, high)
        return nums
    def sortArray(self, nums):
        return self.quickSort(nums, 0, len(nums) - 1)

    # way2:堆排序  -- 通过
    def __shift_down(self, i, n, nums):
        # i 表示当前节点，非叶子结点
        while 2 * i + 1 < n: # 当前节点至少有左子节点
            left, right = 2 * i + 1, 2 * i + 2
            if 2 * i + 2 >= n:
                larger = left # 当前节点只有左子节点
            else:
                if nums[left] >= nums[right]:
                    larger = left
                else:
                    larger = right
            if nums[i] < nums[larger]:
                nums[i], nums[larger] = nums[larger], nums[i]
                i = larger
            else:
                break
    def __buildMaxHeap(self, nums):
        size = len(nums)
        # for i in range(size):
        #     self.max_heap.append(nums[i])
        # 从最后一个非叶子结点开始进行下移调整
        for i in range((size - 2) // 2, -1, -1):
            self.__shift_down(i, size, nums)
    def maxHeapSort(self, nums):
        # 建立初始堆
        self.__buildMaxHeap(nums)
        size = len(nums)
        for i in range(size - 1, -1, -1):
            # 交换根结点和当前最后一个节点
            nums[0], nums[i] = nums[i], nums[0]
            # 从根结点开始，对当前堆进行下移调整
            self.__shift_down(0, i, nums)
        return nums
    def sortArray(self, nums):
        return self.maxHeapSort(nums)

    # way3:快排2
    # 普通快排-超时
    def patition(self, lists, left, right):
        low = left
        high = right
        key = lists[left]
        while left < right:
            while left < right and lists[right]>=key:
                right -= 1
            lists[left] = lists[right]
            while left < right and lists[left]<=key:
                left += 1
            lists[right] = lists[left]
        lists[right] = key
        return left
    def quickSort(self, lists, left, right):
        if left >= right:
            return lists
        else:
            pi = self.patition(lists, left, right)
            self.quickSort(lists, left, pi-1)
            self.quickSort(lists, pi+1, right)
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        self.quickSort(nums, 0, len(nums) - 1)
        return nums

    # 归并
    # way4: 归并排序 通过
    def merge(self, left, right):
        res = []
        i, j = 0, 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
        res += left[i:]
        res += right[j:]
        return res
    def merge_pop(left, right):
        res = []
        i, j = 0, 0
        while left and right:
            if left[0] <= right[0]:
                res.append(left.pop(0))
            else:
                res.append(right.pop(0))
        res += left
        res += right
        return res
    def mergeSort(self, nums):
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        left_nums = self.mergeSort(nums[0:mid])
        right_nums = self.mergeSort(nums[mid:])
        # return self.merge_pop(left_nums, right_nums)
        return self.merge(left_nums, right_nums)
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        return self.mergeSort(nums)

    # way5: 插入排序——超时
    # 假设前面 n-1(其中 n>=2)个数已经是排好顺序的，现将第 n 个数插到前面已经排好的序列中，然后找到合适自己的位置，使得插入第n个数的这个序列也是排好顺序的。
    # 按照此法对所有元素进行插入，直到整个序列排为有序的过程，称为插入排序
    # https://baijiahao.baidu.com/s?id=1773264625586932633&wfr=spider&for=pc
    def insertSort(self, nums):
        for i in range(1, len(nums)):
            key = nums[i] # 当前待插入处理的值
            j = i
            while j > 0 and nums[j - 1] > key:
                nums[j] = nums[j - 1]
                j -= 1
            nums[j] = key # 将待处理的元素插入到适当的位置
        return nums

    def sortArray(self, nums):
        return self.insertSort(nums)

    # way6: 冒泡——超时
    def bubbleSort(self, nums):
        for i in range(len(nums)-1):
            flag = False # 直到某一趟排序过程中不出现元素交换位置的动作，则排序结束
            for j in range(len(nums) - 1 - i):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
                    flag = True
            if not flag:
                break
        return nums
    def sortArray(self, nums):
        return self.bubbleSort(nums)

    # way7: 选择排序——超时
    def selectSort(self, nums):
        for i in range(len(nums) - 1):
            index = i # 记录当前循环最小值位置
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[index]:
                    index = j
            # 将最小值的位置和当前循环的第一个（i)位置换
            if i != index:
                nums[i], nums[index] = nums[index], nums[i]
        return nums
    def sortArray(self, nums):
        return self.selectSort(nums)

    # way8: 希尔排序 通过
    def shellSort(self, nums):
        size = len(nums)
        gap = size // 2
        while gap > 0:
            for i in range(gap, size):
                temp = nums[i]
                j = i
                while j >= gap and nums[j-gap] > temp:
                    nums[j] = nums[j - gap]
                    j -= gap
                nums[j] = temp
            gap = gap // 2
        return nums
    def sortArray(self, nums):
        return self.shellSort(nums)

    # way9: 桶排序——通过
    def insertSort(self, nums):
        for i in range(1, len(nums)):
            j = i
            key = nums[j]
            while j > 0 and key < nums[j-1]:
                nums[j] = nums[j-1]
                j -= 1
            nums[j] = key
        return nums
    def bucketSort(self, nums, bucket_size=5):
        n_max, n_min = max(nums), min(nums)
        n_bucket = (n_max - n_min) // bucket_size + 1 # 桶个数
        buckets = [[] for _ in range(n_bucket)]
        # 分配元素
        for num in nums:
            buckets[(num - n_min)//bucket_size].append(num)
        # 对每个桶单独排序
        res = []
        for buck in buckets:
            self.insertSort(buck)
            res.extend(buck)
        return res
    def sortArray(self, nums):
        return self.bucketSort(nums)

    # 计数排序——
    def counting_sort(self, arr):
        max_val = max(arr)
        min_val = min(arr)
        range_of_elements = max_val - min_val + 1
        count = [0] * range_of_elements
        output = [0] * len(arr)
        # 计算每个元素的出现次数
        for i in range(len(arr)):
            count[arr[i] - min_val] += 1
        # 计算每个元素在输出数组中的位置
        for i in range(1, len(count)):
            count[i] += count[i - 1]
        # 将元素放入输出数组中的正确位置
        for i in range(len(arr) - 1, -1, -1):
            index = arr[i] - min_val
            count_pos = count[index]
            output[count_pos - 1] = arr[i]
            count[index] -= 1
        return output

    def sortArray(self, nums):
        return self.counting_sort(nums)



    def findKthLargest_way1(self, nums, k): # 数组中的第K个最大元素
        # 方法1：
        import heapq
        heap = []
        for num in nums:
            if len(heap) < k:
                heapq.heappush(heap, num)
            else:
                if num > heap[0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, num)
        return heap[0]

        # 方法2：
    def findKthLargest_way2(self, nums, k): # 平均时间复杂度是O(n)
        def quick_select(nums, k):
            # 随机选择基准数
            import random
            pivot = random.choice(nums)
            big, equal, small = [], [], []
            # 将大于、小于、等于 pivot 的元素划分至 big, small, equal 中
            for num in nums:
                if num > pivot:
                    big.append(num)
                elif num < pivot:
                    small.append(num)
                else:
                    equal.append(num)
            if k <= len(big):
                # 第 k 大元素在 big 中，递归划分
                return quick_select(big, k)
            if len(nums) - len(small) < k:
                # 第 k 大元素在 small 中，递归划分
                return quick_select(small, k - len(nums) + len(small))
            # 第 k 大元素在 equal 中，直接返回 pivot
            return pivot

        return quick_select(nums, k)

    # # 方法3：超时 40/41
    def partition(self, nums, left, right):
        key = nums[left]
        while left < right:
            while nums[right] >= key and left < right:
                right -= 1
            nums[left] = nums[right]
            while nums[left] <= key and left < right:
                left += 1
            nums[right] = nums[left]
        nums[left] = key
        return left
    def findKthLargest_way3(self, nums, k):
        left, right = 0, len(nums) - 1
        k1 = len(nums) - k
        if len(nums) == 1:
            return nums[0]
        while left <= right:
            pivot = self.partition(nums, left, right)
            if pivot < k1:
                left = pivot + 1
            elif pivot > k1:
                right = pivot - 1
            else:
                return nums[pivot]

    def mergeTwo(self, nums1, m, nums2, n):  # 88合并两个有序数组：nums2到nums1中，使得合并后的结果非递减排列
        i, j = m - 1, n - 1
        # 总
        index = m + n - 1
        # 从nums1和nums2的后面往前面比较，并且将更大的元素填写到nums1尾部的位置。
        while i >= 0 and j >= 0:
            if nums1[i] < nums2[j]:
                nums1[index] = nums2[j]
                j -= 1
            else:
                nums1[index] = nums1[i]
                i -= 1
            index -= 1
        if j >= 0:
            nums1[:j + 1] = nums2[:j + 1]
        return nums1

    def mergeRange(self, intervals):  # 合并区间
        # 思路：先按照左边界排序，再判断结果集中最后一个元素的右边界和当前元素的左边界的关系
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        res = []
        for interval in intervals:
            if not res or res[-1][1] < interval[0]:
                res.append(interval)
            else:
                res[-1][1] = max(interval[1], res[-1][1])
        return res

    def largestNumber(self, nums):  #  179最大数
        from functools import cmp_to_key
        # 自定义比较函数，比较两个数字的拼接结果
        def compare(x, y):
            return int(y + x) - int(x + y)
        # 将数组中的数字转换为字符串
        nums = list(map(str, nums))
        # 使用自定义比较函数对数组进行排序
        nums.sort(key=cmp_to_key(compare))
        # 将排序后的数组拼接成字符串
        result = ''.join(nums)
        # 如果结果字符串以0开头，说明所有数字都是0，返回"0"；否则返回结果字符串
        return '0' if result[0] == '0' else result




