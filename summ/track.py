class Solution:
    # 回溯算法框架
    # def trackback(路径，选择列表):
    #     if 满足结束条件：
    #         result.add(路径)
    #         return
    #     for 选择 in 选择列表：
    #         剔除不满足条件的选择
    #         做选择
    #         backtrack(路径，选择列表)
    #         撤销选择
    def permute(self, nums):  # 全排列(nums中没有重复数字)
        def trackback1(nums, track):
            if len(track) == self.n:
                # 备注，这里如果append track的话，实际上是在self.res中添加了对track的引用，而不是track的副本。
                # 因此，当你在后面的代码中修改track时，self.res中的所有列表也会被修改。
                self.res.append(track[:])
            for i in range(len(nums)):
                if nums[i] in track:
                    continue
                track.append(nums[i])
                trackback1(nums, track)
                track.pop()

        self.res = []  # 这个是贯穿的变量，所以需要是类的成员变量
        self.n = len(nums)  # 用来记录最后的终止条件
        track = []  # 这个是中间变量，可能有更改，所以是临时变量
        trackback1(nums, track)
        return self.res


    def permuteUnique(self, nums):  # 全排列2(nums中可能存在重复数字), 返回所有不重复的全排列，时间复杂度O(n*n!)
        def trackback1(nums, track):
            if len(track) == self.n:
                self.res.append(track[:])
                return
            for i in range(len(nums)):
                # 如果当前元素和前一个元素相同，跳过当前元素
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                track.append(nums[i])
                trackback1(nums[:i] + nums[i + 1:], track)
                track.pop()

        nums.sort()  # 先对 nums 进行排序
        self.res = []  # 这个是贯穿的变量，所以需要是类的成员变量
        self.n = len(nums)  # 用来记录最后的终止条件
        track = []  # 这个是中间变量，可能有更改，所以是临时变量
        trackback1(nums, track)
        return self.res


    def subsets(self, nums):  # 返回数组中素有可能的子集
        def traceback(nums, index):
            # index用于控制当前的选择列表，[index:]
            self.res.append(self.path[:])
            if index >= self.n:
                return
            for i in range(index, self.n):
                self.path.append(nums[i])
                traceback(nums, i + 1)
                self.path.pop()

        self.res = []
        self.path = []
        self.n = len(nums)
        traceback(nums, 0)
        return self.res


    def combinationSum(self, candidates, target):  # 组合总和，候选集中，所有和为target的组合（候选集中的数字可以重复获取）
        def traceback(total, startindex, path):
            if total > target:
                return
            if total == target:
                self.res.append(path[:])
                return
            for i in range(startindex, len(candidates)):
                traceback(total + candidates[i], i, path + [candidates[i]])

        self.res = []
        # candidates.sort() # 对候选集进行排序
        # total记录当前已经选的元素的和（路径），path记录当前选择的元素，选择列表是所有candidates中的元素
        traceback(0, 0, [])
        return self.res


    def combinationSum2(self, candidates, target):  # 组合总和，候选集中，所有和为target的组合（候选集中的数字一次组合只能用一次）
        def traceback(total, startindex, path):
            if total > target:
                return
            if total == target:
                self.res.append(path[:])
                return
            for i in range(startindex, len(candidates)):  # 防止重合
                # 增加去重逻辑
                if i > startindex and candidates[i] == candidates[i - 1]:
                    continue
                # 注意下一个因为不能重复，所以要从i+1开始
                traceback(total + candidates[i], i + 1, path + [candidates[i]])

        self.res = []
        candidates.sort()  # 对候选集进行排序
        # total记录当前已经选的元素的和（路径），path记录当前选择的元素，选择列表是所有candidates中的元素
        traceback(0, 0, [])
        return self.res


    def restoreIpAddresses(self, s):  # 复原IP地址 判断是否有可能是有效的IP地址
        def traceback(index, path):
            # out = ".".join(path)
            if len(path) == 4 and index == len(s):
                self.res.append(".".join(path))
                return
            for i in range(index, len(s)):
                cur = s[index: i + 1]
                if int(cur) > 255:
                    continue
                if int(cur) > 0 and cur[0] == "0":
                    continue
                if int(cur) == 0 and len(cur) > 1:
                    continue
                # 注意，这里需要是i+1，不能是index+1，需要对index做更新
                traceback(i + 1, path + [cur])

        self.res = []
        # index记录剩余字符开始的位置
        traceback(index=0, path=[])
        return self.res













