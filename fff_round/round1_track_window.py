class Window:
    # 209.长度最小的子数组**
    # 时间复杂度：O(n)，空间复杂度：O(1)
    def minAarrayLen(self, target, num):
        min_len = float('inf')
        total_sum = 0
        left, right = 0, 0
        while right < len(num):
            cur = num[right]
            total_sum += cur
            right += 1
            while total_sum >= target:
                cur1 = num[left]
                min_len = min(min_len, right - left)
                left += 1
                total_sum -= cur1
        if min_len == float('inf'):
            return 0
        else:
            return min_len
        
    # 862.和至少为 K 的最短子数组***
    # 时间复杂度：O(n)，空间复杂度：O(n)
    def shortestArray(self, nums, k):
        presum = [0] * (len(nums) + 1)
        q = []
        min_len = float('inf')
        for i in range(nums):
            presum[i + 1] = presum[i] + nums[i]
        for i, cursum in enumerate(presum):
            while q and presum[q[-1]] > cursum:
                q.pop()
            while q and cursum - presum[q[0]] >= k:
                min_len = min(min_len, i - q[0])
                q.pop(0)
            q.append(i)
        if min_len == float('inf'):
            return -1
        else:
            return min_len

    # 239. 滑动窗口最大值
    # 时间复杂度：O(n)，空间复杂度：O(n)
    def maxSumSlide(self, nums, k):
        res = []
        window = []
        for i in range(len(nums)):
            if (i >= k) and (i - window[0] >= k):
                window.pop(0)
            while window and nums[window[-1]] < nums[i]:
                window.pop()
            window.append(i)
            if i >= k - 1:
                res.append(nums[window[0]])
        return res
    
    # 3.无重复字符的最长子串长度**
    # 时间复杂度：O(n)，空间复杂度：O(n)
    def maxsubstring(self, s):
        left, right = 0, 0
        res = 0
        max_len = 0
        window = {}
        while right < len(s):
            cur = s[right]
            window[cur] = window.get(cur, 0) + 1
            right += 1
            while window[cur] > 1:
                cur0 = s[left]
                window[cur0] -= 1
                left += 1
            max_len = max(max_len, right - left)
        return max_len

    # 76.最小覆盖子串***
    # 时间复杂度：O(n + m)，空间复杂度：O(n + m)
    def mincoversubstring(self, s, t):
        left, right = 0, 0
        window, need = {}, {}
        min_len = float('inf')
        start = 0
        valid = 0
        for i in t:
            need[i] = need.get(i, 0) + 1
        while right < len(s):
            cur = s[right]
            right += 1
            window[cur] = window.get(cur, 0) + 1
            if window[cur] == need.get(cur, 0):
                valid += 1
            while valid == len(need.keys()):
                if right - left < min_len:
                    min_len = right - left
                    start = left
                d = s[left]
                left += 1
                if need.get(d, 0) > 0:
                    if window[d] == need[d]:
                        valid -= 1
                    window[d] -= 1
        return s[start:start + min_len] if min_len != float('inf') else ""

    # 1004.最大连续1的个数 III**
    # 时间复杂度：O(n + m)，空间复杂度：O(1)
    def longestOnes(self, nums, k):
        left, right = 0, 0
        window = {}
        max_len = 0
        num_zeros = 0
        while right < len(nums):
            cur = nums[right]
            if cur == 0:
                num_zeros += 1
            right += 1
            while num_zeros > k:
                d = nums[left]
                if d == 0:
                    num_zeros -= 1
                left += 1
            max_len = max(right - left, max_len)
        return max_len
    
# =================================================================================
#include <deque>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>

using namespace std;

class Window {
public:
    // 209.长度最小的子数组
    int minArrayLen(int target, vector<int>& nums) {
        int min_len = numeric_limits<int>::max();
        int total_sum = 0;
        int left = 0, right = 0;
        while (right < nums.size()) {
            total_sum += nums[right++];
            while (total_sum >= target) {
                min_len = min(min_len, right - left);
                total_sum -= nums[left++];
            }
        }
        return min_len == numeric_limits<int>::max() ? 0 : min_len;
    }

    // 862.和至少为 K 的最短子数组
    int shortestArray(vector<int>& nums, int k) {
        vector<int> presum(nums.size() + 1, 0);
        deque<int> q;
        int min_len = numeric_limits<int>::max();
        for (int i = 0; i < nums.size(); ++i) {
            presum[i + 1] = presum[i] + nums[i];
        }
        for (int i = 0; i < presum.size(); ++i) {
            while (!q.empty() && presum[q.back()] > presum[i]) {
                q.pop_back();
            }
            while (!q.empty() && presum[i] - presum[q.front()] >= k) {
                min_len = min(min_len, i - q.front());
                q.pop_front();
            }
            q.push_back(i);
        }
        return min_len == numeric_limits<int>::max() ? -1 : min_len;
    }

    // 239. 滑动窗口最大值
    vector<int> maxSumSlide(vector<int>& nums, int k) {
        vector<int> res;
        deque<int> window;
        for (int i = 0; i < nums.size(); ++i) {
            if (i >= k && i - window.front() >= k) {
                window.pop_front();
            }
            while (!window.empty() && nums[window.back()] < nums[i]) {
                window.pop_back();
            }
            window.push_back(i);
            if (i >= k - 1) {
                res.push_back(nums[window.front()]);
            }
        }
        return res;
    }

    // 3.无重复字符的最长子串长度
    int maxSubstring(string s) {
        int left = 0, right = 0;
        int max_len = 0;
        unordered_map<char, int> window;
        while (right < s.size()) {
            char cur = s[right];
            ++window[cur];
            ++right;
            while (window[cur] > 1) {
                --window[s[left++]];
            }
            max_len = max(max_len, right - left);
        }
        return max_len;
    }

    // 76.最小覆盖子串
    string minCoverSubstring(string s, string t) {
        int left = 0, right = 0;
        unordered_map<char, int> window, need;
        int min_len = numeric_limits<int>::max();
        int start = 0;
        int valid = 0;
        for (char i : t) {
            ++need[i];
        }
        while (right < s.size()) {
            char cur = s[right];
            ++right;
            ++window[cur];
            if (window[cur] == need[cur]) {
                ++valid;
            }
            while (valid == need.size()) {
                if (right - left < min_len) {
                    min_len = right - left;
                    start = left;
                }
                char d = s[left];
                ++left;
                if (need.count(d)) {
                    if (window[d] == need[d]) {
                        --valid;
                    }
                    --window[d];
                }
            }
        }
        return min_len == numeric_limits<int>::max() ? "" : s.substr(start, min_len);
    }

    // 1004.最大连续1的个数 III
    int longestOnes(vector<int>& nums, int k) {
        int left = 0, right = 0;
        int max_len = 0;
        int num_zeros = 0;
        while (right < nums.size()) {
            if (nums[right] == 0) {
                ++num_zeros;
            }
            ++right;
            while (num_zeros > k) {
                if (nums[left++] == 0) {
                    --num_zeros;
                }
            }
            max_len = max

# ==========================================================
# 46. 全排列
# 47. 全排列 II
# 37. 解数独
# 22. 括号生成
# 78. 子集
# 39. 组合总和
# 40. 组合总和 II
# 93. 复原 IP 地址
# 79. 单词搜索
# 679. 24 点游戏
class TRACK:
    # 46.全排列
    def permute(self, nums):
        def track(path, num):
            if len(path) == self.n:
                self.res.append(path[:])
            for i in range(len(num)):
                track(path + [num[i]], num[:i] + num[i+1:])
        self.n = len(nums)
        self.res = []
        track([], nums)
        return self.res
        
    # 47.全排列 II
    def permute2(self, nums):
        def track(path, num):
            if len(path) == self.n:
                self.res.append(path[:])
            for i in range(len(num)):
                if i > 0 and num[i] == num[i - 1]:
                    continue
                track(path + [num[i]], num[:i] + num[i+1:])
        self.n = len(nums)
        self.res = []
        nums.sort()
        track([], nums)
        return self.res
        
    # 78.子集
    def subset(self, nums):
        def track(path, start_index):
            self.res.append(path[:])
            for i in range(start_index, self.n):
                track(path + [nums[i]], i + 1)
                
        self.n = len(nums)
        self.res = []
        track([], 0)
        return self.res

        
    # 39.组合总和
    def combination(self, nums, target):
        def track(total, path, start_index):
            if total > target:
                return
            if total == target:
                self.res.append(path[:])
            for i in range(start_index, self.n):
                track(total + nums[i], path + [nums[i]], i)
        self.n = len(nums)
        self.path = []
        self.res = []
        track(0, [], 0)
        return self.res
        
    # 40.组合总和2
    def combination2(self, nums, target):
        def track(total, path, start_index):
            if total > target:
                return
            if total == target:
                self.res.append(path[:])
            for i in range(start_index, self.n):
                # 这里需要从start_index开始
                if i > start_index and nums[i] == nums[i - 1]:
                    continue
                track(total + nums[i], path + [nums[i]], i + 1)
        self.n = len(nums)
        nums.sort()
        self.path = []
        self.res = []
        track(0, [], 0)
        return self.res
    
    # 22.括号生成**
    def generageParents(self, n):
        def track(left, right, path):
            if len(path) == 2 * n:
                self.res.append("".join(path))
            if left < n:
                track(left + 1, right, path + ["("])
            if right < left:
                track(left, right + 1, path + [")"])
        self.res = []
        track(0, 0, [])
        return self.res
    
    # 93.复原 IP 地址**
    def restoreIpAddresses(self, s):
        def track(start_index, path):
            if len(path) == 4 and start_index == self.n:
                self.res.append('.'.join(path))
            for i in range(start_index, self.n):
                cur = s[start_index: i + 1]
                if (int(cur) > 255) or (int(cur) == 0 and len(cur) > 1) or (int(cur) > 0 and cur[0] == '0'):
                    break
                track(i + 1, path + [cur])
        self.res = []
        self.n = len(s)
        track(0, [])
        return self.res
    
    # 79.单词搜索**       
    def existWordPath(self, board, word):
        def track(i, j, index): # 从board[i][j]开始搜索word[index]及之后的单词
            if board[i][j] == word[index]:
                if index == len(word) - 1:
                    return True
                temp = board[i][j]
                board[i][j] = "-"
                for r, c in self.direct:
                    new_row, new_col = i + r, j + c
                    if (0 <= new_row <= self.row - 1) and (0 <= new_col <= self.col - 1):
                        if track(new_row, new_col, index + 1): # 这里必须加上！！！！
                            return True
                board[i][j] = temp 
            else:
                return False
        
        self.direct = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        self.row, self.col = len(board), len(board[0])
        for i in range(len(board)):
            for j in range(len(board[0])):
                if track(i, j, 0):
                    return True
        return False
    
    # 37.解数独***
    def sudoku(self, board):
        def valid(board, i, j, k):
            for m in range(9):
                if board[m][j] == str(k) or board[i][m] == str(k):
                    return False
            p, q = i // 3 * 3, j // 3 * 3
            for m in range(p, p + 3):
                for n in range(q, q + 3):
                    if board[m][n] == str(k):
                        return False
            return True
        def track(board):
            for i in range(len(board)):
                for j in range(len(board[0])):
                    if board[i][j] != '.':
                        continue
                    for k in range(1, 10):
                        if valid(board, i, j, k):
                            board[i][j] = str(k)
                            if track(board):
                                return True
                            board[i][j] = "."
                    return False
            return True
        track(board)
# =================================================
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class TRACK {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        permute(nums, 0, res);
        return res;
    }

    void permute(vector<int>& nums, int start, vector<vector<int>>& res) {
        if (start == nums.size()) {
            res.push_back(nums);
            return;
        }
        for (int i = start; i < nums.size(); i++) {
            swap(nums[start], nums[i]);
            permute(nums, start + 1, res);
            swap(nums[start], nums[i]);
        }
    }

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        permute(nums, 0, res);
        return res;
    }

    void permuteUnique(vector<int>& nums, int start, vector<vector<int>>& res) {
        if (start == nums.size()) {
            res.push_back(nums);
            return;
        }
        for (int i = start; i < nums.size(); i++) {
            if (i != start && nums[i] == nums[start]) continue;
            swap(nums[start], nums[i]);
            permute(nums, start + 1, res);
            swap(nums[start], nums[i]);
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> out;
        subsets(nums, 0, out, res);
        return res;
    }

    void subsets(vector<int>& nums, int start, vector<int>& out, vector<vector<int>>& res) {
        res.push_back(out);
        for (int i = start; i < nums.size(); i++) {
            out.push_back(nums[i]);
            subsets(nums, i + 1, out, res);
            out.pop_back();
        }
    }

    vector<vector<int>> combination(vector<int>& nums, int target) {
        vector<vector<int>> res;
        vector<int> path;
        backtrack(nums, target, 0, path, res);
        return res;
    }

    void backtrack(vector<int>& nums, int target, int start, vector<int>& path, vector<vector<int>>& res) {
        if (target < 0) return;
        if (target == 0) {
            res.push_back(path);
            return;
        }
        for (int i = start; i < nums.size(); i++) {
            path.push_back(nums[i]);
            backtrack(nums, target - nums[i], i, path, res);
            path.pop_back();
        }
    }

    vector<vector<int>> combination2(vector<int>& nums, int target) {
        vector<vector<int>> res;
        vector<int> path;
        sort(nums.begin(), nums.end());
        backtrack2(nums, target, 0, path, res);
        return res;
    }

    void backtrack2(vector<int>& nums, int target, int start, vector<int>& path, vector<vector<int>>& res) {
        if (target < 0) return;
        if (target == 0) {
            res.push_back(path);
            return;
        }
        for (int i = start; i < nums.size(); i++) {
            if (i > start && nums[i] == nums[i - 1]) continue;
            path.push_back(nums[i]);
            backtrack2(nums, target - nums[i], i + 1, path, res);
            path.pop_back();
        }
    }

    vector<string> generateParenthesis(int n) {
        vector<string> res;
        string path;
        backtrack3(n, 0, 0, path, res);
        return res;
    }

    void backtrack3(int n, int left, int right, string& path, vector<string>& res) {
        if (path.size() == 2 * n) {
            res.push_back(path);
            return;
        }
        if (left < n) {
            path.push_back('(');
            backtrack3(n, left + 1, right, path, res);
            path.pop_back();
        }
        if (right < left) {
            path.push_back(')');
            backtrack3(n, left, right + 1, path, res);
            path.pop_back();
        }
    }

    vector<string> restoreIpAddresses(string s) {
        vector<string> res;
        vector<string> path;
        backtrack4(s, 0, path, res);
        return res;
    }

    void backtrack4(string& s, int start, vector<string>& path, vector<string>& res) {
        if (path.size() == 4 && start == s.size()) {
            res.push_back(path[0] + '.' + path[1] + '.' + path[2] + '.' + path[3]);
            return;
        }
        for (int i = start; i < s.size(); i++) {
            string cur = s.substr(start, i - start + 1);
            int num = stoi(cur);
            if (num > 255 || (num == 0 && cur.size() > 1) || (num > 0 && cur[0] == '0')) break;
            path.push_back(cur);
            backtrack4(s, i + 1, path, res);
            path.pop_back();
        }
    }

    bool existWordPath(vector<vector<char>>& board, string word) {
        int row = board.size(), col = board[0].size();
        vector<pair<int, int>> direct = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (track(board, word, direct, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool track(vector<vector<char>>& board, string& word, vector<pair<int, int>>& direct, int i, int j, int index) {
        if (board[i][j] == word[index]) {
            if (index == word.size() - 1) {
                return true;
            }
            char temp = board[i][j];
            board[i][j] = '-';
            for (auto& d : direct) {
                int new_row = i + d.first, new_col = j + d.second;
                if (new_row >= 0 && new_row < board.size() && new_col >= 0 && new_col < board[0].size()) {
                    if (track(board, word, direct, new_row, new_col, index + 1)) {
                        return true;
                    }
                }
            }
            board[i][j] = temp;
        }
        return false;
    }

    void sudoku(vector<vector<char>>& board) {
        backtrack(board);
    }

    bool backtrack(vector<vector<char>>& board) {
        int row = board.size(), col = board[0].size();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (board[i][j] != '.') {
                    continue;
                }
                for (char k = '1'; k <= '9'; k++) {
                    if (valid(board, i, j, k)) {
                        board[i][j] = k;
                        if (backtrack(board)) {
                            return true;
                        }
                        board[i][j] = '.';
                    }
                }
                return false;
            }
        }
        return true;
    }

    bool valid(vector<vector<char>>& board, int i, int j, char k) {
        for (int m = 0; m < 9; m++) {
            if (board[m][j] == k || board[i][m] == k) {
                return false;
            }
        }
        int p = i / 3 * 3, q = j / 3 * 3;
        for (int m = p; m < p + 3; m++) {
            for (int n = q; n < q + 3; n++) {
                if (board[m][n] == k) {
                    return false;
                }
            }
        }
        return true;
    }
};
