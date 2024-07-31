
def rotate(self, nums: List[int], k: int) -> None:  # 189. 轮转数组
        """
        Do not return anything, modify nums in-place instead.
        """
        def reverse(nums, left, right):
            while left < right:
                temp = nums[left]
                nums[left] = nums[right]
                nums[right] = temp
                left += 1
                right -= 1
        k = k % len(nums)
        reverse(nums, 0, len(nums) - 1)
        reverse(nums, 0, k - 1)
        reverse(nums, k, len(nums) - 1)
        return nums


def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:  # 498. 对角线遍历
        m = len(mat)
        n = len(mat[0])
        count = m * n
        x, y = 0, 0
        res = []
        for i in range(count):
            print(x, y)
            res.append(mat[x][y])
            if (x + y) % 2 == 0:
                if y == n - 1:
                    x += 1
                elif x == 0:
                    y += 1
                else:
                    x -= 1
                    y += 1
            else:
                if x == m - 1:
                    y += 1
                elif y == 0:
                    x += 1
                else:
                    x += 1
                    y -= 1
        return res

def rotate(self, matrix):   # 48. 旋转图像
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n // 2):
            for j in range( (n + 1) // 2):
                temp = matrix[i][j]
                matrix[i][j] = matrix[n - j - 1][i]
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1]
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1]
                matrix[j][n - i - 1] = temp
        return matrix


def spiralOrder(self, matrix):  # 54. 螺旋矩阵
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        up, down, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
        res = []
        while True:
            for i in range(left, right + 1):
                res.append(matrix[up][i])
            up += 1
            if up > down:
                break
            for i in range(up, down + 1):
                res.append(matrix[i][right])
            right -= 1
            if left > right:
                break
            for i in range(right, left - 1, -1):
                res.append(matrix[down][i])
            down -= 1
            if up > down:
                break
            for i in range(down, up - 1, -1):
                res.append(matrix[i][left])
            left += 1
            if left > right:
                break
        return res

def generateMatrix(self, n):  # 59. 螺旋矩阵 II
        """
        :type n: int
        :rtype: List[List[int]]
        """
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        up, down, left, right = 0, n - 1, 0, n - 1
        index = 1
        while True:
            for i in range(left, right + 1):
                matrix[up][i] = index
                index += 1
            up += 1
            if up > down:
                break
            for i in range(up, down + 1):
                matrix[i][right] = index
                index += 1
            right -= 1
            if left > right:
                break
            for i in range(right, left - 1, -1):
                matrix[down][i] = index
                index += 1
            down -= 1
            if up > down:
                break
            for i in range(down, up - 1, -1):
                matrix[i][left] = index
                index += 1
            left += 1
            if left > right:
                break
        return matrix