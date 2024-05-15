

```python
def search():
  print("Hello, world!" )
  return
```





```c++ 
#include <iostream>
int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
```



# 二分搜索

MARKDOWN语法 https://markdown.com.cn/basic-syntax/htmls.html

### 不大于target的数

#### 涉及题目：二维数组的搜索

```python
def search_left_bound(self, arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right if right >= 0 else -1
```

### 左边界&右边界

```python
def right_loc(self, nums, target):
    # 右边界
    left, right = 0, len(nums) - 1
    while left <= right:  # 闭区间
        mid = left + (right - left) // 2
        if nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    if left < 0 or nums[right] != target:
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
    if left > len(nums) or nums[left] != target:
        return -1
    return left

```



### head11

### head12

## head2





