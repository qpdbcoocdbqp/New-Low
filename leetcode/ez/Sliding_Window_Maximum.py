from collections import deque

def maxSlidingWindow(nums, k):
    queue = deque()
    result = []

    for i, num in enumerate(nums):
        # 移除已經離開視窗的索引
        while queue and queue[0] <= i - k:
            queue.popleft()

        # 移除比目前數字小的元素
        while queue and nums[queue[-1]] <= num:
            queue.pop()

        queue.append(i)

        # 視窗形成後，queue 最前面就是最大值
        if i >= k - 1:
            result.append(nums[queue[0]])

    return result

nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3

print(maxSlidingWindow(nums, k))
