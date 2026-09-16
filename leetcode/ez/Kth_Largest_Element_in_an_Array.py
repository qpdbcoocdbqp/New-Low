def findKthLargest(nums, k):
    nums.sort(reverse=True)
    return nums[k - 1]


nums = [1, 2, 3, 4, 5, 6]
k = 6

print(findKthLargest(nums, k))
