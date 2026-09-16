def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        diff = target - x
        if diff in seen:
            return [seen[diff], i]
        seen[x] = i
    return None


nums = [2,7,11,13]
target = 9

print(two_sum(nums, target))