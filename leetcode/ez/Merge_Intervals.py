def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    result = []

    for start, end in intervals:
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)

    return result

intervals = [[1, 3], [2, 6], [8, 10], [9, 12]]
print(merge(intervals))
