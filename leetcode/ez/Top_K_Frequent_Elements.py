
from collections import Counter

def topKFrequent(nums, k):
    count = Counter(nums)
    return [num for num, freq in count.most_common(k)]


nums = [1, 1, 1, 2, 2, 3]
k = 2

print(topKFrequent(nums, k))

import pyarrow as pa

df = pa.table(pa.array(nums).value_counts()).sort_by([("counts", "descending")])
print(df["values"][:k].to_pylist())
