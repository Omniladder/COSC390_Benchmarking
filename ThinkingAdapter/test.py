

llm_code = """```python
def maxSum(nums, n, banned):
    num_set = set(banned)
    max_sum = sum(nums)
    
    count = 0
    
    for i in range(n, 0, -1):
        if i not in num_set and max_sum - nums[i-1] >= i:
            max_sum += nums[i-1]
            count += 1
    
    return count
```"""


