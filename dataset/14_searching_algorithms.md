# Searching Algorithms

## Introduction

Searching is the process of finding a specific element in a data structure. Different searching algorithms have different characteristics and use cases.

## Linear Search

### Basic Linear Search

Sequential search through each element.

```python
def linear_search(arr, target):
    """
    Linear search - Check each element sequentially
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Time: O(n)
# Space: O(1)
# Works on unsorted arrays
```

### Variations

**Find All Occurrences:**
```python
def linear_search_all(arr, target):
    """Find all indices where target appears"""
    indices = []
    for i in range(len(arr)):
        if arr[i] == target:
            indices.append(i)
    return indices
```

**Sentinel Linear Search:**
```python
def sentinel_linear_search(arr, target):
    """Optimized linear search with sentinel"""
    n = len(arr)
    last = arr[n - 1]
    arr[n - 1] = target
    
    i = 0
    while arr[i] != target:
        i += 1
    
    arr[n - 1] = last
    
    if i < n - 1 or arr[n - 1] == target:
        return i
    return -1
```

## Binary Search

### Basic Binary Search

Efficient search on sorted arrays by repeatedly dividing search space in half.

```python
def binary_search(arr, target):
    """
    Binary search - Requires sorted array
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Time: O(log n)
# Space: O(1)
# Requires sorted array
```

**Recursive Version:**
```python
def binary_search_recursive(arr, target, left, right):
    """Recursive binary search"""
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Time: O(log n)
# Space: O(log n) for recursion stack
```

### Binary Search Variations

**Find First Occurrence:**
```python
def find_first_occurrence(arr, target):
    """
    Find first (leftmost) occurrence of target
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Example: [1, 2, 2, 2, 3], target=2 → returns 1
```

**Find Last Occurrence:**
```python
def find_last_occurrence(arr, target):
    """
    Find last (rightmost) occurrence of target
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Example: [1, 2, 2, 2, 3], target=2 → returns 3
```

**Count Occurrences:**
```python
def count_occurrences(arr, target):
    """Count occurrences using first and last"""
    first = find_first_occurrence(arr, target)
    if first == -1:
        return 0
    
    last = find_last_occurrence(arr, target)
    return last - first + 1
```

**Find Insert Position:**
```python
def search_insert_position(arr, target):
    """
    Find index where target should be inserted to maintain order
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return left  # Position to insert

# Example: [1, 3, 5, 6], target=2 → returns 1
```

## Binary Search on Answer

Search for answer in a range where we can verify if a value works.

### Template

```python
def binary_search_on_answer(min_val, max_val):
    """
    Binary search on answer space
    """
    left, right = min_val, max_val
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if is_valid(mid):
            result = mid
            # Depending on problem, search for better answer
            right = mid - 1  # or left = mid + 1
        else:
            left = mid + 1  # or right = mid - 1
    
    return result

def is_valid(value):
    """Check if value satisfies the condition"""
    # Implement problem-specific validation
    pass
```

### Problems

**Problem 1: Square Root**
```python
def my_sqrt(x):
    """
    Find integer square root (floor)
    """
    if x == 0:
        return 0
    
    left, right = 1, x
    result = 0
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if mid <= x // mid:  # mid * mid <= x (avoid overflow)
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Time: O(log n)
```

**Problem 2: Find Peak Element**
```python
def find_peak_element(nums):
    """
    Find a peak element (greater than neighbors)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            # Peak is on left side (including mid)
            right = mid
        else:
            # Peak is on right side
            left = mid + 1
    
    return left

# Time: O(log n)
```

**Problem 3: Capacity To Ship Packages**
```python
def ship_within_days(weights, days):
    """
    Find minimum capacity to ship all packages within given days
    """
    def can_ship(capacity):
        """Check if we can ship with given capacity"""
        days_needed = 1
        current_load = 0
        
        for weight in weights:
            if current_load + weight > capacity:
                days_needed += 1
                current_load = weight
            else:
                current_load += weight
        
        return days_needed <= days
    
    # Binary search on capacity
    left = max(weights)  # Min capacity needed
    right = sum(weights)  # Max capacity needed
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship(mid):
            right = mid  # Try smaller capacity
        else:
            left = mid + 1  # Need larger capacity
    
    return left

# Time: O(n log(sum - max))
```

**Problem 4: Koko Eating Bananas**
```python
def min_eating_speed(piles, h):
    """
    Find minimum eating speed to finish all bananas in h hours
    """
    import math
    
    def can_finish(speed):
        """Check if Koko can finish with given speed"""
        hours = 0
        for pile in piles:
            hours += math.ceil(pile / speed)
        return hours <= h
    
    left, right = 1, max(piles)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

## Search in Rotated/Modified Arrays

### Rotated Sorted Array

```python
def search_rotated_array(nums, target):
    """
    Search in rotated sorted array
    Example: [4, 5, 6, 7, 0, 1, 2], target=0
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Time: O(log n)
```

**Find Minimum in Rotated Sorted Array:**
```python
def find_min_rotated(nums):
    """
    Find minimum element in rotated sorted array
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid
    
    return nums[left]
```

### Search in 2D Matrix

```python
def search_matrix(matrix, target):
    """
    Search in 2D matrix where:
    - Each row is sorted left to right
    - First element of each row > last element of previous row
    """
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        mid_val = matrix[mid // cols][mid % cols]
        
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# Time: O(log(m × n))
```

**Search 2D Matrix II:**
```python
def search_matrix_ii(matrix, target):
    """
    Search in matrix where rows and columns are sorted
    """
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1  # Start from top-right
    
    while row < rows and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            row += 1  # Move down
        else:
            col -= 1  # Move left
    
    return False

# Time: O(m + n)
```

## Ternary Search

Used to find maximum/minimum of unimodal functions.

```python
def ternary_search(left, right, func):
    """
    Ternary search for maximum of unimodal function
    """
    epsilon = 1e-9
    
    while right - left > epsilon:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        if func(mid1) < func(mid2):
            left = mid1
        else:
            right = mid2
    
    return (left + right) / 2

# Time: O(log n)
# Use cases: Finding peak in unimodal function
```

## Exponential Search

Useful for unbounded/infinite arrays.

```python
def exponential_search(arr, target):
    """
    Exponential search - For unbounded arrays
    """
    if arr[0] == target:
        return 0
    
    # Find range for binary search by repeated doubling
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    
    # Binary search in found range
    return binary_search_range(arr, target, i // 2, min(i, len(arr) - 1))

def binary_search_range(arr, target, left, right):
    """Binary search in given range"""
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Time: O(log n)
# Space: O(1)
```

## Interpolation Search

Better than binary search for uniformly distributed sorted arrays.

```python
def interpolation_search(arr, target):
    """
    Interpolation search - Estimates position based on value
    Better for uniformly distributed data
    """
    left, right = 0, len(arr) - 1
    
    while left <= right and target >= arr[left] and target <= arr[right]:
        if left == right:
            if arr[left] == target:
                return left
            return -1
        
        # Estimate position
        pos = left + int(((target - arr[left]) / (arr[right] - arr[left])) * (right - left))
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    
    return -1

# Time: O(log log n) average for uniform data, O(n) worst case
# Space: O(1)
```

## Jump Search

Block jumping then linear search.

```python
import math

def jump_search(arr, target):
    """
    Jump search - Jump by sqrt(n) blocks
    """
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0
    
    # Jump to find block containing target
    while prev < n and arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Linear search in block
    while prev < n and arr[prev] < target:
        prev += 1
    
    if prev < n and arr[prev] == target:
        return prev
    
    return -1

# Time: O(√n)
# Space: O(1)
# Good compromise between linear and binary
```

## Search Problems

### Problem 1: Find in Mountain Array

```python
def find_in_mountain_array(target, mountain_arr):
    """
    Find target in mountain array (increases then decreases)
    """
    def find_peak():
        left, right = 0, mountain_arr.length() - 1
        while left < right:
            mid = left + (right - left) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                left = mid + 1
            else:
                right = mid
        return left
    
    def binary_search_asc(left, right):
        while left <= right:
            mid = left + (right - left) // 2
            val = mountain_arr.get(mid)
            if val == target:
                return mid
            elif val < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    def binary_search_desc(left, right):
        while left <= right:
            mid = left + (right - left) // 2
            val = mountain_arr.get(mid)
            if val == target:
                return mid
            elif val > target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    # Find peak
    peak = find_peak()
    
    # Search ascending part
    result = binary_search_asc(0, peak)
    if result != -1:
        return result
    
    # Search descending part
    return binary_search_desc(peak + 1, mountain_arr.length() - 1)
```

### Problem 2: Find K Closest Elements

```python
def find_closest_elements(arr, k, x):
    """
    Find k closest elements to x
    """
    left, right = 0, len(arr) - k
    
    while left < right:
        mid = left + (right - left) // 2
        
        # Compare distances
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid
    
    return arr[left:left + k]

# Time: O(log(n - k) + k)
```

### Problem 3: Median of Two Sorted Arrays

```python
def find_median_sorted_arrays(nums1, nums2):
    """
    Find median of two sorted arrays in O(log(min(m,n)))
    """
    # Ensure nums1 is smaller
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found correct partition
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1

# Time: O(log(min(m, n)))
```

## Summary

### Algorithm Comparison

| Algorithm | Time (Avg) | Time (Worst) | Space | Requirements |
|-----------|------------|--------------|-------|--------------|
| Linear | O(n) | O(n) | O(1) | None |
| Binary | O(log n) | O(log n) | O(1) | Sorted |
| Jump | O(√n) | O(√n) | O(1) | Sorted |
| Interpolation | O(log log n) | O(n) | O(1) | Sorted, Uniform |
| Exponential | O(log n) | O(log n) | O(1) | Sorted, Unbounded |
| Ternary | O(log₃ n) | O(log₃ n) | O(1) | Unimodal |

### When to Use

**Linear Search:**
- Small arrays
- Unsorted data
- Need all occurrences

**Binary Search:**
- Large sorted arrays
- Need O(log n) performance
- Most common choice for sorted data

**Binary Search on Answer:**
- Optimization problems
- Can verify if value works
- Answer space is monotonic

**Interpolation Search:**
- Uniformly distributed sorted data
- Better than binary for uniform data

**Jump Search:**
- Sorted linked lists (no random access)
- Compromise between linear and binary

**Exponential Search:**
- Unbounded/infinite sorted arrays
- Target likely near beginning

### Key Patterns

1. **Basic Binary Search**: Find exact match
2. **Lower Bound**: First element ≥ target
3. **Upper Bound**: First element > target
4. **Binary Search on Answer**: Search answer space
5. **Modified Arrays**: Handle rotations, 2D matrices
6. **Two Pointers with Search**: Combine searching strategies

### Common Mistakes

- Off-by-one errors in binary search
- Integer overflow in `(left + right) / 2` - use `left + (right - left) / 2`
- Wrong boundary conditions
- Not handling empty arrays
- Incorrect comparison in binary search on answer

### Practice Tips

- Master basic binary search first
- Understand invariants (what remains true after each iteration)
- Draw diagrams for edge cases
- Test with: empty array, single element, two elements
- Practice binary search variations extensively

