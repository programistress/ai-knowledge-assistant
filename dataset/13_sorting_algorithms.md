# Sorting Algorithms

## Introduction

Sorting is the process of arranging elements in a specific order (ascending or descending). It's one of the most fundamental operations in computer science.

### Sorting Algorithm Properties

1. **Time Complexity**: Best, average, and worst case
2. **Space Complexity**: Extra memory required
3. **Stability**: Preserves relative order of equal elements
4. **In-place**: Sorts within the original array
5. **Adaptive**: Takes advantage of existing order

## Comparison-Based Sorting

### Bubble Sort

Repeatedly swaps adjacent elements if they're in wrong order.

```python
def bubble_sort(arr):
    """
    Bubble Sort - Simple but inefficient
    """
    n = len(arr)
    
    for i in range(n):
        swapped = False
        
        # Last i elements are already sorted
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swaps, array is sorted
        if not swapped:
            break
    
    return arr

# Time Complexity:
# - Best: O(n) when already sorted
# - Average: O(n²)
# - Worst: O(n²)
# Space: O(1)
# Stable: Yes
# Adaptive: Yes (with optimization)
```

### Selection Sort

Finds minimum element and places it at beginning.

```python
def selection_sort(arr):
    """
    Selection Sort - Find minimum and swap
    """
    n = len(arr)
    
    for i in range(n):
        # Find minimum in unsorted portion
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap minimum with first unsorted element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

# Time Complexity: O(n²) for all cases
# Space: O(1)
# Stable: No (can be made stable with modifications)
# Adaptive: No
```

### Insertion Sort

Builds sorted array one element at a time.

```python
def insertion_sort(arr):
    """
    Insertion Sort - Insert each element into sorted portion
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key
    
    return arr

# Time Complexity:
# - Best: O(n) when nearly sorted
# - Average: O(n²)
# - Worst: O(n²)
# Space: O(1)
# Stable: Yes
# Adaptive: Yes
```

### Merge Sort

Divide and conquer algorithm that recursively divides and merges.

```python
def merge_sort(arr):
    """
    Merge Sort - Divide and conquer
    """
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (Merge)
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays"""
    result = []
    i = j = 0
    
    # Merge elements in sorted order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# In-place version (more complex but saves space)
def merge_sort_inplace(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        merge_sort_inplace(arr, left, mid)
        merge_sort_inplace(arr, mid + 1, right)
        merge_inplace(arr, left, mid, right)

def merge_inplace(arr, left, mid, right):
    # Create temp arrays
    L = arr[left:mid + 1]
    R = arr[mid + 1:right + 1]
    
    i = j = 0
    k = left
    
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
    
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1

# Time Complexity: O(n log n) for all cases
# Space: O(n) for arrays, O(log n) for recursion stack
# Stable: Yes
# Adaptive: No
```

### Quick Sort

Divide and conquer using pivot partitioning.

```python
def quick_sort(arr, low, high):
    """
    Quick Sort - Partition around pivot
    """
    if low < high:
        # Partition and get pivot position
        pi = partition(arr, low, high)
        
        # Recursively sort left and right
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)
    
    return arr

def partition(arr, low, high):
    """Lomuto partition scheme"""
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Hoare partition scheme (alternative)
def partition_hoare(arr, low, high):
    """Hoare partition scheme (more efficient)"""
    pivot = arr[low]
    i = low - 1
    j = high + 1
    
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        
        j -= 1
        while arr[j] > pivot:
            j -= 1
        
        if i >= j:
            return j
        
        arr[i], arr[j] = arr[j], arr[i]

# Randomized Quick Sort (better average case)
import random

def quick_sort_randomized(arr, low, high):
    if low < high:
        pi = partition_randomized(arr, low, high)
        quick_sort_randomized(arr, low, pi - 1)
        quick_sort_randomized(arr, pi + 1, high)
    return arr

def partition_randomized(arr, low, high):
    # Choose random pivot
    pivot_idx = random.randint(low, high)
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
    return partition(arr, low, high)

# Time Complexity:
# - Best: O(n log n)
# - Average: O(n log n)
# - Worst: O(n²) when already sorted or all equal
# Space: O(log n) for recursion stack
# Stable: No
# Adaptive: No
```

### Heap Sort

Uses heap data structure to sort.

```python
def heap_sort(arr):
    """
    Heap Sort using max heap
    """
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        # Move current root to end
        arr[0], arr[i] = arr[i], arr[0]
        # Heapify reduced heap
        heapify(arr, i, 0)
    
    return arr

def heapify(arr, n, i):
    """Heapify subtree rooted at index i"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# Time Complexity: O(n log n) for all cases
# Space: O(1)
# Stable: No
# Adaptive: No
```

## Non-Comparison Based Sorting

### Counting Sort

Sorts by counting occurrences of each value.

```python
def counting_sort(arr):
    """
    Counting Sort - For small range of integers
    """
    if not arr:
        return arr
    
    # Find range
    max_val = max(arr)
    min_val = min(arr)
    range_size = max_val - min_val + 1
    
    # Count occurrences
    count = [0] * range_size
    for num in arr:
        count[num - min_val] += 1
    
    # Calculate cumulative count
    for i in range(1, range_size):
        count[i] += count[i - 1]
    
    # Build output array
    output = [0] * len(arr)
    for num in reversed(arr):
        index = count[num - min_val] - 1
        output[index] = num
        count[num - min_val] -= 1
    
    return output

# Time Complexity: O(n + k) where k is range
# Space: O(n + k)
# Stable: Yes
# Good when range of values is not significantly larger than n
```

### Radix Sort

Sorts by processing digits from least significant to most significant.

```python
def radix_sort(arr):
    """
    Radix Sort - Sort by digits
    """
    if not arr:
        return arr
    
    # Find maximum to determine number of digits
    max_val = max(arr)
    
    # Sort by each digit
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_by_digit(arr, exp):
    """Counting sort based on digit at position exp"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    # Count occurrences
    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1
    
    # Calculate cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output array (process from right for stability)
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    
    # Copy to original array
    for i in range(n):
        arr[i] = output[i]

# Time Complexity: O(d × (n + k)) where d is digits, k is base (10)
# Space: O(n + k)
# Stable: Yes
# Good for fixed-length integer keys
```

### Bucket Sort

Distributes elements into buckets, sorts buckets individually.

```python
def bucket_sort(arr):
    """
    Bucket Sort - Distribute to buckets and sort
    """
    if not arr:
        return arr
    
    # Find range
    min_val = min(arr)
    max_val = max(arr)
    
    # Create buckets
    bucket_count = len(arr)
    bucket_range = (max_val - min_val) / bucket_count
    buckets = [[] for _ in range(bucket_count)]
    
    # Distribute elements to buckets
    for num in arr:
        if num == max_val:
            index = bucket_count - 1
        else:
            index = int((num - min_val) / bucket_range)
        buckets[index].append(num)
    
    # Sort each bucket and concatenate
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(sorted(bucket))  # Using Python's Timsort
    
    return sorted_arr

# Time Complexity:
# - Best: O(n + k) when uniformly distributed
# - Average: O(n + k)
# - Worst: O(n²) when all in one bucket
# Space: O(n + k)
# Stable: Yes (if underlying sort is stable)
# Good for uniformly distributed data
```

## Special Sorting Algorithms

### Shell Sort

Generalization of insertion sort with gap sequence.

```python
def shell_sort(arr):
    """
    Shell Sort - Insertion sort with gaps
    """
    n = len(arr)
    gap = n // 2
    
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            
            arr[j] = temp
        
        gap //= 2
    
    return arr

# Time Complexity: Depends on gap sequence
# - O(n log n) to O(n²)
# Space: O(1)
# Stable: No
```

### Tim Sort (Python's Default)

Hybrid of merge sort and insertion sort used in Python's sorted() and list.sort().

```python
def tim_sort(arr):
    """
    Simplified Tim Sort concept
    - Uses insertion sort for small arrays
    - Uses merge sort for larger arrays
    """
    MIN_MERGE = 32
    
    def insertion_sort_range(arr, left, right):
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    n = len(arr)
    
    # Sort small runs with insertion sort
    for start in range(0, n, MIN_MERGE):
        end = min(start + MIN_MERGE - 1, n - 1)
        insertion_sort_range(arr, start, end)
    
    # Merge sorted runs
    size = MIN_MERGE
    while size < n:
        for start in range(0, n, size * 2):
            mid = start + size - 1
            end = min(start + size * 2 - 1, n - 1)
            
            if mid < end:
                merge_inplace(arr, start, mid, end)
        
        size *= 2
    
    return arr

# Time: O(n log n)
# Space: O(n)
# Stable: Yes
# Adaptive: Yes
```

## Comparison of Sorting Algorithms

| Algorithm | Best | Average | Worst | Space | Stable | Notes |
|-----------|------|---------|-------|-------|--------|-------|
| Bubble | O(n) | O(n²) | O(n²) | O(1) | Yes | Simple, slow |
| Selection | O(n²) | O(n²) | O(n²) | O(1) | No | Fewer swaps |
| Insertion | O(n) | O(n²) | O(n²) | O(1) | Yes | Good for small/nearly sorted |
| Merge | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | Predictable, not in-place |
| Quick | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Fast average, randomize for safety |
| Heap | O(n log n) | O(n log n) | O(n log n) | O(1) | No | In-place, predictable |
| Counting | O(n + k) | O(n + k) | O(n + k) | O(k) | Yes | Limited to integers |
| Radix | O(d(n + k)) | O(d(n + k)) | O(d(n + k)) | O(n + k) | Yes | For fixed-length keys |
| Bucket | O(n + k) | O(n + k) | O(n²) | O(n + k) | Yes | Good for uniform data |

## Sorting Problems

### Problem 1: Sort Colors (Dutch National Flag)

```python
def sort_colors(nums):
    """
    Sort array with 0s, 1s, and 2s (3-way partitioning)
    """
    low = mid = 0
    high = len(nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1

# Time: O(n), Space: O(1)
```

### Problem 2: Merge Sorted Arrays

```python
def merge_sorted_arrays(nums1, m, nums2, n):
    """
    Merge nums2 into nums1 (nums1 has size m+n)
    """
    p1 = m - 1
    p2 = n - 1
    p = m + n - 1
    
    # Fill from back to avoid overwriting
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    
    # Add remaining from nums2
    while p2 >= 0:
        nums1[p] = nums2[p2]
        p2 -= 1
        p -= 1

# Time: O(m + n), Space: O(1)
```

### Problem 3: Largest Number

```python
from functools import cmp_to_key

def largest_number(nums):
    """
    Arrange numbers to form largest number
    """
    # Custom comparator
    def compare(x, y):
        if x + y > y + x:
            return -1
        elif x + y < y + x:
            return 1
        else:
            return 0
    
    # Convert to strings and sort
    nums_str = list(map(str, nums))
    nums_str.sort(key=cmp_to_key(compare))
    
    # Handle edge case: all zeros
    result = ''.join(nums_str)
    return '0' if result[0] == '0' else result

# Example: [3, 30, 34, 5, 9] → "9534330"
```

### Problem 4: Meeting Rooms II

```python
def min_meeting_rooms(intervals):
    """
    Find minimum number of meeting rooms needed
    """
    if not intervals:
        return 0
    
    # Separate start and end times
    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])
    
    rooms = 0
    end_ptr = 0
    
    for start in starts:
        if start < ends[end_ptr]:
            rooms += 1
        else:
            end_ptr += 1
    
    return rooms

# Time: O(n log n), Space: O(n)
```

## When to Use Which Sort?

**Small Arrays (n < 10-50):**
- Insertion Sort - Simple, adaptive, low overhead

**Nearly Sorted:**
- Insertion Sort or Tim Sort - Adaptive advantage

**Memory Constrained:**
- Heap Sort or Quick Sort - O(1) or O(log n) space

**Stability Required:**
- Merge Sort or Tim Sort - Stable, predictable

**General Purpose:**
- Quick Sort (randomized) - Fast average case
- Tim Sort - Python's default, adaptive

**Known Range (integers):**
- Counting Sort or Radix Sort - Linear time

**Uniform Distribution:**
- Bucket Sort - Linear time possible

**Linked Lists:**
- Merge Sort - Works well with linked structures

## Summary

**Key Concepts:**
- Comparison-based sorts have Ω(n log n) lower bound
- Non-comparison sorts can be linear but have constraints
- Stability matters when sorting complex objects
- In-place sorts save memory
- Adaptive sorts benefit from partial order

**Practice Tips:**
- Understand time/space tradeoffs
- Know when to use each algorithm
- Master merge and quick sort first
- Consider stability requirements
- Use built-in sorts unless there's a specific reason

