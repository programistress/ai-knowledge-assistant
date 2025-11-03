# Problem-Solving Patterns

## Introduction

Problem-solving patterns are reusable strategies and techniques that can be applied to solve various algorithmic problems. Recognizing these patterns helps you quickly identify the right approach for a problem.

## Pattern 1: Sliding Window

### Concept

The sliding window pattern is used to track a subset of data (a "window") in an array or string. The window either grows or shrinks based on certain conditions.

### When to Use
- Finding subarrays or substrings that satisfy certain conditions
- Problems involving consecutive elements
- Optimization problems (min/max) with constraints

### Types of Sliding Windows

#### Fixed Size Window

The window size remains constant.

**Problem: Maximum Sum Subarray of Size K**
```python
def max_sum_subarray(arr, k):
    """
    Find maximum sum of any subarray of size k
    """
    if len(arr) < k:
        return None
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        # Remove leftmost element, add rightmost element
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Example: arr = [1, 4, 2, 10, 23, 3, 1, 0, 20], k = 4
# Windows: [1,4,2,10]=17, [4,2,10,23]=39, [2,10,23,3]=38, ...
# Answer: 39

# Time: O(n), Space: O(1)
```

**Problem: Average of Subarrays of Size K**
```python
def find_averages(arr, k):
    result = []
    window_sum = 0
    window_start = 0
    
    for window_end in range(len(arr)):
        window_sum += arr[window_end]
        
        # When window reaches size k
        if window_end >= k - 1:
            result.append(window_sum / k)
            # Slide window
            window_sum -= arr[window_start]
            window_start += 1
    
    return result
```

#### Variable Size Window

The window size changes based on conditions.

**Problem: Smallest Subarray with Sum >= S**
```python
def min_subarray_len(target, nums):
    """
    Find minimum length subarray with sum >= target
    """
    min_length = float('inf')
    window_sum = 0
    window_start = 0
    
    for window_end in range(len(nums)):
        window_sum += nums[window_end]
        
        # Shrink window while condition is met
        while window_sum >= target:
            min_length = min(min_length, window_end - window_start + 1)
            window_sum -= nums[window_start]
            window_start += 1
    
    return min_length if min_length != float('inf') else 0

# Example: nums = [2,3,1,2,4,3], target = 7
# Window [4,3] has sum 7, length 2
# Time: O(n), Space: O(1)
```

**Problem: Longest Substring Without Repeating Characters**
```python
def length_of_longest_substring(s):
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # If duplicate found, shrink window from left
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Example: s = "abcabcbb"
# Longest substring: "abc", length = 3
# Time: O(n), Space: O(min(n, alphabet_size))
```

**Problem: Longest Substring with At Most K Distinct Characters**
```python
def longest_substring_k_distinct(s, k):
    char_count = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Add character to window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink window if more than k distinct characters
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Example: s = "araaci", k = 2
# Longest: "araa", length = 4
```

## Pattern 2: Two Pointers

### Concept

Uses two pointers to traverse the data structure, often from different ends or at different speeds.

### When to Use
- Searching pairs in a sorted array
- Comparing elements
- Problems involving palindromes
- Cycle detection

### Types of Two Pointers

#### Opposite Direction (Converging Pointers)

Pointers start at both ends and move toward each other.

**Problem: Two Sum (Sorted Array)**
```python
def two_sum_sorted(arr, target):
    """
    Find two numbers that add up to target in sorted array
    """
    left = 0
    right = len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return [-1, -1]

# Time: O(n), Space: O(1)
```

**Problem: Valid Palindrome**
```python
def is_palindrome(s):
    """
    Check if string is a palindrome (ignore non-alphanumeric)
    """
    left = 0
    right = len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

# Example: "A man, a plan, a canal: Panama" → True
```

**Problem: Container With Most Water**
```python
def max_area(height):
    """
    Find two lines that together with x-axis forms container with most water
    """
    left = 0
    right = len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate area
        width = right - left
        current_height = min(height[left], height[right])
        current_water = width * current_height
        max_water = max(max_water, current_water)
        
        # Move pointer at shorter line
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# Time: O(n), Space: O(1)
```

#### Same Direction (Fast and Slow Pointers)

Both pointers move in same direction but at different speeds.

**Problem: Remove Duplicates from Sorted Array**
```python
def remove_duplicates(nums):
    """
    Remove duplicates in-place, return new length
    """
    if not nums:
        return 0
    
    slow = 0  # Position for next unique element
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1

# Example: [1,1,2,2,3,4,4] → [1,2,3,4,...], length = 4
# Time: O(n), Space: O(1)
```

**Problem: Move Zeros to End**
```python
def move_zeros(nums):
    """
    Move all zeros to end while maintaining relative order
    """
    slow = 0  # Position for next non-zero
    
    # Move all non-zeros forward
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
    
    # Fill remaining with zeros
    for i in range(slow, len(nums)):
        nums[i] = 0

# Example: [0,1,0,3,12] → [1,3,12,0,0]
```

**Problem: Linked List Cycle Detection (Floyd's Algorithm)**
```python
def has_cycle(head):
    """
    Detect cycle in linked list using fast and slow pointers
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True

# If there's a cycle, fast will eventually meet slow
# Time: O(n), Space: O(1)
```

**Problem: Find Middle of Linked List**
```python
def find_middle(head):
    """
    Find middle node of linked list
    """
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # Slow is at middle when fast reaches end
```

## Pattern 3: Divide and Conquer

### Concept

Breaks problem into smaller subproblems, solves them recursively, and combines results.

### Steps
1. **Divide**: Break problem into smaller instances
2. **Conquer**: Solve subproblems recursively
3. **Combine**: Merge solutions of subproblems

### When to Use
- Sorting and searching
- Tree-based problems
- Problems that can be broken into independent subproblems

**Problem: Merge Sort**
```python
def merge_sort(arr):
    # Base case
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (Combine)
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Time: O(n log n), Space: O(n)
```

**Problem: Quick Sort**
```python
def quick_sort(arr, low, high):
    if low < high:
        # Divide: partition array and get pivot position
        pivot_index = partition(arr, low, high)
        
        # Conquer: recursively sort left and right partitions
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)
    
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Time: O(n log n) average, O(n²) worst, Space: O(log n)
```

**Problem: Maximum Subarray (Divide and Conquer)**
```python
def max_subarray(arr, low, high):
    # Base case
    if low == high:
        return arr[low]
    
    # Divide
    mid = (low + high) // 2
    
    # Conquer
    left_max = max_subarray(arr, low, mid)
    right_max = max_subarray(arr, mid + 1, high)
    cross_max = max_crossing_sum(arr, low, mid, high)
    
    # Combine
    return max(left_max, right_max, cross_max)

def max_crossing_sum(arr, low, mid, high):
    # Maximum sum on left side
    left_sum = float('-inf')
    current_sum = 0
    for i in range(mid, low - 1, -1):
        current_sum += arr[i]
        left_sum = max(left_sum, current_sum)
    
    # Maximum sum on right side
    right_sum = float('-inf')
    current_sum = 0
    for i in range(mid + 1, high + 1):
        current_sum += arr[i]
        right_sum = max(right_sum, current_sum)
    
    return left_sum + right_sum

# Time: O(n log n), Space: O(log n)
```

**Problem: Count Inversions**
```python
def count_inversions(arr):
    """
    Count pairs (i,j) where i < j but arr[i] > arr[j]
    """
    if len(arr) <= 1:
        return arr, 0
    
    # Divide
    mid = len(arr) // 2
    left, left_inv = count_inversions(arr[:mid])
    right, right_inv = count_inversions(arr[mid:])
    
    # Conquer (merge and count split inversions)
    merged, split_inv = merge_and_count(left, right)
    
    total_inv = left_inv + right_inv + split_inv
    return merged, total_inv

def merge_and_count(left, right):
    result = []
    inversions = 0
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            # All remaining elements in left are inversions with right[j]
            inversions += len(left) - i
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result, inversions
```

## Pattern 4: Greedy Algorithms

### Concept

Makes the locally optimal choice at each step, hoping to find a global optimum.

### When to Use
- Optimization problems
- When local optimum leads to global optimum
- Activity selection, scheduling problems
- Minimum spanning tree

### Characteristics
- Makes choice that looks best at the moment
- Never reconsiders earlier choices
- Not always optimal, but often efficient

**Problem: Activity Selection**
```python
def max_activities(start, finish):
    """
    Select maximum number of non-overlapping activities
    """
    # Sort by finish time
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    
    selected = [activities[0]]
    last_finish = activities[0][1]
    
    for s, f in activities[1:]:
        # If activity starts after last selected finishes
        if s >= last_finish:
            selected.append((s, f))
            last_finish = f
    
    return len(selected)

# Greedy choice: always pick activity that finishes earliest
# Time: O(n log n), Space: O(n)
```

**Problem: Coin Change (Greedy - works for certain coin systems)**
```python
def coin_change_greedy(coins, amount):
    """
    Make change using minimum number of coins (greedy approach)
    Only works for certain coin systems (like US coins)
    """
    coins.sort(reverse=True)  # Largest first
    count = 0
    
    for coin in coins:
        if amount == 0:
            break
        # Use as many of this coin as possible
        count += amount // coin
        amount %= coin
    
    return count if amount == 0 else -1

# Example: coins = [25, 10, 5, 1], amount = 41
# Use: 1×25, 1×10, 1×5, 1×1 = 4 coins
```

**Problem: Jump Game**
```python
def can_jump(nums):
    """
    Can you reach the last index from first index?
    Each element represents maximum jump length at that position.
    """
    max_reach = 0
    
    for i in range(len(nums)):
        # If current position is unreachable
        if i > max_reach:
            return False
        
        # Update maximum reachable position
        max_reach = max(max_reach, i + nums[i])
        
        # If we can reach the end
        if max_reach >= len(nums) - 1:
            return True
    
    return True

# Greedy: always try to reach as far as possible
# Time: O(n), Space: O(1)
```

**Problem: Minimum Number of Platforms**
```python
def min_platforms(arrivals, departures):
    """
    Find minimum number of platforms needed at railway station
    """
    arrivals.sort()
    departures.sort()
    
    platforms_needed = 1
    max_platforms = 1
    i = j = 1
    
    while i < len(arrivals) and j < len(departures):
        # If train arrives before previous one departs
        if arrivals[i] <= departures[j]:
            platforms_needed += 1
            i += 1
        else:
            platforms_needed -= 1
            j += 1
        
        max_platforms = max(max_platforms, platforms_needed)
    
    return max_platforms

# Time: O(n log n), Space: O(1)
```

**Problem: Fractional Knapsack**
```python
def fractional_knapsack(weights, values, capacity):
    """
    Fill knapsack with items to maximize value (can take fractions)
    """
    # Calculate value per weight
    items = [(values[i] / weights[i], weights[i], values[i]) 
             for i in range(len(weights))]
    
    # Sort by value per weight (descending)
    items.sort(reverse=True)
    
    total_value = 0
    
    for value_per_weight, weight, value in items:
        if capacity == 0:
            break
        
        # Take as much as possible of this item
        amount = min(weight, capacity)
        total_value += amount * value_per_weight
        capacity -= amount
    
    return total_value

# Greedy: always take item with highest value/weight ratio
# Time: O(n log n), Space: O(n)
```

## Pattern 5: Fast and Slow Pointers (Tortoise and Hare)

### Concept

Two pointers moving at different speeds to detect cycles or find middle elements.

### When to Use
- Cycle detection in linked lists
- Finding middle of linked list
- Detecting patterns in sequences

**Problem: Find Duplicate Number**
```python
def find_duplicate(nums):
    """
    Find duplicate in array of n+1 integers where each integer is between 1 and n
    """
    # Phase 1: Find intersection point in the cycle
    slow = fast = nums[0]
    
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    # Phase 2: Find entrance to cycle (duplicate)
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow

# Time: O(n), Space: O(1)
```

**Problem: Happy Number**
```python
def is_happy(n):
    """
    A happy number is a number where repeatedly replacing it by sum of 
    squares of its digits eventually leads to 1
    """
    def get_next(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    slow = n
    fast = get_next(n)
    
    while fast != 1 and slow != fast:
        slow = get_next(slow)
        fast = get_next(get_next(fast))
    
    return fast == 1

# If there's a cycle that doesn't include 1, slow and fast will meet
```

## Pattern 6: Merge Intervals

### Concept

Problems involving overlapping intervals that need to be merged, inserted, or analyzed.

### When to Use
- Interval scheduling
- Finding overlapping ranges
- Time-based problems

**Problem: Merge Overlapping Intervals**
```python
def merge_intervals(intervals):
    """
    Merge all overlapping intervals
    """
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        
        # If intervals overlap
        if current[0] <= last_merged[1]:
            # Merge by updating end time
            last_merged[1] = max(last_merged[1], current[1])
        else:
            # No overlap, add as new interval
            merged.append(current)
    
    return merged

# Example: [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]
# Time: O(n log n), Space: O(n)
```

**Problem: Insert Interval**
```python
def insert_interval(intervals, new_interval):
    """
    Insert a new interval and merge if necessary
    """
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals that come before new interval
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge all overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result

# Time: O(n), Space: O(n)
```

**Problem: Meeting Rooms**
```python
def can_attend_all_meetings(intervals):
    """
    Check if person can attend all meetings (no overlaps)
    """
    if not intervals:
        return True
    
    intervals.sort(key=lambda x: x[0])
    
    for i in range(1, len(intervals)):
        # If meeting starts before previous ends
        if intervals[i][0] < intervals[i-1][1]:
            return False
    
    return True
```

## Summary

### Pattern Selection Guide

| Problem Type | Pattern | Time Complexity |
|--------------|---------|-----------------|
| Consecutive elements, substring/subarray | Sliding Window | O(n) |
| Pairs in sorted array, palindromes | Two Pointers | O(n) |
| Sorting, divide problem | Divide & Conquer | O(n log n) |
| Optimization, scheduling | Greedy | O(n log n) |
| Cycle detection, middle element | Fast & Slow | O(n) |
| Overlapping ranges | Merge Intervals | O(n log n) |

### Key Takeaways

1. **Recognize the pattern** in the problem statement
2. **Choose the right approach** based on constraints
3. **Optimize** by considering edge cases
4. **Practice** to build pattern recognition skills

These patterns are fundamental building blocks for solving complex algorithmic problems efficiently!

