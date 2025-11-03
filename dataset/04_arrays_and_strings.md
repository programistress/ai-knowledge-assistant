# Arrays and Strings

## Arrays

### Introduction

An array is a contiguous block of memory that stores elements of the same type. Arrays provide O(1) random access but may have expensive insertion/deletion operations.

### Array Basics

**Characteristics:**
- Fixed or dynamic size
- Contiguous memory allocation
- O(1) access time by index
- Cache-friendly due to locality

**Time Complexities:**
- Access: O(1)
- Search: O(n) unsorted, O(log n) sorted (binary search)
- Insertion: O(n) (worst case, need to shift elements)
- Deletion: O(n) (worst case, need to shift elements)
- Append: O(1) amortized for dynamic arrays

### 1D Arrays

**Basic Operations:**

```python
# Creating arrays
arr = [1, 2, 3, 4, 5]
arr = [0] * 10  # Array of 10 zeros
arr = list(range(1, 11))  # [1, 2, 3, ..., 10]

# Accessing elements
first = arr[0]  # O(1)
last = arr[-1]  # O(1)

# Modifying elements
arr[0] = 10  # O(1)

# Inserting elements
arr.append(6)  # O(1) amortized
arr.insert(0, 0)  # O(n) - shifts all elements

# Deleting elements
arr.pop()  # O(1) - remove last
arr.pop(0)  # O(n) - remove first, shifts all
arr.remove(3)  # O(n) - find and remove

# Slicing
subarray = arr[2:5]  # O(k) where k is slice length
reversed_arr = arr[::-1]  # O(n)
```

### Common Array Problems

**Problem 1: Two Sum**
```python
def two_sum(nums, target):
    """
    Find indices of two numbers that add up to target
    """
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []

# Time: O(n), Space: O(n)
```

**Problem 2: Maximum Subarray (Kadane's Algorithm)**
```python
def max_subarray(nums):
    """
    Find contiguous subarray with largest sum
    """
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        # Either extend existing subarray or start new one
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Example: [-2,1,-3,4,-1,2,1,-5,4] → 6 (subarray [4,-1,2,1])
# Time: O(n), Space: O(1)
```

**Problem 3: Product of Array Except Self**
```python
def product_except_self(nums):
    """
    Return array where output[i] = product of all elements except nums[i]
    Without using division and in O(n) time
    """
    n = len(nums)
    result = [1] * n
    
    # Left pass: result[i] = product of all elements to the left
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Right pass: multiply by product of all elements to the right
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result

# Example: [1,2,3,4] → [24,12,8,6]
# Time: O(n), Space: O(1) (excluding output array)
```

**Problem 4: Rotate Array**
```python
def rotate(nums, k):
    """
    Rotate array to the right by k steps
    """
    n = len(nums)
    k %= n  # Handle k > n
    
    # Reverse entire array
    reverse(nums, 0, n - 1)
    # Reverse first k elements
    reverse(nums, 0, k - 1)
    # Reverse remaining elements
    reverse(nums, k, n - 1)

def reverse(nums, start, end):
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1

# Example: [1,2,3,4,5], k=2 → [4,5,1,2,3]
# Time: O(n), Space: O(1)
```

**Problem 5: Contains Duplicate**
```python
def contains_duplicate(nums):
    """
    Check if array contains any duplicates
    """
    return len(nums) != len(set(nums))

# Or using hash set for early termination:
def contains_duplicate_optimized(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# Time: O(n), Space: O(n)
```

### 2D Arrays (Matrices)

**Basic Operations:**

```python
# Creating 2D arrays
matrix = [[1, 2, 3], 
          [4, 5, 6], 
          [7, 8, 9]]

# Create m x n matrix filled with zeros
m, n = 3, 4
matrix = [[0] * n for _ in range(m)]

# Accessing elements
element = matrix[row][col]  # O(1)

# Dimensions
rows = len(matrix)
cols = len(matrix[0]) if matrix else 0

# Iterating
for i in range(rows):
    for j in range(cols):
        print(matrix[i][j])

# Or
for row in matrix:
    for element in row:
        print(element)
```

**Problem 1: Matrix Traversal Patterns**

```python
# Row-wise traversal
for i in range(rows):
    for j in range(cols):
        print(matrix[i][j])

# Column-wise traversal
for j in range(cols):
    for i in range(rows):
        print(matrix[i][j])

# Diagonal traversal (main diagonal)
for i in range(min(rows, cols)):
    print(matrix[i][i])

# Anti-diagonal
for i in range(min(rows, cols)):
    print(matrix[i][cols - 1 - i])

# Spiral traversal
def spiral_order(matrix):
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        
        # Traverse down
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        
        # Traverse left
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        
        # Traverse up
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    
    return result
```

**Problem 2: Rotate Matrix 90 Degrees**
```python
def rotate_matrix(matrix):
    """
    Rotate n x n matrix 90 degrees clockwise in-place
    """
    n = len(matrix)
    
    # Transpose matrix
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for i in range(n):
        matrix[i].reverse()

# Example: [[1,2,3],    [[7,4,1],
#           [4,5,6], →   [8,5,2],
#           [7,8,9]]     [9,6,3]]

# Time: O(n²), Space: O(1)
```

**Problem 3: Set Matrix Zeroes**
```python
def set_zeroes(matrix):
    """
    If element is 0, set its entire row and column to 0
    In-place with O(1) space
    """
    rows, cols = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(cols))
    first_col_zero = any(matrix[i][0] == 0 for i in range(rows))
    
    # Use first row and column as markers
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    
    # Set zeros based on markers
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    # Handle first row and column
    if first_row_zero:
        for j in range(cols):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(rows):
            matrix[i][0] = 0

# Time: O(m*n), Space: O(1)
```

**Problem 4: Search in 2D Matrix**
```python
def search_matrix(matrix, target):
    """
    Search in matrix where:
    - Each row is sorted left to right
    - First element of each row > last element of previous row
    """
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_value = matrix[mid // cols][mid % cols]
        
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# Time: O(log(m*n)), Space: O(1)
```

### Dynamic Arrays

Dynamic arrays automatically resize when capacity is reached.

```python
class DynamicArray:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.arr = [None] * self.capacity
    
    def get(self, index):
        if 0 <= index < self.size:
            return self.arr[index]
        raise IndexError("Index out of bounds")
    
    def set(self, index, value):
        if 0 <= index < self.size:
            self.arr[index] = value
        else:
            raise IndexError("Index out of bounds")
    
    def append(self, value):
        # Resize if needed
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        
        self.arr[self.size] = value
        self.size += 1
    
    def _resize(self, new_capacity):
        new_arr = [None] * new_capacity
        for i in range(self.size):
            new_arr[i] = self.arr[i]
        self.arr = new_arr
        self.capacity = new_capacity
    
    def __len__(self):
        return self.size

# Amortized O(1) append due to doubling strategy
```

## Strings

### String Basics

Strings are immutable sequences of characters in most languages.

**Time Complexities:**
- Access: O(1)
- Concatenation: O(n) (creates new string)
- Substring: O(n)
- Search: O(n*m) naive, O(n+m) with KMP

```python
# Creating strings
s = "Hello"
s = 'World'
s = str(123)  # "123"

# Accessing characters
first = s[0]  # O(1)
last = s[-1]  # O(1)

# Strings are immutable - this creates new string
s = s + " World"  # O(n)

# String methods
s.lower()  # Convert to lowercase
s.upper()  # Convert to uppercase
s.strip()  # Remove leading/trailing whitespace
s.split(' ')  # Split into list
','.join(['a', 'b', 'c'])  # Join list into string

# Slicing
substring = s[2:5]  # O(k)
reversed_s = s[::-1]  # O(n)

# Character operations
ord('A')  # Get ASCII value: 65
chr(65)  # Get character from ASCII: 'A'
```

### Common String Problems

**Problem 1: Valid Palindrome**
```python
def is_palindrome(s):
    """
    Check if string is a palindrome (ignore non-alphanumeric, case-insensitive)
    """
    # Filter and convert
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

# Or using two pointers (more space-efficient):
def is_palindrome_two_pointers(s):
    left, right = 0, len(s) - 1
    
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

# Time: O(n), Space: O(1) for two-pointer version
```

**Problem 2: Longest Substring Without Repeating Characters**
```python
def length_longest_substring(s):
    """
    Find length of longest substring without repeating characters
    """
    char_index = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        # If char seen and is in current window
        if char in char_index and char_index[char] >= start:
            # Move start to after last occurrence
            start = char_index[char] + 1
        
        char_index[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Example: "abcabcbb" → 3 ("abc")
# Time: O(n), Space: O(min(n, alphabet_size))
```

**Problem 3: Group Anagrams**
```python
def group_anagrams(strs):
    """
    Group strings that are anagrams of each other
    """
    from collections import defaultdict
    
    anagrams = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        anagrams[key].append(s)
    
    return list(anagrams.values())

# Alternative: use character count as key
def group_anagrams_optimized(strs):
    from collections import defaultdict
    
    anagrams = defaultdict(list)
    
    for s in strs:
        # Count characters
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        # Use tuple of counts as key
        anagrams[tuple(count)].append(s)
    
    return list(anagrams.values())

# Example: ["eat","tea","tan","ate","nat","bat"]
# → [["eat","tea","ate"], ["tan","nat"], ["bat"]]

# Time: O(n*k*log k) or O(n*k) optimized, where k is max string length
```

**Problem 4: Longest Palindromic Substring**
```python
def longest_palindrome(s):
    """
    Find longest palindromic substring
    """
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    if not s:
        return ""
    
    start = end = 0
    
    for i in range(len(s)):
        # Odd length palindrome (center is single character)
        len1 = expand_around_center(i, i)
        # Even length palindrome (center is between two characters)
        len2 = expand_around_center(i, i + 1)
        
        max_len = max(len1, len2)
        
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    
    return s[start:end + 1]

# Example: "babad" → "bab" or "aba"
# Time: O(n²), Space: O(1)
```

**Problem 5: String to Integer (atoi)**
```python
def my_atoi(s):
    """
    Convert string to 32-bit signed integer
    """
    s = s.lstrip()  # Remove leading whitespace
    
    if not s:
        return 0
    
    sign = 1
    i = 0
    
    # Check for sign
    if s[0] in ['+', '-']:
        sign = -1 if s[0] == '-' else 1
        i += 1
    
    result = 0
    
    while i < len(s) and s[i].isdigit():
        result = result * 10 + int(s[i])
        i += 1
    
    result *= sign
    
    # Clamp to 32-bit integer range
    INT_MAX = 2**31 - 1
    INT_MIN = -2**31
    
    if result > INT_MAX:
        return INT_MAX
    if result < INT_MIN:
        return INT_MIN
    
    return result

# Time: O(n), Space: O(1)
```

**Problem 6: Valid Parentheses**
```python
def is_valid(s):
    """
    Check if parentheses/brackets are valid and properly nested
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

# Example: "([{}])" → True, "([)]" → False
# Time: O(n), Space: O(n)
```

**Problem 7: Longest Common Prefix**
```python
def longest_common_prefix(strs):
    """
    Find longest common prefix among array of strings
    """
    if not strs:
        return ""
    
    # Start with first string
    prefix = strs[0]
    
    for s in strs[1:]:
        # Shrink prefix until it matches
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix

# Alternative: vertical scanning
def longest_common_prefix_vertical(strs):
    if not strs:
        return ""
    
    for i in range(len(strs[0])):
        char = strs[0][i]
        for s in strs[1:]:
            if i >= len(s) or s[i] != char:
                return strs[0][:i]
    
    return strs[0]

# Example: ["flower","flow","flight"] → "fl"
# Time: O(S) where S is sum of all characters in all strings
```

### String Building Efficiently

**Problem: Building string with repeated concatenation**
```python
# ❌ Inefficient: O(n²) due to string immutability
def build_string_bad(n):
    s = ""
    for i in range(n):
        s += str(i)  # Creates new string each time!
    return s

# ✅ Efficient: O(n) using list
def build_string_good(n):
    parts = []
    for i in range(n):
        parts.append(str(i))
    return ''.join(parts)

# ✅ Best: Use StringBuilder-like pattern
from io import StringIO

def build_string_best(n):
    builder = StringIO()
    for i in range(n):
        builder.write(str(i))
    return builder.getvalue()
```

### Character Frequency Patterns

```python
from collections import Counter

# Count character frequencies
def char_frequency(s):
    # Method 1: Counter
    freq = Counter(s)
    
    # Method 2: Dictionary
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    # Method 3: Array for lowercase letters
    freq = [0] * 26
    for char in s:
        freq[ord(char) - ord('a')] += 1
    
    return freq

# Check if two strings are anagrams
def is_anagram(s1, s2):
    return Counter(s1) == Counter(s2)

# Or
def is_anagram_sorted(s1, s2):
    return sorted(s1) == sorted(s2)
```

## Summary

### Arrays
- **Best for**: Random access, cache-friendly operations
- **Watch out for**: O(n) insertions/deletions, resizing costs
- **Common patterns**: Two pointers, sliding window, prefix sums

### Strings
- **Best for**: Text processing, pattern matching
- **Watch out for**: Immutability (use lists for building), encoding issues
- **Common patterns**: Two pointers, sliding window, hashing, DP

### Key Techniques
1. Two Pointers (opposite/same direction)
2. Sliding Window (fixed/variable size)
3. Hash Maps for O(1) lookups
4. In-place modifications to save space
5. Sorting as preprocessing step

### Practice Tips
- Master the basic operations first
- Understand time/space tradeoffs
- Practice with edge cases (empty, single element, duplicates)
- Learn to recognize patterns in problem statements

