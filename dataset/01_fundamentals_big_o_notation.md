# Big O Notation and Time & Space Complexity

## Introduction

Big O notation is a mathematical notation used to describe the asymptotic behavior of functions. In computer science, it's used to classify algorithms according to how their running time or space requirements grow as the input size grows.

## What is Big O Notation?

Big O notation characterizes functions according to their growth rates. It provides an upper bound on the growth rate of an algorithm's time or space requirements. When we say an algorithm is O(n), we mean that the time/space it requires grows linearly with the input size n.

## Common Time Complexities

### O(1) - Constant Time

An algorithm that always takes the same amount of time regardless of input size.

**Examples:**
- Accessing an array element by index: `arr[5]`
- Hash table lookup (average case)
- Simple arithmetic operations
- Pushing/popping from a stack

```python
def get_first_element(arr):
    return arr[0]  # O(1) - always one operation
```

### O(log n) - Logarithmic Time

The algorithm's running time grows logarithmically with input size. Common in algorithms that divide the problem in half at each step.

**Examples:**
- Binary search on a sorted array
- Balanced binary search tree operations
- Finding an element in a balanced tree

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**Why logarithmic?** Each iteration eliminates half the remaining elements. For n elements, we need at most log₂(n) iterations.

### O(n) - Linear Time

The running time grows directly proportional to the input size.

**Examples:**
- Iterating through an array once
- Linear search
- Finding min/max in unsorted array
- Counting elements

```python
def find_maximum(arr):
    max_val = arr[0]
    for num in arr:  # O(n) - one pass through array
        if num > max_val:
            max_val = num
    return max_val
```

### O(n log n) - Linearithmic Time

Common in efficient sorting algorithms.

**Examples:**
- Merge Sort
- Quick Sort (average case)
- Heap Sort
- Sorting-based solutions

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # log n divisions
    right = merge_sort(arr[mid:])   # log n divisions
    
    return merge(left, right)       # n work at each level
```

### O(n²) - Quadratic Time

Running time is proportional to the square of input size. Often seen in nested loops.

**Examples:**
- Bubble Sort, Selection Sort, Insertion Sort
- Checking all pairs in an array
- Simple matrix multiplication

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):           # O(n)
        for j in range(n - i - 1):  # O(n)
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

### O(2ⁿ) - Exponential Time

Doubles with each additional input element. Very inefficient for large inputs.

**Examples:**
- Recursive Fibonacci (naive implementation)
- Generating all subsets of a set
- Solving Tower of Hanoi

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)  # O(2^n)
```

### O(n!) - Factorial Time

Extremely slow, grows factorially with input size.

**Examples:**
- Generating all permutations
- Traveling Salesman Problem (brute force)
- Solving N-Queens (brute force)

```python
def generate_permutations(arr):
    if len(arr) <= 1:
        return [arr]
    
    perms = []
    for i in range(len(arr)):
        rest = arr[:i] + arr[i+1:]
        for p in generate_permutations(rest):
            perms.append([arr[i]] + p)
    return perms  # O(n!)
```

## Space Complexity

Space complexity measures the amount of memory an algorithm needs relative to input size.

### Types of Space

1. **Input Space**: Space required to store the input
2. **Auxiliary Space**: Extra space required by the algorithm (excluding input)
3. **Total Space**: Input space + Auxiliary space

### Common Space Complexities

**O(1) - Constant Space:**
```python
def swap_elements(arr, i, j):
    temp = arr[i]  # Only one extra variable
    arr[i] = arr[j]
    arr[j] = temp
```

**O(n) - Linear Space:**
```python
def create_frequency_map(arr):
    freq = {}  # Space grows with input
    for num in arr:
        freq[num] = freq.get(num, 0) + 1
    return freq
```

**O(log n) - Logarithmic Space:**
```python
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
    # Recursion depth: O(log n)
```

## Rules for Calculating Big O

### 1. Drop Constants

O(2n) → O(n)
O(500) → O(1)
O(13n²) → O(n²)

```python
def example(arr):
    for i in range(len(arr)):  # O(n)
        print(arr[i])
    
    for i in range(len(arr)):  # O(n)
        print(arr[i])
    
    # Total: O(n) + O(n) = O(2n) = O(n)
```

### 2. Drop Non-Dominant Terms

O(n² + n) → O(n²)
O(n + log n) → O(n)
O(5*2ⁿ + 1000n²) → O(2ⁿ)

```python
def example(arr):
    # O(n²) loop
    for i in range(len(arr)):
        for j in range(len(arr)):
            print(arr[i], arr[j])
    
    # O(n) loop
    for i in range(len(arr)):
        print(arr[i])
    
    # Total: O(n²) + O(n) = O(n²)
```

### 3. Different Inputs → Different Variables

```python
def example(arr1, arr2):
    for i in arr1:        # O(a) where a = len(arr1)
        print(i)
    
    for j in arr2:        # O(b) where b = len(arr2)
        print(j)
    
    # Total: O(a + b), NOT O(n)
```

### 4. Be Aware of Best, Average, and Worst Cases

Different inputs can lead to different complexities:

```python
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

# Best case: O(1) - target is first element
# Average case: O(n/2) → O(n)
# Worst case: O(n) - target is last or not present
```

## Analyzing Complex Code

### Example 1: Nested Loops with Different Ranges

```python
def example(n):
    for i in range(n):           # O(n)
        for j in range(i):       # O(i) where i goes from 0 to n
            print(i, j)

# Total: 0 + 1 + 2 + ... + (n-1) = n(n-1)/2 = O(n²)
```

### Example 2: Loop with Multiplicative Changes

```python
def example(n):
    i = 1
    while i < n:
        print(i)
        i *= 2

# i takes values: 1, 2, 4, 8, ..., 2^k where 2^k < n
# Therefore k < log₂(n), so O(log n)
```

### Example 3: Multiple Recursive Calls

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Each call makes 2 more calls, creating a binary tree
# Tree height: n, nodes: 2^n
# Time: O(2^n), Space: O(n) for recursion stack
```

## Practical Tips

### 1. Single Loop over n elements → O(n)

### 2. Nested loops over n elements → multiply complexities
- Two nested loops: O(n²)
- Three nested loops: O(n³)

### 3. Halving/Doubling in loop → O(log n)

### 4. Sorting → typically O(n log n) for good algorithms

### 5. Recursion → analyze tree depth and branching factor

## Common Mistakes

### Mistake 1: Confusing O(n + m) with O(n * m)

```python
# O(n + m) - sequential
for i in range(n):
    print(i)
for j in range(m):
    print(j)

# O(n * m) - nested
for i in range(n):
    for j in range(m):
        print(i, j)
```

### Mistake 2: Ignoring Hidden Complexity

```python
def example(arr):
    for i in range(len(arr)):
        arr.sort()  # O(n log n) each iteration!
    
    # Total: O(n) * O(n log n) = O(n² log n)
    # NOT O(n)!
```

### Mistake 3: Assuming All Operations Are O(1)

- String concatenation: O(n) in most languages
- Array insertion at beginning: O(n)
- Hash table operations: O(1) average, O(n) worst case

## Comparison of Growth Rates

For n = 1,000,000:

| Complexity | Operations | Realistic? |
|------------|------------|------------|
| O(1) | 1 | Instant |
| O(log n) | ~20 | Instant |
| O(n) | 1,000,000 | Fast |
| O(n log n) | ~20,000,000 | Fast |
| O(n²) | 1,000,000,000,000 | Slow |
| O(2ⁿ) | 2^1000000 | Never finishes |
| O(n!) | 1000000! | Never finishes |

## Space-Time Tradeoff

Often you can trade space for time:

### Example: Fibonacci

**Time-optimized (with memoization):**
```python
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Time: O(n), Space: O(n)
```

**Space-optimized (iterative):**
```python
def fib_iterative(n):
    if n <= 1:
        return n
    
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr

# Time: O(n), Space: O(1)
```

## Summary

Understanding Big O notation is crucial for:
1. Comparing algorithm efficiency
2. Predicting scalability
3. Making informed optimization decisions
4. Communicating algorithm performance

Always consider both time and space complexity, and remember that Big O gives us the worst-case upper bound, helping us understand how algorithms scale with input size.

