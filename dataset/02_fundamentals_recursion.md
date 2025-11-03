# Recursion and Recursion Patterns

## Introduction to Recursion

Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems. A recursive solution consists of:

1. **Base Case**: The simplest instance of the problem, solved directly
2. **Recursive Case**: Breaks the problem into simpler instances of the same problem

## Basic Recursion Concepts

### Anatomy of a Recursive Function

```python
def recursive_function(parameters):
    # Base case(s) - when to stop recursing
    if base_condition:
        return base_value
    
    # Recursive case - break down the problem
    # Make the problem smaller
    result = recursive_function(modified_parameters)
    
    # Process result and return
    return processed_result
```

### Simple Example: Factorial

```python
def factorial(n):
    # Base case
    if n == 0 or n == 1:
        return 1
    
    # Recursive case
    return n * factorial(n - 1)

# factorial(5) = 5 * factorial(4)
#              = 5 * 4 * factorial(3)
#              = 5 * 4 * 3 * factorial(2)
#              = 5 * 4 * 3 * 2 * factorial(1)
#              = 5 * 4 * 3 * 2 * 1 = 120
```

### How Recursion Works: The Call Stack

```python
factorial(3)
├─ 3 * factorial(2)
│  ├─ 2 * factorial(1)
│  │  └─ return 1
│  └─ return 2 * 1 = 2
└─ return 3 * 2 = 6
```

Each function call is placed on the call stack. When a base case is reached, calls start returning and popping off the stack.

## Common Recursion Patterns

### Pattern 1: Linear Recursion (Single Recursive Call)

Each function makes at most one recursive call.

**Example: Sum of Array**
```python
def array_sum(arr, index=0):
    # Base case: reached end of array
    if index == len(arr):
        return 0
    
    # Recursive case: current element + sum of rest
    return arr[index] + array_sum(arr, index + 1)

# array_sum([1, 2, 3, 4, 5])
# = 1 + array_sum([2, 3, 4, 5])
# = 1 + 2 + array_sum([3, 4, 5])
# = 1 + 2 + 3 + array_sum([4, 5])
# = 1 + 2 + 3 + 4 + array_sum([5])
# = 1 + 2 + 3 + 4 + 5 + 0 = 15
```

**Example: Power Function**
```python
def power(base, exponent):
    # Base case
    if exponent == 0:
        return 1
    
    # Recursive case
    return base * power(base, exponent - 1)

# Time: O(n), Space: O(n) for call stack
```

### Pattern 2: Binary Recursion (Two Recursive Calls)

Each function makes two recursive calls.

**Example: Fibonacci Numbers**
```python
def fibonacci(n):
    # Base cases
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Two recursive calls
    return fibonacci(n - 1) + fibonacci(n - 2)

# Recursion tree for fib(5):
#                    fib(5)
#                   /      \
#              fib(4)      fib(3)
#             /    \       /    \
#        fib(3)  fib(2) fib(2) fib(1)
#        /   \   /   \   /   \
#    fib(2) fib(1) ...
#    /   \
# fib(1) fib(0)

# Time: O(2^n), Space: O(n) for call stack
```

**Example: Binary Tree Traversal**
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    if root is None:
        return []
    
    # Left subtree, root, right subtree
    return (inorder_traversal(root.left) + 
            [root.val] + 
            inorder_traversal(root.right))
```

### Pattern 3: Multiple Recursion

Makes more than two recursive calls.

**Example: Tree with Multiple Children**
```python
class TreeNode:
    def __init__(self, val, children=None):
        self.val = val
        self.children = children or []

def sum_tree(root):
    if root is None:
        return 0
    
    total = root.val
    for child in root.children:
        total += sum_tree(child)
    
    return total
```

### Pattern 4: Tail Recursion

The recursive call is the last operation in the function.

**Example: Tail Recursive Factorial**
```python
def factorial_tail(n, accumulator=1):
    # Base case
    if n == 0:
        return accumulator
    
    # Tail recursive call (last operation)
    return factorial_tail(n - 1, n * accumulator)

# factorial_tail(5, 1)
# = factorial_tail(4, 5)
# = factorial_tail(3, 20)
# = factorial_tail(2, 60)
# = factorial_tail(1, 120)
# = factorial_tail(0, 120)
# = 120
```

Tail recursion can be optimized by compilers into iteration (Tail Call Optimization).

### Pattern 5: Divide and Conquer

Divides problem into independent subproblems, solves them recursively, and combines results.

**Example: Merge Sort**
```python
def merge_sort(arr):
    # Base case: array of size 0 or 1
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (combine)
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

**Example: Binary Search**
```python
def binary_search(arr, target, left, right):
    # Base case: not found
    if left > right:
        return -1
    
    # Divide
    mid = (left + right) // 2
    
    # Check middle
    if arr[mid] == target:
        return mid
    
    # Conquer one half
    if arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)

# Time: O(log n), Space: O(log n)
```

### Pattern 6: Backtracking

Explores all possible solutions by trying partial solutions and abandoning them if they don't work.

**Example: Generate All Subsets**
```python
def subsets(nums):
    result = []
    
    def backtrack(start, current):
        # Add current subset to result
        result.append(current[:])
        
        # Try adding each remaining element
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()  # Backtrack
    
    backtrack(0, [])
    return result

# For [1, 2, 3]:
# [], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]
```

**Example: N-Queens Problem**
```python
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # Check anti-diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'  # Backtrack
    
    backtrack(0)
    return result
```

### Pattern 7: Memoization (Top-Down Dynamic Programming)

Stores results of expensive function calls and returns cached result when same inputs occur again.

**Example: Fibonacci with Memoization**
```python
def fib_memo(n, memo={}):
    # Check if already computed
    if n in memo:
        return memo[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Compute and store
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

# Time: O(n), Space: O(n)
# Much better than O(2^n) without memoization!
```

**Example: Climbing Stairs with Memoization**
```python
def climb_stairs(n, memo={}):
    """
    You can climb 1 or 2 steps at a time.
    How many distinct ways to climb n steps?
    """
    if n in memo:
        return memo[n]
    
    if n <= 2:
        return n
    
    memo[n] = climb_stairs(n - 1, memo) + climb_stairs(n - 2, memo)
    return memo[n]
```

## Advanced Recursion Techniques

### Technique 1: Helper Function Pattern

Use a helper function to handle additional parameters.

```python
def reverse_string(s):
    def helper(left, right):
        if left >= right:
            return
        
        # Swap
        s[left], s[right] = s[right], s[left]
        
        # Recurse
        helper(left + 1, right - 1)
    
    s_list = list(s)
    helper(0, len(s_list) - 1)
    return ''.join(s_list)
```

### Technique 2: Multiple Base Cases

Some problems need multiple base cases.

```python
def tribonacci(n):
    # Multiple base cases
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    return tribonacci(n - 1) + tribonacci(n - 2) + tribonacci(n - 3)
```

### Technique 3: Pre and Post Processing

Perform operations before and after recursive call.

```python
def print_list_forward_and_backward(arr, index=0):
    if index == len(arr):
        return
    
    # Pre-processing: print on the way down
    print(f"Going down: {arr[index]}")
    
    # Recursive call
    print_list_forward_and_backward(arr, index + 1)
    
    # Post-processing: print on the way up
    print(f"Coming up: {arr[index]}")

# For [1, 2, 3]:
# Going down: 1
# Going down: 2
# Going down: 3
# Coming up: 3
# Coming up: 2
# Coming up: 1
```

### Technique 4: Mutual Recursion

Two or more functions call each other.

```python
def is_even(n):
    if n == 0:
        return True
    return is_odd(n - 1)

def is_odd(n):
    if n == 0:
        return False
    return is_even(n - 1)
```

## Common Recursion Problems

### Problem 1: Power with Optimization

```python
def power_optimized(base, exp):
    """
    Compute base^exp using divide and conquer
    Time: O(log n) instead of O(n)
    """
    if exp == 0:
        return 1
    
    if exp < 0:
        return 1 / power_optimized(base, -exp)
    
    half = power_optimized(base, exp // 2)
    
    if exp % 2 == 0:
        return half * half
    else:
        return base * half * half

# power(2, 10)
# = power(2, 5)^2
# = (2 * power(2, 2)^2)^2
# = (2 * power(2, 1)^2^2)^2
# = (2 * (2 * 1)^2^2)^2
```

### Problem 2: String Permutations

```python
def permutations(s):
    result = []
    
    def backtrack(path, remaining):
        if not remaining:
            result.append(path)
            return
        
        for i in range(len(remaining)):
            backtrack(
                path + remaining[i],
                remaining[:i] + remaining[i+1:]
            )
    
    backtrack("", s)
    return result

# permutations("ABC")
# = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]
```

### Problem 3: Flatten Nested List

```python
def flatten(nested_list):
    result = []
    
    for item in nested_list:
        if isinstance(item, list):
            # Recursive case: item is a list
            result.extend(flatten(item))
        else:
            # Base case: item is not a list
            result.append(item)
    
    return result

# flatten([1, [2, 3], [[4], 5]])
# = [1, 2, 3, 4, 5]
```

### Problem 4: Path Sum in Binary Tree

```python
def has_path_sum(root, target_sum):
    # Base case: empty tree
    if root is None:
        return False
    
    # Base case: leaf node
    if root.left is None and root.right is None:
        return root.val == target_sum
    
    # Recursive case: check left and right subtrees
    remaining = target_sum - root.val
    return (has_path_sum(root.left, remaining) or 
            has_path_sum(root.right, remaining))
```

## Recursion vs Iteration

### When to Use Recursion

✅ **Good for:**
- Tree and graph traversals
- Divide and conquer algorithms
- Problems naturally defined recursively
- Backtracking problems
- When code clarity is more important than performance

### When to Use Iteration

✅ **Good for:**
- Simple loops
- Performance-critical code
- Limited stack space
- Tail-recursive functions (can be converted to loops)

### Converting Recursion to Iteration

**Recursive:**
```python
def sum_n(n):
    if n == 0:
        return 0
    return n + sum_n(n - 1)
```

**Iterative:**
```python
def sum_n_iterative(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
```

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Missing Base Case

```python
# ❌ Wrong: No base case, infinite recursion
def count_down(n):
    print(n)
    count_down(n - 1)

# ✅ Correct: Has base case
def count_down(n):
    if n <= 0:
        return
    print(n)
    count_down(n - 1)
```

### Pitfall 2: Not Making Progress Toward Base Case

```python
# ❌ Wrong: Always calls with same argument
def broken(n):
    if n == 0:
        return 0
    return broken(n)  # Never decreases n!

# ✅ Correct: Moves toward base case
def correct(n):
    if n == 0:
        return 0
    return correct(n - 1)
```

### Pitfall 3: Stack Overflow

```python
# ❌ Can cause stack overflow for large n
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# ✅ Better: Use iteration for large inputs
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

### Pitfall 4: Redundant Computation

```python
# ❌ Recomputes same values many times
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# ✅ Better: Use memoization
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]
```

## Debugging Recursive Functions

### Technique 1: Add Print Statements

```python
def factorial(n, depth=0):
    indent = "  " * depth
    print(f"{indent}factorial({n}) called")
    
    if n == 0 or n == 1:
        print(f"{indent}factorial({n}) returning 1")
        return 1
    
    result = n * factorial(n - 1, depth + 1)
    print(f"{indent}factorial({n}) returning {result}")
    return result
```

### Technique 2: Trace the Call Stack

Draw out the recursion tree to visualize what's happening.

### Technique 3: Test with Small Inputs

Always test with the smallest possible inputs first.

## Summary

Recursion is a powerful technique that:
- Simplifies complex problems
- Makes code more elegant and readable
- Is essential for tree/graph problems
- Requires careful attention to base cases and progress

Master these patterns:
1. Linear recursion
2. Binary recursion
3. Divide and conquer
4. Backtracking
5. Memoization

Remember:
- Always have a base case
- Make progress toward the base case
- Consider stack space usage
- Use memoization to avoid redundant computation
- Convert to iteration when appropriate for performance

