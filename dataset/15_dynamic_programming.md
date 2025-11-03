# Dynamic Programming

## Introduction

Dynamic Programming (DP) is an optimization technique that solves complex problems by breaking them down into simpler subproblems. It stores the results of subproblems to avoid redundant calculations.

### Key Characteristics

1. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
2. **Overlapping Subproblems**: Same subproblems are solved multiple times
3. **Memoization/Tabulation**: Store and reuse computed results

### When to Use DP

- Optimization problems (max/min)
- Counting problems (how many ways)
- Decision problems (is it possible)
- Problem has overlapping subproblems
- Problem exhibits optimal substructure

## DP Approaches

### 1. Top-Down (Memoization)

Start from original problem, recursively solve subproblems, store results.

```python
def fibonacci_memo(n, memo={}):
    """
    Fibonacci with memoization (top-down)
    """
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Time: O(n), Space: O(n)
```

### 2. Bottom-Up (Tabulation)

Start from base cases, iteratively build up to solution.

```python
def fibonacci_tab(n):
    """
    Fibonacci with tabulation (bottom-up)
    """
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Time: O(n), Space: O(n)
```

### 3. Space-Optimized

Reduce space by keeping only necessary previous states.

```python
def fibonacci_optimized(n):
    """
    Fibonacci with space optimization
    """
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    
    return prev1

# Time: O(n), Space: O(1)
```

## Classic DP Problems

### 1. Climbing Stairs

```python
def climb_stairs(n):
    """
    Count ways to climb n stairs (1 or 2 steps at a time)
    """
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    
    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    
    return prev1

# Time: O(n), Space: O(1)
# Relation: dp[i] = dp[i-1] + dp[i-2]
```

### 2. House Robber

```python
def rob(nums):
    """
    Maximum money robbed from non-adjacent houses
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2, prev1 = 0, 0
    
    for num in nums:
        temp = prev1
        prev1 = max(prev1, prev2 + num)
        prev2 = temp
    
    return prev1

# Time: O(n), Space: O(1)
# dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

**House Robber II (Circular):**
```python
def rob_circular(nums):
    """Houses arranged in circle (can't rob first and last)"""
    if len(nums) == 1:
        return nums[0]
    
    def rob_linear(houses):
        prev2, prev1 = 0, 0
        for house in houses:
            prev2, prev1 = prev1, max(prev1, prev2 + house)
        return prev1
    
    # Either skip first or skip last house
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

### 3. Coin Change

```python
def coin_change(coins, amount):
    """
    Minimum number of coins to make amount
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Time: O(amount × coins), Space: O(amount)
```

**Coin Change II (Count Ways):**
```python
def change(amount, coins):
    """
    Number of combinations to make amount
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]

# Time: O(amount × coins), Space: O(amount)
```

### 4. Longest Increasing Subsequence (LIS)

```python
def length_of_lis(nums):
    """
    Length of longest increasing subsequence
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Time: O(n²), Space: O(n)
```

**LIS with Binary Search (Optimized):**
```python
def length_of_lis_optimized(nums):
    """
    LIS using binary search - O(n log n)
    """
    import bisect
    
    tails = []
    
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)

# Time: O(n log n), Space: O(n)
```

## Knapsack Problems

### 0/1 Knapsack

```python
def knapsack_01(weights, values, capacity):
    """
    0/1 Knapsack - Can take item or leave it
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i
            dp[i][w] = dp[i - 1][w]
            
            # Take item i if it fits
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], 
                              dp[i - 1][w - weights[i - 1]] + values[i - 1])
    
    return dp[n][capacity]

# Time: O(n × capacity), Space: O(n × capacity)
```

**Space-Optimized:**
```python
def knapsack_01_optimized(weights, values, capacity):
    """Space-optimized 0/1 knapsack"""
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Traverse backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Time: O(n × capacity), Space: O(capacity)
```

### Unbounded Knapsack

```python
def knapsack_unbounded(weights, values, capacity):
    """
    Unbounded Knapsack - Can take unlimited of each item
    """
    dp = [0] * (capacity + 1)
    
    for w in range(capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Time: O(n × capacity), Space: O(capacity)
```

## Subsequence Problems

### Longest Common Subsequence (LCS)

```python
def longest_common_subsequence(text1, text2):
    """
    Find length of longest common subsequence
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# Time: O(m × n), Space: O(m × n)
```

**Print LCS:**
```python
def print_lcs(text1, text2):
    """Reconstruct the actual LCS"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Backtrack to find LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))
```

### Edit Distance

```python
def min_distance(word1, word2):
    """
    Minimum operations to convert word1 to word2
    Operations: insert, delete, replace
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )
    
    return dp[m][n]

# Time: O(m × n), Space: O(m × n)
```

### Palindromic Subsequence

```python
def longest_palindromic_subsequence(s):
    """
    Length of longest palindromic subsequence
    """
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    # Every single character is palindrome of length 1
    for i in range(n):
        dp[i][i] = 1
    
    # Build table bottom-up
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1]

# Time: O(n²), Space: O(n²)
```

## DP on Strings

### Word Break

```python
def word_break(s, wordDict):
    """
    Check if string can be segmented into words from dictionary
    """
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]

# Time: O(n² × m) where m is avg word length
# Space: O(n)
```

**Word Break II (Return All Sentences):**
```python
def word_break_ii(s, wordDict):
    """Return all possible sentences"""
    word_set = set(wordDict)
    memo = {}
    
    def backtrack(start):
        if start in memo:
            return memo[start]
        
        if start == len(s):
            return [[]]
        
        sentences = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for rest in backtrack(end):
                    sentences.append([word] + rest)
        
        memo[start] = sentences
        return sentences
    
    return [' '.join(sentence) for sentence in backtrack(0)]
```

### Distinct Subsequences

```python
def num_distinct(s, t):
    """
    Count distinct subsequences of s that equal t
    """
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Empty string is subsequence of any string
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    
    return dp[m][n]

# Time: O(m × n), Space: O(m × n)
```

## DP on Grids

### Unique Paths

```python
def unique_paths(m, n):
    """
    Number of unique paths from top-left to bottom-right
    Can only move right or down
    """
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[m - 1][n - 1]

# Time: O(m × n), Space: O(m × n)
```

**With Obstacles:**
```python
def unique_paths_with_obstacles(grid):
    """Grid with obstacles (1 = obstacle, 0 = empty)"""
    if not grid or grid[0][0] == 1:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dp[i][j] = 0
            elif i == 0 and j == 0:
                continue
            else:
                if i > 0:
                    dp[i][j] += dp[i - 1][j]
                if j > 0:
                    dp[i][j] += dp[i][j - 1]
    
    return dp[m - 1][n - 1]
```

### Minimum Path Sum

```python
def min_path_sum(grid):
    """
    Find path with minimum sum from top-left to bottom-right
    """
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    
    # First row
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    
    # First column
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    
    # Rest of grid
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    
    return dp[m - 1][n - 1]

# Time: O(m × n), Space: O(m × n)
```

### Maximal Square

```python
def maximal_square(matrix):
    """
    Find largest square containing only 1s
    """
    if not matrix:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i - 1][j], 
                                  dp[i][j - 1], 
                                  dp[i - 1][j - 1]) + 1
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side

# Time: O(m × n), Space: O(m × n)
```

## State Machine DP

### Best Time to Buy and Sell Stock

**Single Transaction:**
```python
def max_profit_one(prices):
    """Maximum profit with at most one transaction"""
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    
    return max_profit
```

**Multiple Transactions:**
```python
def max_profit_unlimited(prices):
    """Maximum profit with unlimited transactions"""
    profit = 0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    
    return profit
```

**At Most K Transactions:**
```python
def max_profit_k(k, prices):
    """Maximum profit with at most k transactions"""
    if not prices or k == 0:
        return 0
    
    if k >= len(prices) // 2:
        return max_profit_unlimited(prices)
    
    # dp[i][j] = max profit with at most i transactions by day j
    buy = [-prices[0]] * (k + 1)
    sell = [0] * (k + 1)
    
    for price in prices:
        for i in range(k, 0, -1):
            sell[i] = max(sell[i], buy[i] + price)
            buy[i] = max(buy[i], sell[i - 1] - price)
    
    return sell[k]
```

## DP Patterns Summary

### Pattern 1: Linear DP
- Single array/string
- dp[i] depends on previous states
- Examples: Fibonacci, Climbing Stairs, House Robber

### Pattern 2: Two Sequence DP
- Two arrays/strings
- dp[i][j] combines both sequences
- Examples: LCS, Edit Distance

### Pattern 3: Interval DP
- Process intervals of increasing length
- dp[i][j] = answer for range [i, j]
- Examples: Palindromic Subsequence, Matrix Chain

### Pattern 4: Knapsack DP
- Items and capacity constraint
- Optimize value within constraint
- Examples: 0/1 Knapsack, Coin Change

### Pattern 5: Grid DP
- 2D grid traversal
- dp[i][j] = answer for cell (i, j)
- Examples: Unique Paths, Min Path Sum

### Pattern 6: State Machine DP
- Multiple states at each step
- Transitions between states
- Examples: Stock Trading, Jump Game

## DP Optimization Techniques

1. **Space Optimization**: Keep only necessary previous states
2. **Rolling Array**: Use modulo to reuse array space
3. **Monotonic Queue/Stack**: Optimize range queries
4. **Divide and Conquer**: Combine with DP
5. **Convex Hull Trick**: Optimize linear DP

## Summary

**Key Concepts:**
- Identify optimal substructure and overlapping subproblems
- Choose between top-down (memoization) and bottom-up (tabulation)
- Define state and transition clearly
- Consider space optimization

**Common Mistakes:**
- Not identifying DP pattern
- Wrong state definition
- Incorrect base cases
- Missing state transitions

**Practice Tips:**
- Start with classic problems (Fibonacci, Climbing Stairs)
- Learn to identify DP patterns
- Practice state definition and transition
- Optimize space after correctness
- Draw state transition diagrams

