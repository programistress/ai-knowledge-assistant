# Backtracking and Advanced Topics

## Backtracking

### Introduction

Backtracking is an algorithmic technique for solving problems by trying to build a solution incrementally, abandoning solutions ("backtracking") when they fail to satisfy constraints.

**Key Concepts:**
- Try all possible solutions
- Abandon partial solutions that can't lead to valid solution
- Recursively explore solution space
- Undo choices (backtrack) when necessary

### Backtracking Template

```python
def backtrack(path, choices):
    """
    General backtracking template
    """
    # Base case: found solution
    if is_solution(path):
        results.append(path[:])  # Add copy
        return
    
    # Try each choice
    for choice in choices:
        # Make choice
        if is_valid(choice, path):
            path.append(choice)
            
            # Recurse
            backtrack(path, next_choices)
            
            # Undo choice (backtrack)
            path.pop()
```

### Classic Backtracking Problems

#### 1. Permutations

```python
def permute(nums):
    """
    Generate all permutations of nums
    """
    result = []
    
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        
        for i in range(len(remaining)):
            # Choose
            path.append(remaining[i])
            # Explore
            backtrack(path, remaining[:i] + remaining[i+1:])
            # Unchoose
            path.pop()
    
    backtrack([], nums)
    return result

# Time: O(n × n!), Space: O(n!)
# Example: [1,2,3] → [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**Permutations II (with duplicates):**
```python
def permute_unique(nums):
    """Permutations with duplicate numbers"""
    result = []
    nums.sort()  # Sort to handle duplicates
    
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            # Skip if used or duplicate
            if used[i] or (i > 0 and nums[i] == nums[i-1] and not used[i-1]):
                continue
            
            path.append(nums[i])
            used[i] = True
            backtrack(path, used)
            path.pop()
            used[i] = False
    
    backtrack([], [False] * len(nums))
    return result
```

#### 2. Combinations

```python
def combine(n, k):
    """
    Generate all combinations of k numbers from 1..n
    """
    result = []
    
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return result

# Time: O(C(n,k) × k), Space: O(C(n,k))
# Example: n=4, k=2 → [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```

#### 3. Subsets

```python
def subsets(nums):
    """
    Generate all subsets (power set)
    """
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# Time: O(n × 2^n), Space: O(2^n)
# Example: [1,2,3] → [[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]
```

**Subsets II (with duplicates):**
```python
def subsets_with_dup(nums):
    """Subsets with duplicate numbers"""
    result = []
    nums.sort()
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[i-1]:
                continue
            
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

#### 4. N-Queens

```python
def solve_n_queens(n):
    """
    Place n queens on n×n chessboard so none attack each other
    """
    result = []
    board = [['.'] * n for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check upper-left diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # Check upper-right diagonal
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
                board[row][col] = '.'
    
    backtrack(0)
    return result

# Time: O(n!), Space: O(n²)
```

#### 5. Sudoku Solver

```python
def solve_sudoku(board):
    """
    Solve 9×9 Sudoku puzzle
    """
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False
        
        # Check 3×3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            
                            if backtrack():
                                return True
                            
                            board[i][j] = '.'
                    
                    return False
        
        return True
    
    backtrack()
```

#### 6. Palindrome Partitioning

```python
def partition(s):
    """
    Partition string into palindrome substrings
    """
    result = []
    
    def is_palindrome(sub):
        return sub == sub[::-1]
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                path.append(substring)
                backtrack(end, path)
                path.pop()
    
    backtrack(0, [])
    return result

# Example: "aab" → [["a","a","b"],["aa","b"]]
```

#### 7. Letter Combinations of Phone Number

```python
def letter_combinations(digits):
    """
    Generate all letter combinations from phone number
    """
    if not digits:
        return []
    
    phone = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, path):
        if index == len(digits):
            result.append(''.join(path))
            return
        
        for letter in phone[digits[index]]:
            path.append(letter)
            backtrack(index + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# Example: "23" → ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

#### 8. Generate Parentheses

```python
def generate_parentheses(n):
    """
    Generate all valid combinations of n pairs of parentheses
    """
    result = []
    
    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return
        
        # Add opening parenthesis
        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()
        
        # Add closing parenthesis
        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()
    
    backtrack([], 0, 0)
    return result

# Example: n=3 → ["((()))","(()())","(())()","()(())","()()()"]
```

#### 9. Word Search

```python
def exist(board, word):
    """
    Find if word exists in 2D board
    """
    rows, cols = len(board), len(board[0])
    
    def backtrack(r, c, index):
        if index == len(word):
            return True
        
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[index]):
            return False
        
        # Mark as visited
        temp = board[r][c]
        board[r][c] = '#'
        
        # Explore all 4 directions
        found = (backtrack(r + 1, c, index + 1) or
                backtrack(r - 1, c, index + 1) or
                backtrack(r, c + 1, index + 1) or
                backtrack(r, c - 1, index + 1))
        
        # Restore
        board[r][c] = temp
        
        return found
    
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    
    return False

# Time: O(m×n×4^L) where L is word length
```

## Advanced Topics

### Bit Manipulation

#### Basic Operations

```python
# Set bit at position i
def set_bit(num, i):
    return num | (1 << i)

# Clear bit at position i
def clear_bit(num, i):
    return num & ~(1 << i)

# Toggle bit at position i
def toggle_bit(num, i):
    return num ^ (1 << i)

# Check if bit at position i is set
def is_bit_set(num, i):
    return (num & (1 << i)) != 0

# Get bit at position i
def get_bit(num, i):
    return (num >> i) & 1
```

#### Common Bit Manipulation Problems

**Count Set Bits:**
```python
def count_set_bits(n):
    """Count number of 1s in binary representation"""
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# Or using Brian Kernighan's algorithm
def count_set_bits_fast(n):
    count = 0
    while n:
        n &= (n - 1)  # Clear rightmost set bit
        count += 1
    return count
```

**Single Number:**
```python
def single_number(nums):
    """Find number that appears once (others appear twice)"""
    result = 0
    for num in nums:
        result ^= num
    return result

# XOR properties: a^a=0, a^0=a
```

**Power of Two:**
```python
def is_power_of_two(n):
    """Check if n is power of 2"""
    return n > 0 and (n & (n - 1)) == 0
```

**Reverse Bits:**
```python
def reverse_bits(n):
    """Reverse bits of 32-bit integer"""
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
```

### String Matching Algorithms

#### KMP (Knuth-Morris-Pratt)

```python
def kmp_search(text, pattern):
    """
    KMP pattern matching algorithm
    """
    def compute_lps(pattern):
        """Compute Longest Prefix Suffix array"""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    n, m = len(text), len(pattern)
    lps = compute_lps(pattern)
    
    i = j = 0
    occurrences = []
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            occurrences.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return occurrences

# Time: O(n + m), Space: O(m)
```

#### Rabin-Karp

```python
def rabin_karp(text, pattern):
    """
    Rabin-Karp string matching using rolling hash
    """
    n, m = len(text), len(pattern)
    
    if m > n:
        return []
    
    # Parameters for hash
    d = 256  # Number of characters
    q = 101  # Prime number
    
    # Calculate hash value of pattern and first window
    pattern_hash = 0
    text_hash = 0
    h = pow(d, m - 1, q)
    
    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % q
        text_hash = (d * text_hash + ord(text[i])) % q
    
    occurrences = []
    
    # Slide pattern over text
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            # Check character by character
            if text[i:i + m] == pattern:
                occurrences.append(i)
        
        # Calculate hash for next window
        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % q
            if text_hash < 0:
                text_hash += q
    
    return occurrences

# Time: O(n + m) average, O(nm) worst
```

### Tries (Prefix Trees)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert word into trie"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
    
    def search(self, word):
        """Search for exact word"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        """Check if any word starts with prefix"""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True
    
    def all_words_with_prefix(self, prefix):
        """Return all words with given prefix"""
        node = self.root
        
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # DFS to find all words
        words = []
        
        def dfs(node, path):
            if node.is_end_of_word:
                words.append(prefix + path)
            
            for char, child in node.children.items():
                dfs(child, path + char)
        
        dfs(node, "")
        return words

# Time: O(m) for insert, search, startsWith where m is key length
# Space: O(ALPHABET_SIZE × N × M) worst case
```

**Word Search II (Using Trie):**
```python
def find_words(board, words):
    """Find all words from list that exist in board"""
    # Build trie
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    rows, cols = len(board), len(board[0])
    result = set()
    
    def backtrack(r, c, node, path):
        if node.is_end_of_word:
            result.add(path)
        
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] not in node.children):
            return
        
        char = board[r][c]
        board[r][c] = '#'
        
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            backtrack(r + dr, c + dc, node.children[char], path + char)
        
        board[r][c] = char
    
    for i in range(rows):
        for j in range(cols):
            if board[i][j] in trie.root.children:
                backtrack(i, j, trie.root, "")
    
    return list(result)
```

### LRU Cache Implementation

```python
class Node:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    """
    LRU Cache with O(1) operations
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        
        # Dummy head and tail
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node):
        """Remove node from linked list"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node):
        """Add node right after head (most recent)"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key):
        """Get value (mark as recently used)"""
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.value
    
    def put(self, key, value):
        """Put key-value pair"""
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = Node(key, value)
        self.cache[key] = node
        self._add_to_head(node)
        
        if len(self.cache) > self.capacity:
            # Remove LRU (node before tail)
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

# Time: O(1) for get and put
```

## Summary

**Backtracking:**
- Systematic way to explore all possibilities
- Key: choose, explore, unchoose (backtrack)
- Common patterns: permutations, combinations, subsets
- Optimization: pruning invalid branches early

**Advanced Topics:**
- **Bit Manipulation**: Efficient operations on binary representations
- **String Matching**: KMP, Rabin-Karp for pattern matching
- **Tries**: Efficient prefix-based operations
- **Caching**: LRU for managing frequently accessed data

**Practice Tips:**
- Draw recursion trees for backtracking
- Identify when to backtrack (constraints violated)
- Learn bit manipulation tricks
- Understand when tries are useful (autocomplete, spell check)
- Master template patterns for each technique

