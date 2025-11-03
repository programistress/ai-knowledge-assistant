# Problem-Solving Resources and Patterns

## Common Coding Interview Patterns

### Pattern Recognition Guide

Being able to recognize problem patterns quickly is crucial for coding interviews. Here's a comprehensive guide to the most common patterns.

## Pattern 1: Two Pointers

**When to Use:**
- Array/string traversal from both ends
- Finding pairs with specific properties
- Palindrome problems
- Removing duplicates in-place

**Example Problems:**
- Two Sum II (sorted array)
- 3Sum, 4Sum
- Container With Most Water
- Trapping Rain Water
- Remove Duplicates
- Valid Palindrome

**Template:**
```python
def two_pointer_template(arr):
    left, right = 0, len(arr) - 1
    
    while left < right:
        # Process current pointers
        if condition:
            # Do something
            left += 1
        else:
            right -= 1
    
    return result
```

## Pattern 2: Sliding Window

**When to Use:**
- Subarray/substring problems
- Finding optimal window satisfying conditions
- Problems with "consecutive" or "contiguous"

**Fixed Window:**
```python
def fixed_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

**Variable Window:**
```python
def variable_window(arr, target):
    left = 0
    window_sum = 0
    result = float('inf')
    
    for right in range(len(arr)):
        window_sum += arr[right]
        
        while window_sum >= target:
            result = min(result, right - left + 1)
            window_sum -= arr[left]
            left += 1
    
    return result
```

**Example Problems:**
- Maximum Sum Subarray of Size K
- Longest Substring Without Repeating Characters
- Minimum Window Substring
- Longest Substring with K Distinct Characters

## Pattern 3: Fast & Slow Pointers

**When to Use:**
- Cycle detection
- Finding middle element
- Finding nth from end

**Example Problems:**
- Linked List Cycle
- Find Middle of Linked List
- Happy Number
- Find Duplicate Number

**Template:**
```python
def fast_slow_pointers(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True  # Cycle detected
    
    return False
```

## Pattern 4: Merge Intervals

**When to Use:**
- Overlapping intervals
- Scheduling problems
- Time-based problems

**Example Problems:**
- Merge Intervals
- Insert Interval
- Meeting Rooms I/II
- Minimum Meeting Rooms

**Template:**
```python
def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:
            # Overlap: merge
            last[1] = max(last[1], current[1])
        else:
            # No overlap: add new interval
            merged.append(current)
    
    return merged
```

## Pattern 5: Cyclic Sort

**When to Use:**
- Array contains numbers in given range
- Finding missing/duplicate numbers
- Problems with range [1, n] or [0, n-1]

**Example Problems:**
- Find Missing Number
- Find All Missing Numbers
- Find Duplicate Number
- Find All Duplicates

**Template:**
```python
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        j = nums[i] - 1
        if nums[i] != nums[j]:
            nums[i], nums[j] = nums[j], nums[i]
        else:
            i += 1
    return nums
```

## Pattern 6: In-place Reversal of Linked List

**When to Use:**
- Reversing linked list
- Reversing part of linked list

**Example Problems:**
- Reverse Linked List
- Reverse Linked List II
- Reverse Nodes in k-Group

**Template:**
```python
def reverse_linked_list(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev
```

## Pattern 7: Tree BFS

**When to Use:**
- Level-order traversal
- Finding level/depth
- Problems requiring level-by-level processing

**Example Problems:**
- Binary Tree Level Order Traversal
- Zigzag Level Order Traversal
- Minimum Depth
- Level Averages

**Template:**
```python
from collections import deque

def tree_bfs(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

## Pattern 8: Tree DFS

**When to Use:**
- Path finding
- Subtree problems
- Pre/in/post-order traversal

**Example Problems:**
- Path Sum
- All Paths for a Sum
- Maximum Depth
- Diameter of Tree

**Template:**
```python
def tree_dfs(root):
    if not root:
        return 0
    
    left = tree_dfs(root.left)
    right = tree_dfs(root.right)
    
    # Process current node with children results
    return process(root.val, left, right)
```

## Pattern 9: Two Heaps

**When to Use:**
- Finding median
- Dividing data into two halves
- Optimization problems with two groups

**Example Problems:**
- Find Median from Data Stream
- Sliding Window Median
- IPO

**Template:**
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # Max heap (negative values)
        self.large = []  # Min heap
    
    def add_num(self, num):
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        
        # Balance heaps
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
```

## Pattern 10: Subsets

**When to Use:**
- Finding all combinations
- Power set problems
- Permutations/combinations

**Example Problems:**
- Subsets
- Subsets II
- Permutations
- Combinations

**Template:**
```python
def subsets(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

## Pattern 11: Modified Binary Search

**When to Use:**
- Rotated sorted arrays
- Finding peak element
- Search in infinite array
- Optimization problems (binary search on answer)

**Example Problems:**
- Search in Rotated Sorted Array
- Find Peak Element
- First/Last Position of Element
- Capacity to Ship Packages

## Pattern 12: Top K Elements

**When to Use:**
- Finding K largest/smallest
- Frequency-based problems

**Example Problems:**
- Top K Frequent Elements
- Kth Largest Element
- K Closest Points to Origin

**Template:**
```python
import heapq

def top_k_frequent(nums, k):
    from collections import Counter
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

## Pattern 13: K-way Merge

**When to Use:**
- Merging K sorted arrays/lists
- Problems with multiple sorted inputs

**Example Problems:**
- Merge K Sorted Lists
- Kth Smallest in M Sorted Arrays
- Smallest Range Covering K Lists

**Template:**
```python
import heapq

def merge_k_lists(lists):
    heap = []
    
    # Add first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next
```

## Pattern 14: Topological Sort

**When to Use:**
- Dependency resolution
- Course prerequisites
- Build systems

**Example Problems:**
- Course Schedule
- Course Schedule II
- Alien Dictionary

## Pattern 15: Union-Find

**When to Use:**
- Connectivity problems
- Cycle detection in undirected graphs
- Dynamic connectivity

**Example Problems:**
- Number of Connected Components
- Redundant Connection
- Accounts Merge

## Typical Edge Cases

### Array/String Edge Cases
- Empty array/string
- Single element
- Two elements
- All elements same
- Already sorted/reverse sorted
- Duplicates
- Negative numbers
- Integer overflow

### Linked List Edge Cases
- Empty list (null head)
- Single node
- Two nodes
- Cycle vs no cycle
- Finding middle (odd vs even length)

### Tree Edge Cases
- Empty tree (null root)
- Single node
- Only left/right subtree
- Balanced vs skewed
- Complete vs incomplete

### Graph Edge Cases
- No edges (disconnected)
- Self-loops
- Multiple edges between nodes
- Cycles

## Problem-Solving Framework

### Step 1: Understand the Problem
1. Read carefully, identify input/output
2. Ask clarifying questions
3. Work through examples
4. Identify constraints

### Step 2: Plan the Approach
1. Recognize pattern
2. Choose data structure
3. Identify algorithm
4. Estimate complexity

### Step 3: Implement
1. Start with brute force if needed
2. Write clean, modular code
3. Handle edge cases
4. Test with examples

### Step 4: Optimize
1. Analyze time/space complexity
2. Look for redundant operations
3. Consider tradeoffs
4. Can you do better?

### Step 5: Test
1. Test with provided examples
2. Test edge cases
3. Test large inputs
4. Test invalid inputs

## Time Complexity Cheat Sheet

| n | O(1) | O(log n) | O(n) | O(n log n) | O(n²) | O(2ⁿ) |
|---|------|----------|------|------------|-------|-------|
| 10 | 1 | 3 | 10 | 30 | 100 | 1,024 |
| 100 | 1 | 7 | 100 | 700 | 10,000 | ∞ |
| 1,000 | 1 | 10 | 1,000 | 10,000 | 1,000,000 | ∞ |
| 10,000 | 1 | 13 | 10,000 | 130,000 | 100,000,000 | ∞ |

**Guidelines:**
- n ≤ 10: O(n!) or O(2ⁿ)
- n ≤ 20: O(2ⁿ)
- n ≤ 500: O(n³)
- n ≤ 5,000: O(n²)
- n ≤ 1,000,000: O(n log n)
- n > 1,000,000: O(n) or O(log n)

## Data Structure Selection Guide

| Operation | Best Data Structure |
|-----------|---------------------|
| Fast search/insert/delete | Hash Table |
| Maintain order + fast operations | BST/AVL/Red-Black Tree |
| Range queries | Segment Tree/Fenwick Tree |
| FIFO | Queue |
| LIFO | Stack |
| Priority-based | Heap/Priority Queue |
| Prefix matching | Trie |
| Disjoint sets | Union-Find |
| Graph connectivity | Adjacency List |
| Dense graph | Adjacency Matrix |

## Common Mistakes to Avoid

### 1. Not Understanding the Problem
- Rushing to code
- Missing constraints
- Incorrect assumptions

### 2. Poor Edge Case Handling
- Not testing empty inputs
- Missing boundary conditions
- Integer overflow

### 3. Inefficient Solutions
- Not optimizing when possible
- Using wrong data structure
- Redundant computations

### 4. Code Quality Issues
- Poor variable names
- No comments
- Not modular
- Magic numbers

### 5. Time Management
- Spending too long on optimization
- Not moving on from stuck problems
- Not testing thoroughly

## LeetCode Problem Categories

### Easy (Fundamentals)
- Arrays and strings
- Hash tables
- Two pointers
- Basic recursion
- Simple math

### Medium (Patterns)
- Dynamic programming (basic)
- Binary search variations
- Tree traversals
- Graph BFS/DFS
- Backtracking
- Sliding window

### Hard (Advanced)
- Advanced DP
- Hard graph algorithms
- Segment trees
- Complex backtracking
- Multiple patterns combined

## Interview Tips

### Before the Interview
1. Review common patterns
2. Practice explaining approach
3. Practice coding on whiteboard/editor
4. Review time complexities

### During the Interview
1. **Think aloud** - Communicate your thought process
2. **Ask questions** - Clarify requirements
3. **Start simple** - Brute force then optimize
4. **Test thoroughly** - Walk through examples
5. **Handle feedback** - Be receptive to hints

### Communication Strategies
- Explain your approach before coding
- Walk through examples
- Discuss tradeoffs
- Explain time/space complexity
- Ask if interviewer wants you to optimize

## Practice Strategy

### 1. Learn Patterns
- Study each pattern thoroughly
- Solve 3-5 problems per pattern
- Understand when to apply each

### 2. Practice Regularly
- Consistency over marathon sessions
- Mix difficulty levels
- Time yourself

### 3. Review Solutions
- Study optimal solutions
- Understand different approaches
- Learn from mistakes

### 4. Mock Interviews
- Practice explaining solutions
- Get comfortable with pressure
- Receive feedback

## Resources

### Online Judges
- **LeetCode**: Most popular, interview-focused
- **HackerRank**: Good for beginners
- **CodeForces**: Competitive programming
- **AtCoder**: Japanese competitive programming
- **TopCoder**: Algorithmic challenges

### Books
- "Cracking the Coding Interview" by Gayle McDowell
- "Introduction to Algorithms" (CLRS)
- "Algorithm Design Manual" by Skiena
- "Elements of Programming Interviews"

### Courses
- MIT 6.006: Introduction to Algorithms
- Stanford Algorithms Specialization
- Princeton Algorithms on Coursera

## Summary

**Key Takeaways:**
- Pattern recognition is crucial
- Practice diverse problems
- Master fundamentals first
- Understand time/space complexity
- Test thoroughly
- Communicate clearly

**Success Formula:**
1. Understand patterns
2. Practice consistently
3. Review and learn from mistakes
4. Simulate interview conditions
5. Focus on communication

**Remember:**
- Interview is not just about getting answer
- Process and communication matter
- It's okay to not know everything
- Continuous learning is key
- Practice makes perfect!

Good luck with your coding interviews and algorithmic journey!

