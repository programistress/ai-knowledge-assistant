# Heaps and Priority Queues

## Introduction

A heap is a specialized tree-based data structure that satisfies the heap property. It's commonly used to implement priority queues.

## Types of Heaps

### Min Heap
In a min heap, for any given node, the node's value is less than or equal to the values of its children. The minimum element is at the root.

### Max Heap
In a max heap, for any given node, the node's value is greater than or equal to the values of its children. The maximum element is at the root.

## Heap Properties

1. **Complete Binary Tree**: All levels are filled except possibly the last, which is filled from left to right
2. **Heap Property**: Parent-child relationship follows min-heap or max-heap property
3. **Array Representation**: Can be efficiently represented as an array

### Array Representation

For a node at index `i`:
- Parent: `(i - 1) // 2`
- Left child: `2 * i + 1`
- Right child: `2 * i + 2`

```
Array: [1, 3, 6, 5, 9, 8]
Tree representation:
       1
      / \
     3   6
    / \  /
   5  9 8
```

## Min Heap Implementation

```python
class MinHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, val):
        """Insert value into heap - O(log n)"""
        # Add to end
        self.heap.append(val)
        # Bubble up
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        """Move element up to maintain heap property"""
        parent = self.parent(i)
        
        # If not root and violates heap property
        if i > 0 and self.heap[i] < self.heap[parent]:
            self.swap(i, parent)
            self._heapify_up(parent)
    
    def extract_min(self):
        """Remove and return minimum element - O(log n)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        # Save minimum
        min_val = self.heap[0]
        
        # Move last element to root
        self.heap[0] = self.heap.pop()
        
        # Bubble down
        self._heapify_down(0)
        
        return min_val
    
    def _heapify_down(self, i):
        """Move element down to maintain heap property"""
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        # Find smallest among node and its children
        if left < len(self.heap) and self.heap[left] < self.heap[min_index]:
            min_index = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[min_index]:
            min_index = right
        
        # If smallest is not the node itself
        if min_index != i:
            self.swap(i, min_index)
            self._heapify_down(min_index)
    
    def peek(self):
        """Return minimum without removing - O(1)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0

# Time Complexity:
# - Insert: O(log n)
# - Extract min/max: O(log n)
# - Peek: O(1)
# - Build heap: O(n)
# Space: O(n)
```

## Max Heap Implementation

```python
class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, val):
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        parent = self.parent(i)
        
        if i > 0 and self.heap[i] > self.heap[parent]:
            self.swap(i, parent)
            self._heapify_up(parent)
    
    def extract_max(self):
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return max_val
    
    def _heapify_down(self, i):
        max_index = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] > self.heap[max_index]:
            max_index = left
        
        if right < len(self.heap) and self.heap[right] > self.heap[max_index]:
            max_index = right
        
        if max_index != i:
            self.swap(i, max_index)
            self._heapify_down(max_index)
    
    def peek(self):
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
```

## Building a Heap (Heapify)

Building a heap from an array can be done in O(n) time.

```python
def heapify(arr):
    """
    Convert array to min heap in-place
    """
    n = len(arr)
    
    # Start from last non-leaf node and heapify down
    for i in range(n // 2 - 1, -1, -1):
        _heapify_down(arr, n, i)
    
    return arr

def _heapify_down(arr, n, i):
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] < arr[smallest]:
        smallest = left
    
    if right < n and arr[right] < arr[smallest]:
        smallest = right
    
    if smallest != i:
        arr[i], arr[smallest] = arr[smallest], arr[i]
        _heapify_down(arr, n, smallest)

# Time: O(n), Space: O(log n) for recursion
```

**Why O(n) and not O(n log n)?**
- Most nodes are near the bottom (leaves need no work)
- Mathematical analysis shows sum of work is O(n)

## Python's heapq Module

Python provides a built-in heap implementation (min heap by default).

```python
import heapq

# Create empty heap
heap = []

# Insert elements
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
# heap = [1, 3, 4]

# Pop minimum
min_val = heapq.heappop(heap)  # Returns 1

# Peek minimum
if heap:
    min_val = heap[0]

# Convert list to heap in-place
arr = [3, 1, 4, 1, 5, 9]
heapq.heapify(arr)  # O(n)

# Push and pop in one operation
heapq.heappushpop(heap, 2)  # Push 2, then pop and return smallest

# Pop and push in one operation
heapq.heapreplace(heap, 2)  # Pop smallest, then push 2

# N largest elements
heapq.nlargest(3, [1, 4, 2, 8, 5])  # [8, 5, 4]

# N smallest elements
heapq.nsmallest(3, [1, 4, 2, 8, 5])  # [1, 2, 4]

# For max heap, negate values
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
max_val = -heapq.heappop(max_heap)  # Returns 3
```

## Common Heap Problems

### Problem 1: Kth Largest Element

```python
import heapq

def find_kth_largest(nums, k):
    """
    Find kth largest element in array
    """
    # Method 1: Min heap of size k
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]

# Alternative: use heapq.nlargest
def find_kth_largest_v2(nums, k):
    return heapq.nlargest(k, nums)[-1]

# Time: O(n log k), Space: O(k)
```

### Problem 2: Merge K Sorted Lists

```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_sorted_lists(lists):
    """
    Merge k sorted linked lists
    """
    heap = []
    
    # Add first node from each list
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        # Add next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next

# Time: O(N log k) where N is total nodes, k is number of lists
# Space: O(k)
```

### Problem 3: Top K Frequent Elements

```python
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    """
    Find k most frequent elements
    """
    # Count frequencies
    count = Counter(nums)
    
    # Method 1: Use heapq.nlargest
    return heapq.nlargest(k, count.keys(), key=count.get)
    
    # Method 2: Min heap of size k
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]

# Time: O(n log k), Space: O(n)
```

### Problem 4: Find Median from Data Stream

```python
import heapq

class MedianFinder:
    """
    Find median in a stream of numbers
    Uses two heaps: max heap for smaller half, min heap for larger half
    """
    def __init__(self):
        self.small = []  # Max heap (use negative values)
        self.large = []  # Min heap
    
    def addNum(self, num):
        """Add number to data structure"""
        # Add to appropriate heap
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        
        # Balance heaps (size difference at most 1)
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        """Return median of all numbers"""
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0

# addNum: O(log n), findMedian: O(1)
```

### Problem 5: Sliding Window Median

```python
import heapq
from collections import defaultdict

def median_sliding_window(nums, k):
    """
    Find median of each sliding window of size k
    """
    def get_median():
        if k % 2 == 0:
            return (small[0][0] + large[0][0]) / 2.0
        return -small[0][0]
    
    def balance():
        # Balance size difference
        while len(small) > len(large) + 1:
            heapq.heappush(large, heapq.heappop(small))
        while len(large) > len(small):
            heapq.heappush(small, heapq.heappop(large))
    
    small = []  # Max heap (negative values)
    large = []  # Min heap
    result = []
    
    for i, num in enumerate(nums):
        # Add number
        if not small or num <= -small[0]:
            heapq.heappush(small, -num)
        else:
            heapq.heappush(large, num)
        
        balance()
        
        # Remove element leaving window
        if i >= k:
            out = nums[i - k]
            if out <= -small[0]:
                small.remove(-out)
                heapq.heapify(small)
            else:
                large.remove(out)
                heapq.heapify(large)
            
            balance()
        
        # Add median
        if i >= k - 1:
            result.append(get_median())
    
    return result
```

### Problem 6: Reorganize String

```python
import heapq
from collections import Counter

def reorganize_string(s):
    """
    Rearrange string so no two adjacent characters are same
    """
    # Count frequencies
    count = Counter(s)
    
    # Max heap (use negative counts)
    heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(heap)
    
    result = []
    prev_freq, prev_char = 0, ''
    
    while heap:
        freq, char = heapq.heappop(heap)
        result.append(char)
        
        # Add back previous character if still has occurrences
        if prev_freq < 0:
            heapq.heappush(heap, (prev_freq, prev_char))
        
        # Update previous
        prev_freq, prev_char = freq + 1, char
    
    result_str = ''.join(result)
    return result_str if len(result_str) == len(s) else ""

# Time: O(n log k) where k is unique characters
```

### Problem 7: Kth Smallest Element in Sorted Matrix

```python
import heapq

def kth_smallest(matrix, k):
    """
    Find kth smallest element in row and column sorted matrix
    """
    n = len(matrix)
    heap = []
    
    # Add first element from each row
    for r in range(min(n, k)):
        heapq.heappush(heap, (matrix[r][0], r, 0))
    
    result = 0
    for _ in range(k):
        result, r, c = heapq.heappop(heap)
        
        # Add next element from same row
        if c + 1 < n:
            heapq.heappush(heap, (matrix[r][c + 1], r, c + 1))
    
    return result

# Time: O(k log min(n, k)), Space: O(min(n, k))
```

### Problem 8: Task Scheduler

```python
import heapq
from collections import Counter

def least_interval(tasks, n):
    """
    Find minimum intervals needed to complete all tasks with cooling period n
    """
    # Count task frequencies
    freq = Counter(tasks)
    
    # Max heap of frequencies
    heap = [-f for f in freq.values()]
    heapq.heapify(heap)
    
    time = 0
    
    while heap:
        cycle = []
        
        # Process tasks in one cooling cycle
        for _ in range(n + 1):
            if heap:
                freq = heapq.heappop(heap)
                if freq + 1 < 0:
                    cycle.append(freq + 1)
        
        # Add back tasks that still have occurrences
        for freq in cycle:
            heapq.heappush(heap, freq)
        
        # Add time for this cycle
        time += (n + 1) if heap else len(cycle)
    
    return time

# Time: O(n), Space: O(1) - at most 26 unique tasks
```

## Priority Queue

A priority queue is an abstract data type where each element has a priority. Elements with higher priority are served before elements with lower priority.

### Implementation Using Heap

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0  # For tie-breaking
    
    def push(self, item, priority):
        """Add item with priority (lower number = higher priority)"""
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1
    
    def pop(self):
        """Remove and return item with highest priority"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return heapq.heappop(self.heap)[2]
    
    def peek(self):
        """Return item with highest priority without removing"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return self.heap[0][2]
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def size(self):
        return len(self.heap)

# Usage
pq = PriorityQueue()
pq.push("task1", 3)
pq.push("task2", 1)  # Higher priority
pq.push("task3", 2)

print(pq.pop())  # "task2" (priority 1)
print(pq.pop())  # "task3" (priority 2)
```

### Python's queue.PriorityQueue

```python
from queue import PriorityQueue

pq = PriorityQueue()

# Put items (priority, item)
pq.put((1, "high priority"))
pq.put((3, "low priority"))
pq.put((2, "medium priority"))

# Get items in priority order
while not pq.empty():
    priority, item = pq.get()
    print(item)

# Thread-safe implementation
```

## Heap Sort

Sort an array using heap data structure.

```python
def heap_sort(arr):
    """
    Sort array using heap sort
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

# Time: O(n log n)
# Space: O(1) - in-place sorting
# Not stable
```

## Heap Applications

1. **Priority Queues**: Task scheduling, event simulation
2. **Heap Sort**: O(n log n) sorting
3. **K-way Merge**: Merge k sorted arrays/lists
4. **Order Statistics**: Kth largest/smallest element
5. **Graph Algorithms**: Dijkstra's, Prim's MST
6. **Median Maintenance**: Running median in stream
7. **Top K Problems**: K most/least frequent elements

## Heap Patterns

### Pattern 1: Top/Bottom K Elements
Use min heap of size k for top k largest, max heap for top k smallest.

### Pattern 2: Two Heaps
Use max heap for smaller half, min heap for larger half (median finding).

### Pattern 3: K-way Merge
Use heap to track smallest element across k sorted sequences.

### Pattern 4: Scheduling
Use heap to prioritize tasks by deadline, priority, or other criteria.

## Time and Space Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Insert | O(log n) | O(1) |
| Extract min/max | O(log n) | O(1) |
| Peek | O(1) | O(1) |
| Build heap | O(n) | O(1) |
| Heap sort | O(n log n) | O(1) |
| Search | O(n) | O(1) |

## Summary

**Key Concepts:**
- Complete binary tree with heap property
- Efficient priority queue implementation
- Array-based representation
- O(log n) insert and extract
- O(n) heap construction

**When to Use:**
- Need to repeatedly access min/max element
- Priority-based processing
- K-largest/smallest problems
- Median finding in stream
- Merge k sorted sequences

**Common Mistakes:**
- Forgetting heap is not fully sorted (only root is min/max)
- Not maintaining heap size in top-k problems
- Using wrong heap type (min vs max)
- Inefficient search (heaps aren't optimized for search)

**Python Tips:**
- Use `heapq` module (min heap by default)
- Negate values for max heap
- Use tuples for priority and tie-breaking
- Remember heapq operates on lists in-place

