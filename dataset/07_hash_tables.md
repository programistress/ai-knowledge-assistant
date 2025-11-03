# Hash Tables, Hash Maps, and Sets

## Introduction

A hash table (also called hash map or dictionary) is a data structure that implements an associative array, mapping keys to values. It uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found.

## Core Concepts

### Hash Function

A hash function takes a key and computes an integer (hash code) that determines where the value should be stored.

**Properties of a good hash function:**
1. **Deterministic**: Same key always produces same hash
2. **Uniform distribution**: Spreads keys evenly across buckets
3. **Fast to compute**: O(1) operation
4. **Minimizes collisions**: Different keys produce different hashes

```python
# Simple hash function example
def hash_function(key, table_size):
    """Convert key to index in hash table"""
    # For integers
    if isinstance(key, int):
        return key % table_size
    
    # For strings (polynomial rolling hash)
    hash_value = 0
    for i, char in enumerate(key):
        hash_value += ord(char) * (31 ** i)
    return hash_value % table_size

# Python's built-in hash()
hash("hello")  # Returns hash code
hash(42)
hash((1, 2, 3))  # Tuples are hashable
# hash([1, 2, 3])  # Error: lists are not hashable!
```

### Collision Resolution

When two keys hash to the same index, we have a collision.

#### 1. Chaining (Separate Chaining)

Each bucket contains a linked list of entries that hash to the same index.

```python
class HashTableChaining:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
        self.count = 0
    
    def _hash(self, key):
        """Compute hash for key"""
        return hash(key) % self.size
    
    def put(self, key, value):
        """Insert or update key-value pair"""
        index = self._hash(key)
        bucket = self.table[index]
        
        # Update if key exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new entry
        bucket.append((key, value))
        self.count += 1
        
        # Resize if load factor > 0.75
        if self.count / self.size > 0.75:
            self._resize()
    
    def get(self, key):
        """Retrieve value for key"""
        index = self._hash(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(key)
    
    def remove(self, key):
        """Remove key-value pair"""
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.count -= 1
                return
        
        raise KeyError(key)
    
    def _resize(self):
        """Double the table size and rehash all entries"""
        old_table = self.table
        self.size *= 2
        self.table = [[] for _ in range(self.size)]
        self.count = 0
        
        for bucket in old_table:
            for key, value in bucket:
                self.put(key, value)
    
    def __contains__(self, key):
        try:
            self.get(key)
            return True
        except KeyError:
            return False

# Time Complexity (average):
# - Insert: O(1)
# - Search: O(1)
# - Delete: O(1)
# Worst case: O(n) if all keys collide
```

#### 2. Open Addressing

All entries are stored in the array itself. When a collision occurs, probe for the next available slot.

**Linear Probing:**
```python
class HashTableLinearProbing:
    def __init__(self, size=10):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def _probe(self, index):
        """Linear probing: try next slot"""
        return (index + 1) % self.size
    
    def put(self, key, value):
        """Insert or update key-value pair"""
        if self.count / self.size > 0.7:
            self._resize()
        
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                # Update existing key
                self.values[index] = value
                return
            index = self._probe(index)
        
        # Insert new key
        self.keys[index] = key
        self.values[index] = value
        self.count += 1
    
    def get(self, key):
        """Retrieve value for key"""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = self._probe(index)
        
        raise KeyError(key)
    
    def remove(self, key):
        """Remove key-value pair (requires careful handling)"""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                # Mark as deleted (use special marker)
                self.keys[index] = self.values[index] = None
                self.count -= 1
                return
            index = self._probe(index)
        
        raise KeyError(key)
    
    def _resize(self):
        old_keys = self.keys
        old_values = self.values
        
        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        
        for key, value in zip(old_keys, old_values):
            if key is not None:
                self.put(key, value)
```

**Quadratic Probing:**
```python
def _quadratic_probe(self, index, i):
    """Quadratic probing: try index + i^2"""
    return (index + i * i) % self.size
```

**Double Hashing:**
```python
def _double_hash_probe(self, key, index, i):
    """Double hashing: use second hash function"""
    hash2 = 1 + (hash(key) % (self.size - 1))
    return (index + i * hash2) % self.size
```

## Python's Built-in Dictionary

Python's `dict` is a highly optimized hash table implementation.

```python
# Creating dictionaries
d = {}
d = dict()
d = {'name': 'Alice', 'age': 30}
d = dict(name='Alice', age=30)

# Operations
d['key'] = 'value'           # Insert/Update: O(1) average
value = d['key']             # Access: O(1) average
value = d.get('key', default) # Safe access with default
del d['key']                 # Delete: O(1) average
'key' in d                   # Membership test: O(1) average

# Iteration
for key in d:                # Iterate over keys
    print(key, d[key])

for value in d.values():     # Iterate over values
    print(value)

for key, value in d.items(): # Iterate over key-value pairs
    print(key, value)

# Dictionary methods
d.keys()                     # View of keys
d.values()                   # View of values
d.items()                    # View of (key, value) pairs
d.pop('key')                 # Remove and return value
d.popitem()                  # Remove and return arbitrary (key, value)
d.clear()                    # Remove all items
d.update(other_dict)         # Update with another dict
```

## Sets

A set is an unordered collection of unique elements, implemented using hash tables.

```python
# Creating sets
s = set()
s = {1, 2, 3, 4, 5}
s = set([1, 2, 2, 3, 3])     # {1, 2, 3} - duplicates removed

# Operations
s.add(6)                     # Add element: O(1)
s.remove(3)                  # Remove element: O(1), raises KeyError if not found
s.discard(3)                 # Remove element: O(1), no error if not found
3 in s                       # Membership test: O(1)
len(s)                       # Size: O(1)

# Set operations
s1 = {1, 2, 3}
s2 = {3, 4, 5}

s1.union(s2)                 # {1, 2, 3, 4, 5}
s1 | s2                      # Same as union

s1.intersection(s2)          # {3}
s1 & s2                      # Same as intersection

s1.difference(s2)            # {1, 2}
s1 - s2                      # Same as difference

s1.symmetric_difference(s2)  # {1, 2, 4, 5}
s1 ^ s2                      # Same as symmetric_difference

s1.issubset(s2)              # Check if s1 ⊆ s2
s1 <= s2                     # Same as issubset

s1.issuperset(s2)            # Check if s1 ⊇ s2
s1 >= s2                     # Same as issuperset

# Frozen set (immutable)
fs = frozenset([1, 2, 3])
# Can be used as dict key or in another set
```

## Common Hash Table Problems

### Problem 1: Two Sum

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

### Problem 2: Group Anagrams

```python
from collections import defaultdict

def group_anagrams(strs):
    """
    Group strings that are anagrams
    """
    anagrams = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        anagrams[key].append(s)
    
    return list(anagrams.values())

# Time: O(n * k log k), Space: O(n * k)
# where n = number of strings, k = max length
```

### Problem 3: First Unique Character

```python
from collections import Counter

def first_uniq_char(s):
    """
    Find index of first non-repeating character
    """
    # Count frequencies
    count = Counter(s)
    
    # Find first with count 1
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
    
    return -1

# Time: O(n), Space: O(1) (at most 26 letters)
```

### Problem 4: Longest Consecutive Sequence

```python
def longest_consecutive(nums):
    """
    Find length of longest consecutive sequence
    """
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        # Only start counting from sequence beginning
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            max_length = max(max_length, current_length)
    
    return max_length

# Example: [100, 4, 200, 1, 3, 2] → 4 (sequence [1,2,3,4])
# Time: O(n), Space: O(n)
```

### Problem 5: Subarray Sum Equals K

```python
from collections import defaultdict

def subarray_sum(nums, k):
    """
    Count number of subarrays with sum equal to k
    """
    count = 0
    cumsum = 0
    sum_freq = defaultdict(int)
    sum_freq[0] = 1  # Empty prefix
    
    for num in nums:
        cumsum += num
        
        # If cumsum - k exists, we found subarray(s) with sum k
        if cumsum - k in sum_freq:
            count += sum_freq[cumsum - k]
        
        sum_freq[cumsum] += 1
    
    return count

# Time: O(n), Space: O(n)
```

### Problem 6: Top K Frequent Elements

```python
from collections import Counter
import heapq

def top_k_frequent(nums, k):
    """
    Find k most frequent elements
    """
    # Method 1: Using Counter
    count = Counter(nums)
    return [num for num, _ in count.most_common(k)]
    
    # Method 2: Using heap
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
    
    # Method 3: Bucket sort (O(n))
    count = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in count.items():
        buckets[freq].append(num)
    
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
    
    return result

# Time: O(n) for bucket sort, O(n log k) for heap
```

### Problem 7: Valid Sudoku

```python
def is_valid_sudoku(board):
    """
    Check if 9x9 Sudoku board is valid
    """
    seen = set()
    
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num != '.':
                # Create unique identifiers for row, column, and box
                row_key = f"{num} in row {i}"
                col_key = f"{num} in col {j}"
                box_key = f"{num} in box {i//3},{j//3}"
                
                if row_key in seen or col_key in seen or box_key in seen:
                    return False
                
                seen.add(row_key)
                seen.add(col_key)
                seen.add(box_key)
    
    return True

# Time: O(1) since board size is fixed, Space: O(1)
```

### Problem 8: LRU Cache

```python
from collections import OrderedDict

class LRUCache:
    """
    Least Recently Used Cache with O(1) operations
    """
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        """Get value and mark as recently used"""
        if key not in self.cache:
            return -1
        
        # Move to end (most recent)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        """Put key-value pair, evict LRU if needed"""
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # Evict least recently used (first item)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Time: O(1) for both get and put
# Space: O(capacity)
```

**Manual Implementation with Doubly Linked List:**
```python
class Node:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> Node
        
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
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.value
    
    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = Node(key, value)
        self.cache[key] = node
        self._add_to_head(node)
        
        if len(self.cache) > self.capacity:
            # Remove least recently used (before tail)
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

### Problem 9: Contains Duplicate II

```python
def contains_nearby_duplicate(nums, k):
    """
    Check if there are two distinct indices i and j such that
    nums[i] == nums[j] and abs(i - j) <= k
    """
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        if num in seen and i - seen[num] <= k:
            return True
        seen[num] = i
    
    return False

# Time: O(n), Space: O(min(n, k))
```

### Problem 10: Design HashMap

```python
class MyHashMap:
    """
    Design a HashMap without using built-in libraries
    """
    def __init__(self):
        self.size = 1000
        self.table = [[] for _ in range(self.size)]
    
    def _hash(self, key):
        return key % self.size
    
    def put(self, key, value):
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        bucket.append((key, value))
    
    def get(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return -1
    
    def remove(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return
```

## Advanced: Bloom Filters

A probabilistic data structure for membership testing with very low memory usage.

```python
import hashlib

class BloomFilter:
    """
    Space-efficient probabilistic set membership tester
    Can have false positives but never false negatives
    """
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size
    
    def _hashes(self, item):
        """Generate multiple hash values"""
        hashes = []
        for i in range(self.num_hashes):
            # Use different hash functions
            h = hashlib.md5((str(item) + str(i)).encode())
            hashes.append(int(h.hexdigest(), 16) % self.size)
        return hashes
    
    def add(self, item):
        """Add item to filter"""
        for h in self._hashes(item):
            self.bit_array[h] = True
    
    def contains(self, item):
        """Check if item might be in set"""
        return all(self.bit_array[h] for h in self._hashes(item))

# Use cases:
# - Web crawlers (check if URL already visited)
# - Spell checkers
# - Database query optimization
# - Cache filtering
```

## Hash Table Patterns

### Pattern 1: Frequency Counting
```python
from collections import Counter

# Count frequencies
freq = Counter(arr)
most_common = freq.most_common(k)
```

### Pattern 2: Lookup/Cache
```python
# Store computed values
cache = {}
if key in cache:
    return cache[key]
cache[key] = compute(key)
```

### Pattern 3: Grouping
```python
from collections import defaultdict

# Group items by some property
groups = defaultdict(list)
for item in items:
    key = get_key(item)
    groups[key].append(item)
```

### Pattern 4: Index Mapping
```python
# Map values to indices
index_map = {val: i for i, val in enumerate(arr)}
```

### Pattern 5: Sliding Window with Hash
```python
# Track elements in current window
window = {}
for i in range(len(arr)):
    window[arr[i]] = i
    if i >= k:
        # Remove element leaving window
        if window[arr[i-k]] == i-k:
            del window[arr[i-k]]
```

## Time and Space Complexity

| Operation | Average | Worst Case |
|-----------|---------|------------|
| Search | O(1) | O(n) |
| Insert | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Space | O(n) | O(n) |

**Load Factor**: α = n / m (elements / buckets)
- Keep α < 0.75 for good performance
- Resize when threshold exceeded

## When to Use Hash Tables

✅ **Use when:**
- Need fast lookups (O(1) average)
- Counting frequencies
- Caching/memoization
- Detecting duplicates
- Grouping related items

❌ **Don't use when:**
- Need ordered data (use TreeMap/BST)
- Need range queries
- Keys are not hashable
- Memory is very constrained

## Summary

**Key Concepts:**
- Hash function maps keys to indices
- Collisions handled by chaining or open addressing
- Average O(1) operations with good hash function
- Load factor determines when to resize

**Python Built-ins:**
- `dict`: Hash map (key-value pairs)
- `set`: Hash set (unique elements)
- `Counter`: Frequency counting
- `defaultdict`: Dict with default factory
- `OrderedDict`: Dict that remembers insertion order

**Common Patterns:**
- Two Sum pattern (complement lookup)
- Frequency counting
- Grouping/bucketing
- Caching/memoization
- Set operations

**Remember:**
- Hash tables trade space for speed
- Only hashable objects can be keys (immutable)
- Perfect for O(1) lookups and membership tests

