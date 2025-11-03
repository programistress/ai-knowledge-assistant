# Binary Search Trees and Balanced Trees

## Binary Search Tree (BST)

### Definition

A Binary Search Tree is a binary tree where for each node:
- All values in the left subtree are **less than** the node's value
- All values in the right subtree are **greater than** the node's value
- Both left and right subtrees are also BSTs

### BST Properties

- Inorder traversal gives sorted sequence
- Search, insertion, and deletion can be O(log n) in balanced trees
- Worst case O(n) when tree becomes skewed

### BST Implementation

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        """Insert value into BST"""
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        elif val > node.val:
            node.right = self._insert_recursive(node.right, val)
        # If equal, we don't insert (no duplicates)
        
        return node
    
    def search(self, val):
        """Search for value in BST"""
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_recursive(node.left, val)
        else:
            return self._search_recursive(node.right, val)
    
    def delete(self, val):
        """Delete value from BST"""
        self.root = self._delete_recursive(self.root, val)
    
    def _delete_recursive(self, node, val):
        if not node:
            return None
        
        if val < node.val:
            node.left = self._delete_recursive(node.left, val)
        elif val > node.val:
            node.right = self._delete_recursive(node.right, val)
        else:
            # Node to delete found
            
            # Case 1: No children (leaf)
            if not node.left and not node.right:
                return None
            
            # Case 2: One child
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            
            # Case 3: Two children
            # Find inorder successor (min in right subtree)
            successor = self._find_min(node.right)
            node.val = successor.val
            node.right = self._delete_recursive(node.right, successor.val)
        
        return node
    
    def _find_min(self, node):
        """Find minimum value node"""
        while node.left:
            node = node.left
        return node
    
    def _find_max(self, node):
        """Find maximum value node"""
        while node.right:
            node = node.right
        return node

# Time Complexity:
# - Search: O(h) where h is height
# - Insert: O(h)
# - Delete: O(h)
# Best case: O(log n) for balanced tree
# Worst case: O(n) for skewed tree
```

### BST Common Problems

**Problem 1: Validate BST**
```python
def is_valid_bst(root):
    """
    Check if binary tree is valid BST
    """
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))

# Time: O(n), Space: O(h)
```

**Problem 2: Kth Smallest Element**
```python
def kth_smallest(root, k):
    """
    Find kth smallest element in BST (1-indexed)
    """
    def inorder(node):
        if not node:
            return None
        
        # Search left
        left_result = inorder(node.left)
        if left_result is not None:
            return left_result
        
        # Process current
        self.count += 1
        if self.count == k:
            return node.val
        
        # Search right
        return inorder(node.right)
    
    self.count = 0
    return inorder(root)

# Time: O(n), Space: O(h)
```

**Problem 3: Lowest Common Ancestor in BST**
```python
def lca_bst(root, p, q):
    """
    Find LCA in BST (more efficient than general binary tree)
    """
    while root:
        # Both in left subtree
        if p.val < root.val and q.val < root.val:
            root = root.left
        # Both in right subtree
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            # Split point or one of them is root
            return root
    
    return None

# Time: O(h), Space: O(1)
```

**Problem 4: Convert Sorted Array to BST**
```python
def sorted_array_to_bst(nums):
    """
    Create balanced BST from sorted array
    """
    def build(left, right):
        if left > right:
            return None
        
        # Choose middle as root for balance
        mid = (left + right) // 2
        node = TreeNode(nums[mid])
        
        node.left = build(left, mid - 1)
        node.right = build(mid + 1, right)
        
        return node
    
    return build(0, len(nums) - 1)

# Time: O(n), Space: O(log n)
```

**Problem 5: BST Iterator**
```python
class BSTIterator:
    """
    Implement iterator over BST (inorder traversal)
    """
    def __init__(self, root):
        self.stack = []
        self._push_left(root)
    
    def _push_left(self, node):
        """Push all left nodes to stack"""
        while node:
            self.stack.append(node)
            node = node.left
    
    def next(self):
        """Return next smallest element"""
        node = self.stack.pop()
        self._push_left(node.right)
        return node.val
    
    def hasNext(self):
        """Check if next element exists"""
        return len(self.stack) > 0

# next() and hasNext(): O(1) amortized
# Space: O(h)
```

**Problem 6: Delete Node in BST**
```python
def delete_node(root, key):
    """
    Delete node with given key from BST
    """
    if not root:
        return None
    
    if key < root.val:
        root.left = delete_node(root.left, key)
    elif key > root.val:
        root.right = delete_node(root.right, key)
    else:
        # Node found
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        
        # Two children: find inorder successor
        min_node = root.right
        while min_node.left:
            min_node = min_node.left
        
        root.val = min_node.val
        root.right = delete_node(root.right, min_node.val)
    
    return root
```

## AVL Trees (Self-Balancing BST)

### Introduction

AVL trees are self-balancing BSTs where the height difference between left and right subtrees (balance factor) is at most 1 for every node.

**Balance Factor** = height(left subtree) - height(right subtree)
Must be in {-1, 0, 1}

### AVL Tree Operations

```python
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1  # Height of node

class AVLTree:
    def get_height(self, node):
        if not node:
            return 0
        return node.height
    
    def get_balance(self, node):
        """Get balance factor"""
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)
    
    def update_height(self, node):
        """Update height of node"""
        if node:
            node.height = 1 + max(self.get_height(node.left),
                                  self.get_height(node.right))
    
    def rotate_right(self, y):
        """
        Right rotation
             y                x
            / \              / \
           x   C    -->     A   y
          / \                  / \
         A   B                B   C
        """
        x = y.left
        B = x.right
        
        # Perform rotation
        x.right = y
        y.left = B
        
        # Update heights
        self.update_height(y)
        self.update_height(x)
        
        return x
    
    def rotate_left(self, x):
        """
        Left rotation
           x                  y
          / \                / \
         A   y      -->     x   C
            / \            / \
           B   C          A   B
        """
        y = x.right
        B = y.left
        
        # Perform rotation
        y.left = x
        x.right = B
        
        # Update heights
        self.update_height(x)
        self.update_height(y)
        
        return y
    
    def insert(self, node, val):
        """Insert value and rebalance"""
        # Standard BST insertion
        if not node:
            return AVLNode(val)
        
        if val < node.val:
            node.left = self.insert(node.left, val)
        else:
            node.right = self.insert(node.right, val)
        
        # Update height
        self.update_height(node)
        
        # Get balance factor
        balance = self.get_balance(node)
        
        # Left-Left Case
        if balance > 1 and val < node.left.val:
            return self.rotate_right(node)
        
        # Right-Right Case
        if balance < -1 and val > node.right.val:
            return self.rotate_left(node)
        
        # Left-Right Case
        if balance > 1 and val > node.left.val:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        
        # Right-Left Case
        if balance < -1 and val < node.right.val:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        
        return node
    
    def delete(self, node, val):
        """Delete value and rebalance"""
        # Standard BST deletion
        if not node:
            return None
        
        if val < node.val:
            node.left = self.delete(node.left, val)
        elif val > node.val:
            node.right = self.delete(node.right, val)
        else:
            # Node found
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                # Two children: get inorder successor
                successor = self._min_value_node(node.right)
                node.val = successor.val
                node.right = self.delete(node.right, successor.val)
        
        # Update height
        self.update_height(node)
        
        # Rebalance
        balance = self.get_balance(node)
        
        # Left-Left
        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.rotate_right(node)
        
        # Left-Right
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        
        # Right-Right
        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.rotate_left(node)
        
        # Right-Left
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        
        return node
    
    def _min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current

# Time Complexity: O(log n) for all operations
# Space Complexity: O(1) extra space, O(log n) for recursion stack
```

### Four Rotation Cases

1. **Left-Left (LL)**: Single right rotation
2. **Right-Right (RR)**: Single left rotation
3. **Left-Right (LR)**: Left rotation on left child, then right rotation
4. **Right-Left (RL)**: Right rotation on right child, then left rotation

## Red-Black Trees

### Properties

A Red-Black Tree is a self-balancing BST where:
1. Every node is either red or black
2. Root is always black
3. All leaves (NULL) are black
4. Red node cannot have red parent or red child
5. Every path from root to leaf has same number of black nodes

### Advantages
- Guaranteed O(log n) for search, insert, delete
- Less strict balancing than AVL (fewer rotations)
- Used in many language libraries (C++ std::map, Java TreeMap)

```python
class RBNode:
    def __init__(self, val):
        self.val = val
        self.color = "RED"  # New nodes are red
        self.left = None
        self.right = None
        self.parent = None

class RedBlackTree:
    def __init__(self):
        self.NIL = RBNode(0)
        self.NIL.color = "BLACK"
        self.root = self.NIL
    
    def rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
    
    def rotate_right(self, y):
        x = y.left
        y.left = x.right
        if x.right != self.NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent == None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x
    
    def insert_fixup(self, k):
        """Fix Red-Black properties after insertion"""
        while k.parent.color == "RED":
            if k.parent == k.parent.parent.left:
                u = k.parent.parent.right  # Uncle
                if u.color == "RED":
                    # Case 1: Uncle is red
                    k.parent.color = "BLACK"
                    u.color = "BLACK"
                    k.parent.parent.color = "RED"
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        # Case 2: k is right child
                        k = k.parent
                        self.rotate_left(k)
                    # Case 3: k is left child
                    k.parent.color = "BLACK"
                    k.parent.parent.color = "RED"
                    self.rotate_right(k.parent.parent)
            else:
                # Mirror cases
                u = k.parent.parent.left
                if u.color == "RED":
                    k.parent.color = "BLACK"
                    u.color = "BLACK"
                    k.parent.parent.color = "RED"
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.rotate_right(k)
                    k.parent.color = "BLACK"
                    k.parent.parent.color = "RED"
                    self.rotate_left(k.parent.parent)
            
            if k == self.root:
                break
        
        self.root.color = "BLACK"
    
    def insert(self, val):
        node = RBNode(val)
        node.left = self.NIL
        node.right = self.NIL
        
        parent = None
        current = self.root
        
        # Find position to insert
        while current != self.NIL:
            parent = current
            if node.val < current.val:
                current = current.left
            else:
                current = current.right
        
        node.parent = parent
        if parent == None:
            self.root = node
        elif node.val < parent.val:
            parent.left = node
        else:
            parent.right = node
        
        if node.parent == None:
            node.color = "BLACK"
            return
        
        if node.parent.parent == None:
            return
        
        self.insert_fixup(node)

# Time: O(log n) for all operations
```

## Comparison: BST vs AVL vs Red-Black

| Feature | BST | AVL | Red-Black |
|---------|-----|-----|-----------|
| Balance | No | Strict | Relaxed |
| Height | O(n) worst | O(log n) | O(log n) |
| Search | O(n) worst | O(log n) | O(log n) |
| Insert | O(n) worst | O(log n) | O(log n) |
| Delete | O(n) worst | O(log n) | O(log n) |
| Rotations (insert) | 0 | ≤ 2 | ≤ 2 |
| Rotations (delete) | 0 | O(log n) | ≤ 3 |
| Use case | Simple | Search-heavy | Balanced ops |

**When to use:**
- **BST**: Educational purposes, simple cases
- **AVL**: More searches than insertions/deletions
- **Red-Black**: Balanced read/write operations (most libraries)

## Segment Trees

### Introduction

Segment trees are used for range queries and updates on arrays efficiently.

**Use cases:**
- Range sum queries
- Range min/max queries
- Range updates

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        if arr:
            self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        """Build segment tree"""
        if start == end:
            self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self.build(arr, left_child, start, mid)
        self.build(arr, right_child, mid + 1, end)
        
        self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def update(self, index, value):
        """Update value at index"""
        self._update(0, 0, self.n - 1, index, value)
    
    def _update(self, node, start, end, index, value):
        if start == end:
            self.tree[node] = value
            return
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        if index <= mid:
            self._update(left_child, start, mid, index, value)
        else:
            self._update(right_child, mid + 1, end, index, value)
        
        self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def query(self, left, right):
        """Query sum in range [left, right]"""
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        # No overlap
        if right < start or left > end:
            return 0
        
        # Complete overlap
        if left <= start and end <= right:
            return self.tree[node]
        
        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_sum = self._query(left_child, start, mid, left, right)
        right_sum = self._query(right_child, mid + 1, end, left, right)
        
        return left_sum + right_sum

# Time: O(log n) for query and update
# Space: O(n)
```

## Fenwick Tree (Binary Indexed Tree)

More space-efficient than segment tree for certain operations.

```python
class FenwickTree:
    """
    Binary Indexed Tree for prefix sum queries
    """
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, index, delta):
        """Add delta to element at index (1-indexed)"""
        while index <= self.n:
            self.tree[index] += delta
            index += index & (-index)  # Add LSB
    
    def query(self, index):
        """Get prefix sum up to index (1-indexed)"""
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & (-index)  # Remove LSB
        return result
    
    def range_query(self, left, right):
        """Get sum in range [left, right] (1-indexed)"""
        return self.query(right) - self.query(left - 1)

# Time: O(log n) for update and query
# Space: O(n)
```

## Summary

**BST Basics:**
- Efficient search, insert, delete in O(log n) when balanced
- Can degenerate to O(n) if not balanced
- Inorder traversal gives sorted sequence

**Balanced Trees:**
- **AVL**: Strict balancing, better for search-heavy workloads
- **Red-Black**: Relaxed balancing, better for insert/delete-heavy workloads
- Both guarantee O(log n) height

**Advanced Trees:**
- **Segment Tree**: Range queries and updates
- **Fenwick Tree**: Space-efficient prefix sums
- **Trie**: String operations (covered separately)

**Key Patterns:**
- Validate BST with min/max bounds
- Kth element with inorder traversal
- LCA leverages BST property
- Iterator with stack for controlled traversal

**Practice Tips:**
- Understand BST property thoroughly
- Master rotations for balanced trees
- Practice recursive thinking
- Draw trees to visualize operations

