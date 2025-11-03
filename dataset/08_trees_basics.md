# Trees - Basics and Binary Trees

## Introduction to Trees

A tree is a hierarchical data structure consisting of nodes connected by edges. It's a non-linear structure that represents hierarchical relationships.

### Tree Terminology

- **Node**: Basic unit containing data
- **Root**: Topmost node (has no parent)
- **Parent**: Node that has children
- **Child**: Node with a parent
- **Leaf**: Node with no children
- **Internal Node**: Node with at least one child
- **Edge**: Connection between two nodes
- **Path**: Sequence of nodes connected by edges
- **Height of Node**: Longest path from node to a leaf
- **Depth of Node**: Path length from root to node
- **Height of Tree**: Height of root node
- **Level**: Depth + 1 (root is at level 1)
- **Subtree**: Tree formed by a node and its descendants
- **Degree**: Number of children of a node
- **Siblings**: Nodes with the same parent
- **Ancestor**: Any node on path from root to given node
- **Descendant**: Any node in subtree rooted at given node

```
Example Tree:
        1          <- Root (height=2, depth=0, level=1)
       / \
      2   3        <- Internal nodes (depth=1, level=2)
     / \   \
    4   5   6      <- Leaves (depth=2, level=3, height=0)
```

## Binary Trees

A binary tree is a tree where each node has at most two children, referred to as left child and right child.

### Binary Tree Node

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Types of Binary Trees

#### 1. Full Binary Tree
Every node has either 0 or 2 children (no node has only 1 child).

```
       1
      / \
     2   3
    / \
   4   5
```

#### 2. Complete Binary Tree
All levels are filled except possibly the last, which is filled from left to right.

```
       1
      / \
     2   3
    / \  /
   4  5 6
```

**Used in**: Heaps, segment trees

#### 3. Perfect Binary Tree
All internal nodes have 2 children and all leaves are at the same level.

```
       1
      / \
     2   3
    / \ / \
   4  5 6  7
```

**Properties**:
- Total nodes = 2^(h+1) - 1 where h is height
- Leaf nodes = 2^h

#### 4. Balanced Binary Tree
Height difference between left and right subtrees is at most 1 for every node.

```
       1
      / \
     2   3
    / \
   4   5
```

#### 5. Degenerate (Skewed) Tree
Each parent has only one child (essentially a linked list).

```
1           1
 \         /
  2       2
   \     /
    3   3
```

## Tree Traversals

### Depth-First Search (DFS) Traversals

#### 1. Inorder Traversal (Left, Root, Right)

```python
def inorder_traversal(root):
    """
    Inorder: Left -> Root -> Right
    For BST, gives sorted order
    """
    result = []
    
    def traverse(node):
        if not node:
            return
        
        traverse(node.left)      # Left
        result.append(node.val)  # Root
        traverse(node.right)     # Right
    
    traverse(root)
    return result

# Iterative version
def inorder_iterative(root):
    result = []
    stack = []
    current = root
    
    while current or stack:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process node
        current = stack.pop()
        result.append(current.val)
        
        # Move to right subtree
        current = current.right
    
    return result

# Time: O(n), Space: O(h) where h is height
```

#### 2. Preorder Traversal (Root, Left, Right)

```python
def preorder_traversal(root):
    """
    Preorder: Root -> Left -> Right
    Used for creating a copy of tree
    """
    result = []
    
    def traverse(node):
        if not node:
            return
        
        result.append(node.val)  # Root
        traverse(node.left)      # Left
        traverse(node.right)     # Right
    
    traverse(root)
    return result

# Iterative version
def preorder_iterative(root):
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

# Time: O(n), Space: O(h)
```

#### 3. Postorder Traversal (Left, Right, Root)

```python
def postorder_traversal(root):
    """
    Postorder: Left -> Right -> Root
    Used for deleting tree, evaluating expressions
    """
    result = []
    
    def traverse(node):
        if not node:
            return
        
        traverse(node.left)      # Left
        traverse(node.right)     # Right
        result.append(node.val)  # Root
    
    traverse(root)
    return result

# Iterative version (more complex)
def postorder_iterative(root):
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push left first, then right
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
    # Reverse to get postorder
    return result[::-1]

# Time: O(n), Space: O(h)
```

### Breadth-First Search (BFS) / Level Order Traversal

```python
from collections import deque

def level_order(root):
    """
    Level-order: Visit nodes level by level
    """
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

# Time: O(n), Space: O(w) where w is max width
```

### Morris Traversal (Space-Optimized)

Inorder traversal with O(1) space using threading.

```python
def morris_inorder(root):
    """
    Morris Inorder Traversal - O(1) space
    """
    result = []
    current = root
    
    while current:
        if not current.left:
            # No left child, visit and go right
            result.append(current.val)
            current = current.right
        else:
            # Find inorder predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # Create thread
                predecessor.right = current
                current = current.left
            else:
                # Thread exists, visit and remove thread
                predecessor.right = None
                result.append(current.val)
                current = current.right
    
    return result

# Time: O(n), Space: O(1)
```

## Common Binary Tree Problems

### Problem 1: Maximum Depth

```python
def max_depth(root):
    """
    Find maximum depth (height) of binary tree
    """
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return max(left_depth, right_depth) + 1

# Iterative BFS
def max_depth_iterative(root):
    if not root:
        return 0
    
    queue = deque([root])
    depth = 0
    
    while queue:
        depth += 1
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return depth

# Time: O(n), Space: O(h)
```

### Problem 2: Check if Balanced

```python
def is_balanced(root):
    """
    Check if binary tree is height-balanced
    """
    def check_height(node):
        if not node:
            return 0
        
        left_height = check_height(node.left)
        if left_height == -1:
            return -1
        
        right_height = check_height(node.right)
        if right_height == -1:
            return -1
        
        # Check balance condition
        if abs(left_height - right_height) > 1:
            return -1
        
        return max(left_height, right_height) + 1
    
    return check_height(root) != -1

# Time: O(n), Space: O(h)
```

### Problem 3: Diameter of Binary Tree

```python
def diameter_of_binary_tree(root):
    """
    Find longest path between any two nodes
    """
    diameter = [0]
    
    def height(node):
        if not node:
            return 0
        
        left_height = height(node.left)
        right_height = height(node.right)
        
        # Update diameter (path through this node)
        diameter[0] = max(diameter[0], left_height + right_height)
        
        return max(left_height, right_height) + 1
    
    height(root)
    return diameter[0]

# Time: O(n), Space: O(h)
```

### Problem 4: Invert Binary Tree

```python
def invert_tree(root):
    """
    Mirror the binary tree
    """
    if not root:
        return None
    
    # Swap left and right children
    root.left, root.right = root.right, root.left
    
    # Recursively invert subtrees
    invert_tree(root.left)
    invert_tree(root.right)
    
    return root

# Iterative version
def invert_tree_iterative(root):
    if not root:
        return None
    
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        # Swap children
        node.left, node.right = node.right, node.left
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return root

# Time: O(n), Space: O(h)
```

### Problem 5: Same Tree

```python
def is_same_tree(p, q):
    """
    Check if two trees are identical
    """
    # Both null
    if not p and not q:
        return True
    
    # One is null
    if not p or not q:
        return False
    
    # Values different
    if p.val != q.val:
        return False
    
    # Check subtrees
    return (is_same_tree(p.left, q.left) and 
            is_same_tree(p.right, q.right))

# Time: O(n), Space: O(h)
```

### Problem 6: Symmetric Tree

```python
def is_symmetric(root):
    """
    Check if tree is mirror of itself
    """
    def is_mirror(left, right):
        if not left and not right:
            return True
        
        if not left or not right:
            return False
        
        return (left.val == right.val and
                is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))
    
    return is_mirror(root, root) if root else True

# Time: O(n), Space: O(h)
```

### Problem 7: Path Sum

```python
def has_path_sum(root, target_sum):
    """
    Check if root-to-leaf path with given sum exists
    """
    if not root:
        return False
    
    # Leaf node
    if not root.left and not root.right:
        return root.val == target_sum
    
    # Check left and right subtrees
    remaining = target_sum - root.val
    return (has_path_sum(root.left, remaining) or
            has_path_sum(root.right, remaining))

# Time: O(n), Space: O(h)
```

**Find All Paths:**
```python
def path_sum_all(root, target_sum):
    """
    Find all root-to-leaf paths with given sum
    """
    result = []
    
    def dfs(node, remaining, path):
        if not node:
            return
        
        path.append(node.val)
        
        # Leaf node with target sum
        if not node.left and not node.right and remaining == node.val:
            result.append(path[:])
        else:
            dfs(node.left, remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)
        
        path.pop()  # Backtrack
    
    dfs(root, target_sum, [])
    return result
```

### Problem 8: Lowest Common Ancestor

```python
def lowest_common_ancestor(root, p, q):
    """
    Find LCA of two nodes in binary tree
    """
    if not root or root == p or root == q:
        return root
    
    # Search in left and right subtrees
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    # If both found in different subtrees, root is LCA
    if left and right:
        return root
    
    # Return whichever is not None
    return left if left else right

# Time: O(n), Space: O(h)
```

### Problem 9: Serialize and Deserialize

```python
class Codec:
    """
    Serialize and deserialize binary tree
    """
    def serialize(self, root):
        """Encode tree to string"""
        def preorder(node):
            if not node:
                vals.append('#')
                return
            vals.append(str(node.val))
            preorder(node.left)
            preorder(node.right)
        
        vals = []
        preorder(root)
        return ','.join(vals)
    
    def deserialize(self, data):
        """Decode string to tree"""
        def build():
            val = next(vals)
            if val == '#':
                return None
            
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
        
        vals = iter(data.split(','))
        return build()

# Time: O(n) for both, Space: O(n)
```

### Problem 10: Binary Tree Right Side View

```python
def right_side_view(root):
    """
    Return values of nodes visible from right side
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Last node in level
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result

# Time: O(n), Space: O(w)
```

### Problem 11: Flatten Binary Tree to Linked List

```python
def flatten(root):
    """
    Flatten tree to linked list in-place (preorder)
    """
    if not root:
        return
    
    # Flatten subtrees
    flatten(root.left)
    flatten(root.right)
    
    # Save right subtree
    right = root.right
    
    # Move left subtree to right
    root.right = root.left
    root.left = None
    
    # Attach original right subtree
    current = root
    while current.right:
        current = current.right
    current.right = right

# Time: O(n), Space: O(h)
```

### Problem 12: Count Complete Tree Nodes

```python
def count_nodes(root):
    """
    Count nodes in complete binary tree
    Better than O(n) by exploiting completeness
    """
    if not root:
        return 0
    
    def get_height(node):
        height = 0
        while node:
            height += 1
            node = node.left
        return height
    
    left_height = get_height(root.left)
    right_height = get_height(root.right)
    
    if left_height == right_height:
        # Left subtree is perfect
        return (1 << left_height) + count_nodes(root.right)
    else:
        # Right subtree is perfect
        return (1 << right_height) + count_nodes(root.left)

# Time: O(log²n), Space: O(log n)
```

## Tree Construction

### From Traversals

**Build from Inorder and Preorder:**
```python
def build_tree_inorder_preorder(preorder, inorder):
    """
    Construct binary tree from preorder and inorder traversals
    """
    if not preorder or not inorder:
        return None
    
    # First element in preorder is root
    root_val = preorder[0]
    root = TreeNode(root_val)
    
    # Find root in inorder to split into left and right
    mid = inorder.index(root_val)
    
    # Recursively build subtrees
    root.left = build_tree_inorder_preorder(
        preorder[1:mid+1], inorder[:mid]
    )
    root.right = build_tree_inorder_preorder(
        preorder[mid+1:], inorder[mid+1:]
    )
    
    return root

# Time: O(n²) due to index(), can optimize to O(n) with hashmap
```

**Build from Inorder and Postorder:**
```python
def build_tree_inorder_postorder(inorder, postorder):
    """
    Construct binary tree from inorder and postorder traversals
    """
    if not inorder or not postorder:
        return None
    
    # Last element in postorder is root
    root_val = postorder[-1]
    root = TreeNode(root_val)
    
    # Find root in inorder
    mid = inorder.index(root_val)
    
    # Recursively build subtrees
    root.left = build_tree_inorder_postorder(
        inorder[:mid], postorder[:mid]
    )
    root.right = build_tree_inorder_postorder(
        inorder[mid+1:], postorder[mid:-1]
    )
    
    return root
```

## Tree Patterns

### Pattern 1: Top-Down Recursion
Pass information from parent to children.

```python
def top_down(node, param):
    if not node:
        return
    
    # Use param (passed from parent)
    process(node, param)
    
    # Pass to children
    top_down(node.left, new_param)
    top_down(node.right, new_param)
```

### Pattern 2: Bottom-Up Recursion
Compute information from children and return to parent.

```python
def bottom_up(node):
    if not node:
        return base_value
    
    # Get info from children
    left_result = bottom_up(node.left)
    right_result = bottom_up(node.right)
    
    # Compute and return
    return combine(node, left_result, right_result)
```

### Pattern 3: Level-Order Processing
Process tree level by level using BFS.

```python
def level_order_pattern(root):
    queue = deque([root])
    
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            process(node)
            add_children_to_queue(node, queue)
```

## Summary

**Key Concepts:**
- Trees are hierarchical structures
- Binary trees have at most 2 children per node
- Three main DFS traversals: inorder, preorder, postorder
- BFS traversal processes level by level

**Common Techniques:**
- Recursion (most natural for trees)
- Stack for iterative DFS
- Queue for BFS
- Divide and conquer
- Top-down vs bottom-up approaches

**Time Complexities:**
- Traversal: O(n)
- Search: O(n) for binary tree
- Height: O(n) worst case, O(log n) balanced

**Practice Tips:**
- Draw the tree
- Identify base cases
- Think recursively
- Consider both top-down and bottom-up approaches

