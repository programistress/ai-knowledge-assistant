# Linked Lists

## Introduction

A linked list is a linear data structure where elements are stored in nodes, and each node points to the next node in the sequence. Unlike arrays, linked lists don't require contiguous memory and allow for efficient insertions and deletions.

## Types of Linked Lists

### 1. Singly Linked List

Each node contains data and a pointer to the next node.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class SinglyLinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        """Add node at the end - O(n)"""
        new_node = ListNode(val)
        
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, val):
        """Add node at the beginning - O(1)"""
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, val):
        """Delete first occurrence of value - O(n)"""
        if not self.head:
            return
        
        # If head needs to be deleted
        if self.head.val == val:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                return
            current = current.next
    
    def search(self, val):
        """Search for value - O(n)"""
        current = self.head
        while current:
            if current.val == val:
                return True
            current = current.next
        return False
    
    def display(self):
        """Print the linked list"""
        elements = []
        current = self.head
        while current:
            elements.append(str(current.val))
            current = current.next
        print(" -> ".join(elements))
```

**Time Complexities:**
- Access: O(n)
- Search: O(n)
- Insert at head: O(1)
- Insert at tail: O(n) without tail pointer, O(1) with tail pointer
- Delete at head: O(1)
- Delete at tail: O(n)

### 2. Doubly Linked List

Each node contains data, a pointer to the next node, and a pointer to the previous node.

```python
class DoublyListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def append(self, val):
        """Add node at the end - O(1)"""
        new_node = DoublyListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
            return
        
        self.tail.next = new_node
        new_node.prev = self.tail
        self.tail = new_node
    
    def prepend(self, val):
        """Add node at the beginning - O(1)"""
        new_node = DoublyListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
            return
        
        new_node.next = self.head
        self.head.prev = new_node
        self.head = new_node
    
    def delete(self, val):
        """Delete first occurrence - O(n)"""
        current = self.head
        
        while current:
            if current.val == val:
                # Update previous node's next pointer
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                
                # Update next node's prev pointer
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                
                return
            current = current.next
```

**Advantages over Singly Linked List:**
- Can traverse in both directions
- Easier deletion (have access to previous node)
- Easier reverse traversal

**Disadvantages:**
- Extra memory for previous pointer
- Slightly more complex to maintain

### 3. Circular Linked List

Last node points back to the first node, forming a circle.

```python
class CircularLinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        new_node = ListNode(val)
        
        if not self.head:
            self.head = new_node
            new_node.next = self.head  # Points to itself
            return
        
        current = self.head
        while current.next != self.head:
            current = current.next
        
        current.next = new_node
        new_node.next = self.head
    
    def display(self):
        if not self.head:
            return
        
        current = self.head
        while True:
            print(current.val, end=" -> ")
            current = current.next
            if current == self.head:
                break
        print("(back to head)")
```

**Use Cases:**
- Round-robin scheduling
- Circular buffers
- Music playlists
- Multiplayer game turns

## Common Linked List Problems

### Problem 1: Reverse Linked List

**Iterative Approach:**
```python
def reverse_list(head):
    """
    Reverse a singly linked list
    """
    prev = None
    current = head
    
    while current:
        # Save next node
        next_node = current.next
        # Reverse the pointer
        current.next = prev
        # Move pointers forward
        prev = current
        current = next_node
    
    return prev

# Time: O(n), Space: O(1)
```

**Recursive Approach:**
```python
def reverse_list_recursive(head):
    # Base case
    if not head or not head.next:
        return head
    
    # Reverse rest of the list
    new_head = reverse_list_recursive(head.next)
    
    # Reverse current node's pointer
    head.next.next = head
    head.next = None
    
    return new_head

# Time: O(n), Space: O(n) for recursion stack
```

### Problem 2: Detect Cycle (Floyd's Cycle Detection)

```python
def has_cycle(head):
    """
    Detect if linked list has a cycle using fast and slow pointers
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True

# Time: O(n), Space: O(1)
```

**Find the Start of the Cycle:**
```python
def detect_cycle(head):
    """
    Return the node where cycle begins, or None if no cycle
    """
    if not head or not head.next:
        return None
    
    # Phase 1: Detect cycle
    slow = fast = head
    has_cycle = False
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            has_cycle = True
            break
    
    if not has_cycle:
        return None
    
    # Phase 2: Find start of cycle
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow

# Time: O(n), Space: O(1)
```

### Problem 3: Find Middle of Linked List

```python
def find_middle(head):
    """
    Find middle node using slow and fast pointers
    """
    if not head:
        return None
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

# When fast reaches end, slow is at middle
# Time: O(n), Space: O(1)
```

### Problem 4: Merge Two Sorted Lists

```python
def merge_two_lists(l1, l2):
    """
    Merge two sorted linked lists into one sorted list
    """
    # Create dummy node
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    # Attach remaining nodes
    current.next = l1 if l1 else l2
    
    return dummy.next

# Time: O(n + m), Space: O(1)
```

### Problem 5: Remove Nth Node From End

```python
def remove_nth_from_end(head, n):
    """
    Remove the nth node from the end of the list
    """
    dummy = ListNode(0)
    dummy.next = head
    fast = slow = dummy
    
    # Move fast n+1 steps ahead
    for _ in range(n + 1):
        if not fast:
            return head
        fast = fast.next
    
    # Move both until fast reaches end
    while fast:
        fast = fast.next
        slow = slow.next
    
    # Remove the node
    slow.next = slow.next.next
    
    return dummy.next

# Time: O(n), Space: O(1)
```

### Problem 6: Palindrome Linked List

```python
def is_palindrome(head):
    """
    Check if linked list is a palindrome
    """
    if not head or not head.next:
        return True
    
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second_half = reverse_list(slow.next)
    slow.next = None
    
    # Compare both halves
    p1, p2 = head, second_half
    result = True
    
    while p2:
        if p1.val != p2.val:
            result = False
            break
        p1 = p1.next
        p2 = p2.next
    
    # Optional: restore list
    slow.next = reverse_list(second_half)
    
    return result

# Time: O(n), Space: O(1)
```

### Problem 7: Intersection of Two Linked Lists

```python
def get_intersection_node(headA, headB):
    """
    Find node where two linked lists intersect
    """
    if not headA or not headB:
        return None
    
    # Get lengths
    lenA = lenB = 0
    nodeA, nodeB = headA, headB
    
    while nodeA:
        lenA += 1
        nodeA = nodeA.next
    
    while nodeB:
        lenB += 1
        nodeB = nodeB.next
    
    # Align starting points
    nodeA, nodeB = headA, headB
    
    if lenA > lenB:
        for _ in range(lenA - lenB):
            nodeA = nodeA.next
    else:
        for _ in range(lenB - lenA):
            nodeB = nodeB.next
    
    # Find intersection
    while nodeA and nodeB:
        if nodeA == nodeB:
            return nodeA
        nodeA = nodeA.next
        nodeB = nodeB.next
    
    return None

# Time: O(m + n), Space: O(1)
```

**Elegant Solution:**
```python
def get_intersection_node_elegant(headA, headB):
    """
    When pointers reach end, switch to other list's head.
    They'll meet at intersection or both become None.
    """
    if not headA or not headB:
        return None
    
    pA, pB = headA, headB
    
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    
    return pA

# Time: O(m + n), Space: O(1)
```

### Problem 8: Add Two Numbers

```python
def add_two_numbers(l1, l2):
    """
    Add two numbers represented by linked lists (digits in reverse order)
    Example: 2 -> 4 -> 3 (represents 342)
           + 5 -> 6 -> 4 (represents 465)
           = 7 -> 0 -> 8 (represents 807)
    """
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10
        
        current.next = ListNode(digit)
        current = current.next
        
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next

# Time: O(max(m, n)), Space: O(max(m, n))
```

### Problem 9: Reorder List

```python
def reorder_list(head):
    """
    Reorder list from L0→L1→...→Ln-1→Ln to L0→Ln→L1→Ln-1→L2→Ln-2→...
    """
    if not head or not head.next:
        return
    
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second = slow.next
    slow.next = None
    second = reverse_list(second)
    
    # Merge two halves
    first = head
    while second:
        temp1, temp2 = first.next, second.next
        first.next = second
        second.next = temp1
        first, second = temp1, temp2

# Time: O(n), Space: O(1)
```

### Problem 10: Copy List with Random Pointer

```python
class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copy_random_list(head):
    """
    Create a deep copy of linked list with random pointers
    """
    if not head:
        return None
    
    # Step 1: Create copy nodes interleaved with original
    current = head
    while current:
        copy = Node(current.val)
        copy.next = current.next
        current.next = copy
        current = copy.next
    
    # Step 2: Set random pointers for copy nodes
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next
    
    # Step 3: Separate original and copy lists
    current = head
    copy_head = head.next
    
    while current:
        copy = current.next
        current.next = copy.next
        copy.next = copy.next.next if copy.next else None
        current = current.next
    
    return copy_head

# Time: O(n), Space: O(1) excluding the output
```

## Skip Lists

Skip lists are probabilistic data structures that allow O(log n) search complexity and O(log n) insertion complexity within an ordered sequence of elements.

### Structure

A skip list consists of multiple levels of linked lists:
- Level 0: Regular linked list with all elements
- Higher levels: Express lanes with fewer elements

```python
import random

class SkipNode:
    def __init__(self, val, level):
        self.val = val
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p  # Probability for level increase
        self.head = SkipNode(float('-inf'), max_level)
        self.level = 0
    
    def random_level(self):
        """Generate random level for new node"""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def search(self, target):
        """Search for target value"""
        current = self.head
        
        # Start from highest level
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < target:
                current = current.forward[i]
        
        # Move to level 0
        current = current.forward[0]
        
        return current and current.val == target
    
    def insert(self, val):
        """Insert value into skip list"""
        update = [None] * (self.max_level + 1)
        current = self.head
        
        # Find position to insert
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < val:
                current = current.forward[i]
            update[i] = current
        
        # Generate random level
        new_level = self.random_level()
        
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.head
            self.level = new_level
        
        # Create new node and update pointers
        new_node = SkipNode(val, new_level)
        
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
    
    def delete(self, val):
        """Delete value from skip list"""
        update = [None] * (self.max_level + 1)
        current = self.head
        
        # Find node to delete
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < val:
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        if current and current.val == val:
            # Update all levels
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]
            
            # Update list level
            while self.level > 0 and not self.head.forward[self.level]:
                self.level -= 1

# Time Complexity (average):
# - Search: O(log n)
# - Insert: O(log n)
# - Delete: O(log n)
# Space: O(n log n) expected
```

## Linked List Patterns and Tips

### Pattern 1: Dummy Node

Use a dummy node to simplify edge cases (empty list, operations at head).

```python
dummy = ListNode(0)
dummy.next = head
# Work with dummy.next
return dummy.next
```

### Pattern 2: Fast and Slow Pointers

Used for finding middle, detecting cycles, finding nth from end.

```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
```

### Pattern 3: Previous Pointer

Keep track of previous node for deletion/insertion.

```python
prev = None
current = head
while current:
    # Do something
    prev = current
    current = current.next
```

### Pattern 4: Runner Technique

Use two pointers at different positions.

```python
# Remove nth from end
fast = head
for _ in range(n):
    fast = fast.next

slow = head
while fast.next:
    slow = slow.next
    fast = fast.next
```

## Comparison: Array vs Linked List

| Operation | Array | Linked List |
|-----------|-------|-------------|
| Access by index | O(1) | O(n) |
| Search | O(n) | O(n) |
| Insert at beginning | O(n) | O(1) |
| Insert at end | O(1) amortized | O(n) or O(1) with tail |
| Insert at middle | O(n) | O(n) to find + O(1) to insert |
| Delete at beginning | O(n) | O(1) |
| Delete at end | O(1) | O(n) or O(1) with doubly linked |
| Memory usage | Contiguous, better cache | Non-contiguous, extra pointer |

### When to Use Linked Lists

✅ **Use when:**
- Frequent insertions/deletions at beginning
- Don't need random access
- Size varies significantly
- Implementing stacks, queues, or hash table chains

❌ **Don't use when:**
- Need fast random access
- Memory locality is important
- Frequent searches
- Memory overhead is a concern

## Summary

**Key Concepts:**
- Linear data structure with nodes
- Each node contains data and pointer(s)
- Dynamic size, efficient insertions/deletions
- No random access

**Common Techniques:**
- Two pointers (fast/slow, previous/current)
- Dummy nodes for edge cases
- Recursion for elegant solutions
- In-place modifications

**Practice Tips:**
- Draw diagrams for complex operations
- Handle edge cases: empty list, single node, two nodes
- Watch for null pointer errors
- Consider both iterative and recursive solutions

