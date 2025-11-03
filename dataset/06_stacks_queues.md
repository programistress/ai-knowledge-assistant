# Stacks and Queues

## AStacks

### Introduction

A stack is a linear data structure that follows the **Last In, First Out (LIFO)** principle. Elements are added and removed from the same end, called the "top" of the stack.

### Stack Operations

**Core Operations:**
- **Push**: Add element to top - O(1)
- **Pop**: Remove element from top - O(1)
- **Peek/Top**: View top element without removing - O(1)
- **isEmpty**: Check if stack is empty - O(1)
- **Size**: Get number of elements - O(1)

### Stack Implementation

**Using Python List:**
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        """Return top item without removing"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        """Check if stack is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items"""
        return len(self.items)
    
    def __str__(self):
        return str(self.items)

# Usage
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # 3
print(stack.peek())  # 2
```

**Using Linked List:**
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class StackLinkedList:
    def __init__(self):
        self.top = None
        self._size = 0
    
    def push(self, item):
        """Add item to top"""
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self._size += 1
    
    def pop(self):
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        
        item = self.top.data
        self.top = self.top.next
        self._size -= 1
        return item
    
    def peek(self):
        """Return top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.top.data
    
    def is_empty(self):
        return self.top is None
    
    def size(self):
        return self._size
```

### Common Stack Problems

**Problem 1: Valid Parentheses**
```python
def is_valid_parentheses(s):
    """
    Check if parentheses/brackets are valid and properly nested
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

# Example: "({[]})" → True, "([)]" → False
# Time: O(n), Space: O(n)
```

**Problem 2: Min Stack**
```python
class MinStack:
    """
    Stack that supports push, pop, top, and retrieving minimum in O(1)
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        # Push to min_stack if empty or val is new minimum
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        if not self.stack:
            return
        
        val = self.stack.pop()
        # If popped value was minimum, pop from min_stack too
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val
    
    def top(self):
        return self.stack[-1] if self.stack else None
    
    def get_min(self):
        return self.min_stack[-1] if self.min_stack else None

# All operations: O(1) time, O(n) space
```

**Problem 3: Evaluate Reverse Polish Notation**
```python
def eval_rpn(tokens):
    """
    Evaluate arithmetic expression in Reverse Polish Notation
    Example: ["2","1","+","3","*"] = (2 + 1) * 3 = 9
    """
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            # Pop two operands
            b = stack.pop()
            a = stack.pop()
            
            # Perform operation
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            else:  # token == '/'
                result = int(a / b)  # Truncate toward zero
            
            stack.append(result)
        else:
            stack.append(int(token))
    
    return stack[0]

# Time: O(n), Space: O(n)
```

**Problem 4: Daily Temperatures**
```python
def daily_temperatures(temperatures):
    """
    Return array where answer[i] is number of days until warmer temperature
    """
    n = len(temperatures)
    answer = [0] * n
    stack = []  # Stack of indices
    
    for i, temp in enumerate(temperatures):
        # While current temp is warmer than stack top
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            answer[prev_index] = i - prev_index
        stack.append(i)
    
    return answer

# Example: [73,74,75,71,69,72,76,73] → [1,1,4,2,1,1,0,0]
# Time: O(n), Space: O(n)
```

**Problem 5: Largest Rectangle in Histogram**
```python
def largest_rectangle_area(heights):
    """
    Find area of largest rectangle in histogram
    """
    stack = []  # Stack of indices
    max_area = 0
    heights.append(0)  # Sentinel to flush stack
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height_index = stack.pop()
            height = heights[height_index]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    heights.pop()  # Remove sentinel
    return max_area

# Time: O(n), Space: O(n)
```

**Problem 6: Simplify Path**
```python
def simplify_path(path):
    """
    Simplify Unix-style file path
    Example: "/a/./b/../../c/" → "/c"
    """
    stack = []
    
    for component in path.split('/'):
        if component == '..' and stack:
            stack.pop()
        elif component and component != '.' and component != '..':
            stack.append(component)
    
    return '/' + '/'.join(stack)

# Time: O(n), Space: O(n)
```

**Problem 7: Decode String**
```python
def decode_string(s):
    """
    Decode string with pattern k[encoded_string]
    Example: "3[a2[c]]" → "accaccacc"
    """
    stack = []
    current_num = 0
    current_str = ""
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Push current state to stack
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif char == ']':
            # Pop and decode
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char
    
    return current_str

# Time: O(n), Space: O(n)
```

### Stack Applications

1. **Function call stack** (recursion)
2. **Undo/Redo operations** in editors
3. **Browser history** (back button)
4. **Expression evaluation** (infix to postfix)
5. **Backtracking algorithms**
6. **Depth-First Search** (DFS)
7. **Parsing** (compilers, HTML/XML)

---

## Queues

### Introduction

A queue is a linear data structure that follows the **First In, First Out (FIFO)** principle. Elements are added at the rear and removed from the front.

### Queue Operations

**Core Operations:**
- **Enqueue**: Add element to rear - O(1)
- **Dequeue**: Remove element from front - O(1)
- **Front/Peek**: View front element - O(1)
- **isEmpty**: Check if queue is empty - O(1)
- **Size**: Get number of elements - O(1)

### Queue Implementation

**Using Python List (Inefficient for dequeue):**
```python
class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        """Add item to rear"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.pop(0)  # O(n) - not efficient!
    
    def front(self):
        """Return front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

**Using collections.deque (Efficient):**
```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)  # O(1)
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()  # O(1)
    
    def front(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

**Using Linked List:**
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class QueueLinkedList:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0
    
    def enqueue(self, item):
        """Add item to rear"""
        new_node = Node(item)
        
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        
        self._size += 1
    
    def dequeue(self):
        """Remove and return front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.front.data
        self.front = self.front.next
        
        if self.front is None:
            self.rear = None
        
        self._size -= 1
        return item
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.front.data
    
    def is_empty(self):
        return self.front is None
    
    def size(self):
        return self._size
```

### Circular Queue

A circular queue uses a fixed-size array where rear wraps around to the beginning.

```python
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enqueue(self, item):
        """Add item to rear"""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1
    
    def dequeue(self):
        """Remove and return front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.queue[self.front]
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
```

### Deque (Double-Ended Queue)

A deque allows insertion and deletion at both ends.

```python
from collections import deque

# Python's built-in deque
d = deque()

# Operations at rear
d.append(1)        # Add to right
d.pop()            # Remove from right

# Operations at front
d.appendleft(0)    # Add to left
d.popleft()        # Remove from left

# Access
d[0]               # Front element
d[-1]              # Rear element

# All operations are O(1)
```

**Custom Deque Implementation:**
```python
class Deque:
    def __init__(self):
        self.items = []
    
    def add_front(self, item):
        self.items.insert(0, item)
    
    def add_rear(self, item):
        self.items.append(item)
    
    def remove_front(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items.pop(0)
    
    def remove_rear(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items.pop()
    
    def peek_front(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items[0]
    
    def peek_rear(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

### Common Queue Problems

**Problem 1: Implement Stack Using Queues**
```python
from collections import deque

class MyStack:
    """Implement stack using two queues"""
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
    
    def push(self, x):
        """Add to q2, transfer q1 to q2, swap queues"""
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self):
        return self.q1.popleft()
    
    def top(self):
        return self.q1[0]
    
    def empty(self):
        return len(self.q1) == 0

# push/pop: O(n)/O(1), Space: O(n)
```

**Problem 2: Implement Queue Using Stacks**
```python
class MyQueue:
    """Implement queue using two stacks"""
    def __init__(self):
        self.stack_in = []
        self.stack_out = []
    
    def push(self, x):
        """Add to input stack"""
        self.stack_in.append(x)
    
    def pop(self):
        """Remove from output stack (transfer if needed)"""
        self._move()
        return self.stack_out.pop()
    
    def peek(self):
        """View front element"""
        self._move()
        return self.stack_out[-1]
    
    def _move(self):
        """Transfer from input to output stack if output is empty"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
    
    def empty(self):
        return not self.stack_in and not self.stack_out

# push: O(1), pop/peek: O(1) amortized
```

**Problem 3: Sliding Window Maximum**
```python
from collections import deque

def max_sliding_window(nums, k):
    """
    Return max value in each sliding window of size k
    """
    result = []
    dq = deque()  # Store indices
    
    for i, num in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove indices with smaller values (they're useless)
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Example: nums=[1,3,-1,-3,5,3,6,7], k=3 → [3,3,5,5,6,7]
# Time: O(n), Space: O(k)
```

**Problem 4: First Unique Character in Stream**
```python
from collections import deque

class FirstUnique:
    """Find first unique character in a stream"""
    def __init__(self):
        self.queue = deque()
        self.count = {}
    
    def add(self, char):
        """Add character to stream"""
        self.count[char] = self.count.get(char, 0) + 1
        self.queue.append(char)
    
    def get_first_unique(self):
        """Get first unique character"""
        # Remove non-unique from front
        while self.queue and self.count[self.queue[0]] > 1:
            self.queue.popleft()
        
        return self.queue[0] if self.queue else None

# Time: O(1) amortized for both operations
```

**Problem 5: Design Circular Deque**
```python
class MyCircularDeque:
    def __init__(self, k):
        self.capacity = k
        self.deque = [0] * k
        self.front = 0
        self.rear = 0
        self.size = 0
    
    def insertFront(self, value):
        if self.isFull():
            return False
        self.front = (self.front - 1) % self.capacity
        self.deque[self.front] = value
        self.size += 1
        return True
    
    def insertLast(self, value):
        if self.isFull():
            return False
        self.deque[self.rear] = value
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
        return True
    
    def deleteFront(self):
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True
    
    def deleteLast(self):
        if self.isEmpty():
            return False
        self.rear = (self.rear - 1) % self.capacity
        self.size -= 1
        return True
    
    def getFront(self):
        return -1 if self.isEmpty() else self.deque[self.front]
    
    def getRear(self):
        return -1 if self.isEmpty() else self.deque[(self.rear - 1) % self.capacity]
    
    def isEmpty(self):
        return self.size == 0
    
    def isFull(self):
        return self.size == self.capacity
```

### Queue Applications

1. **Breadth-First Search** (BFS)
2. **Level-order traversal** of trees
3. **Task scheduling** (CPU, printer)
4. **Buffer management** (keyboard, IO)
5. **Asynchronous data transfer** (pipes, file IO)
6. **Waiting lines** (customer service, ticket booking)

## Summary

### Stack vs Queue

| Feature | Stack | Queue |
|---------|-------|-------|
| Principle | LIFO | FIFO |
| Operations | Push, Pop, Peek | Enqueue, Dequeue, Front |
| Use cases | Undo/Redo, DFS, Recursion | BFS, Scheduling, Buffers |
| Real-world | Plate stack, Browser back | Line/Queue, Printer queue |

### Key Patterns

**Stack Patterns:**
- Monotonic stack (next greater/smaller element)
- Using stack to convert recursion to iteration
- Matching/balancing problems (parentheses)
- Expression evaluation

**Queue Patterns:**
- Level-order/BFS traversal
- Sliding window with deque
- Stream processing
- Task scheduling

### Time Complexities

| Operation | Stack | Queue (deque) | Queue (list) |
|-----------|-------|---------------|--------------|
| Push/Enqueue | O(1) | O(1) | O(1) |
| Pop/Dequeue | O(1) | O(1) | O(n) |
| Peek/Front | O(1) | O(1) | O(1) |
| Search | O(n) | O(n) | O(n) |

### Best Practices

1. Use `collections.deque` for efficient queues in Python
2. Use dummy nodes/sentinels to simplify edge cases
3. Consider circular implementations for fixed-size buffers
4. Think about stack when you see nested structures or need LIFO
5. Think about queue when you see level-by-level processing or need FIFO

