# Graphs - Fundamentals and Traversals

## Introduction to Graphs

A graph is a non-linear data structure consisting of vertices (nodes) and edges (connections between nodes). Graphs model relationships and networks in many real-world scenarios.

### Graph Terminology

- **Vertex (Node)**: Basic unit of a graph
- **Edge**: Connection between two vertices
- **Adjacent**: Two vertices connected by an edge
- **Degree**: Number of edges connected to a vertex
  - **In-degree**: Number of incoming edges (directed graphs)
  - **Out-degree**: Number of outgoing edges (directed graphs)
- **Path**: Sequence of vertices connected by edges
- **Cycle**: Path that starts and ends at the same vertex
- **Connected**: Graph where path exists between any two vertices
- **Component**: Maximal connected subgraph
- **Weighted**: Edges have associated weights/costs
- **Sparse**: Few edges relative to vertices
- **Dense**: Many edges relative to vertices

### Types of Graphs

#### 1. Undirected Graph
Edges have no direction. If (u, v) exists, you can go from u to v and v to u.

```
    1 --- 2
    |     |
    3 --- 4
```

#### 2. Directed Graph (Digraph)
Edges have direction. (u, v) means you can only go from u to v.

```
    1 --> 2
    ^     |
    |     v
    3 <-- 4
```

#### 3. Weighted Graph
Edges have weights/costs.

```
    1 -5- 2
    |     |
   10    15
    |     |
    3 -20- 4
```

#### 4. Cyclic vs Acyclic
- **Cyclic**: Contains at least one cycle
- **Acyclic**: No cycles (trees are acyclic graphs)
- **DAG**: Directed Acyclic Graph

#### 5. Complete Graph
Every pair of vertices is connected by an edge.
- n vertices have n(n-1)/2 edges in undirected graph

#### 6. Bipartite Graph
Vertices can be divided into two sets where edges only connect vertices from different sets.

```
Set A: 1, 3
Set B: 2, 4
Edges: 1-2, 1-4, 3-2, 3-4
```

## Graph Representation

### 1. Adjacency Matrix

2D array where `matrix[i][j]` = 1 if edge exists between vertex i and j.

```python
class GraphMatrix:
    def __init__(self, num_vertices):
        self.V = num_vertices
        self.matrix = [[0] * num_vertices for _ in range(num_vertices)]
    
    def add_edge(self, u, v, directed=False):
        """Add edge from u to v"""
        self.matrix[u][v] = 1
        if not directed:
            self.matrix[v][u] = 1
    
    def remove_edge(self, u, v, directed=False):
        """Remove edge from u to v"""
        self.matrix[u][v] = 0
        if not directed:
            self.matrix[v][u] = 0
    
    def has_edge(self, u, v):
        """Check if edge exists"""
        return self.matrix[u][v] == 1
    
    def get_neighbors(self, v):
        """Get all neighbors of vertex v"""
        return [i for i in range(self.V) if self.matrix[v][i] == 1]
    
    def print_graph(self):
        for row in self.matrix:
            print(row)

# For weighted graph
class WeightedGraphMatrix:
    def __init__(self, num_vertices):
        self.V = num_vertices
        self.matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
        # Distance from vertex to itself is 0
        for i in range(num_vertices):
            self.matrix[i][i] = 0
    
    def add_edge(self, u, v, weight, directed=False):
        self.matrix[u][v] = weight
        if not directed:
            self.matrix[v][u] = weight

# Space: O(V²)
# Edge lookup: O(1)
# Get neighbors: O(V)
# Good for dense graphs
```

### 2. Adjacency List

Array of lists where each index represents a vertex and contains list of adjacent vertices.

```python
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v, directed=False):
        """Add edge from u to v"""
        self.graph[u].append(v)
        if not directed:
            self.graph[v].append(u)
    
    def remove_edge(self, u, v, directed=False):
        """Remove edge from u to v"""
        if v in self.graph[u]:
            self.graph[u].remove(v)
        if not directed and u in self.graph[v]:
            self.graph[v].remove(u)
    
    def get_neighbors(self, v):
        """Get all neighbors of vertex v"""
        return self.graph[v]
    
    def has_edge(self, u, v):
        """Check if edge exists"""
        return v in self.graph[u]
    
    def print_graph(self):
        for vertex in self.graph:
            print(f"{vertex}: {self.graph[vertex]}")

# For weighted graph
class WeightedGraph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v, weight, directed=False):
        """Add weighted edge"""
        self.graph[u].append((v, weight))
        if not directed:
            self.graph[v].append((u, weight))

# Space: O(V + E)
# Edge lookup: O(degree)
# Get neighbors: O(1)
# Good for sparse graphs
```

### 3. Edge List

List of all edges in the graph.

```python
class EdgeListGraph:
    def __init__(self):
        self.edges = []
    
    def add_edge(self, u, v, weight=1):
        """Add edge"""
        self.edges.append((u, v, weight))
    
    def print_edges(self):
        for u, v, w in self.edges:
            print(f"{u} -> {v} (weight: {w})")

# Space: O(E)
# Good for algorithms that iterate over all edges
# Used in Kruskal's MST, Bellman-Ford
```

## Graph Traversal

### Depth-First Search (DFS)

Explores as far as possible along each branch before backtracking.

**Recursive Implementation:**
```python
def dfs(graph, start, visited=None):
    """
    DFS traversal of graph
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start, end=' ')
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited

# Time: O(V + E)
# Space: O(V) for recursion stack and visited set
```

**Iterative Implementation (using stack):**
```python
def dfs_iterative(graph, start):
    """
    Iterative DFS using stack
    """
    visited = set()
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        
        if vertex not in visited:
            visited.add(vertex)
            print(vertex, end=' ')
            
            # Add neighbors to stack
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return visited

# Time: O(V + E), Space: O(V)
```

**DFS with Path Tracking:**
```python
def dfs_path(graph, start, end, path=None):
    """
    Find a path from start to end using DFS
    """
    if path is None:
        path = []
    
    path = path + [start]
    
    if start == end:
        return path
    
    for neighbor in graph[start]:
        if neighbor not in path:
            new_path = dfs_path(graph, neighbor, end, path)
            if new_path:
                return new_path
    
    return None
```

**Find All Paths:**
```python
def find_all_paths(graph, start, end, path=None):
    """
    Find all paths from start to end
    """
    if path is None:
        path = []
    
    path = path + [start]
    
    if start == end:
        return [path]
    
    paths = []
    for neighbor in graph[start]:
        if neighbor not in path:
            new_paths = find_all_paths(graph, neighbor, end, path)
            paths.extend(new_paths)
    
    return paths
```

### Breadth-First Search (BFS)

Explores all vertices at current depth before moving to next depth level.

```python
from collections import deque

def bfs(graph, start):
    """
    BFS traversal of graph
    """
    visited = set([start])
    queue = deque([start])
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited

# Time: O(V + E)
# Space: O(V)
```

**BFS with Level Tracking:**
```python
def bfs_levels(graph, start):
    """
    BFS with level information
    """
    visited = {start}
    queue = deque([(start, 0)])
    levels = {start: 0}
    
    while queue:
        vertex, level = queue.popleft()
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                levels[neighbor] = level + 1
                queue.append((neighbor, level + 1))
    
    return levels
```

**Shortest Path (Unweighted):**
```python
def shortest_path(graph, start, end):
    """
    Find shortest path in unweighted graph using BFS
    """
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        vertex, path = queue.popleft()
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                
                if neighbor == end:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    
    return None  # No path found
```

## DFS vs BFS

| Feature | DFS | BFS |
|---------|-----|-----|
| Data Structure | Stack (recursion) | Queue |
| Memory | O(h) where h is height | O(w) where w is width |
| Path Found | Not necessarily shortest | Shortest (unweighted) |
| Complete | No (may get stuck in infinite path) | Yes |
| Optimal | No | Yes (for unweighted) |
| Use Cases | Topological sort, cycle detection, path finding | Shortest path, level-order, nearest neighbors |

## Common Graph Problems

### Problem 1: Number of Islands

```python
def num_islands(grid):
    """
    Count number of islands in 2D grid
    '1' = land, '0' = water
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        # Out of bounds or water
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == '0'):
            return
        
        # Mark as visited
        grid[r][c] = '0'
        
        # Explore all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)
    
    return islands

# Time: O(rows × cols), Space: O(rows × cols) worst case
```

### Problem 2: Clone Graph

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node):
    """
    Create deep copy of graph
    """
    if not node:
        return None
    
    clones = {}
    
    def dfs(node):
        if node in clones:
            return clones[node]
        
        # Create clone
        clone = Node(node.val)
        clones[node] = clone
        
        # Clone neighbors
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)

# Time: O(V + E), Space: O(V)
```

### Problem 3: Course Schedule (Cycle Detection)

```python
def can_finish(num_courses, prerequisites):
    """
    Check if all courses can be finished (detect cycle in directed graph)
    """
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * num_courses
    
    def has_cycle(course):
        if state[course] == 1:
            return True  # Cycle detected
        if state[course] == 2:
            return False  # Already checked
        
        state[course] = 1  # Mark as visiting
        
        for next_course in graph[course]:
            if has_cycle(next_course):
                return True
        
        state[course] = 2  # Mark as visited
        return False
    
    for course in range(num_courses):
        if has_cycle(course):
            return False
    
    return True

# Time: O(V + E), Space: O(V + E)
```

### Problem 4: Number of Connected Components

```python
def count_components(n, edges):
    """
    Count number of connected components in undirected graph
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    components = 0
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    for node in range(n):
        if node not in visited:
            components += 1
            dfs(node)
    
    return components

# Time: O(V + E), Space: O(V + E)
```

### Problem 5: Is Graph Bipartite

```python
def is_bipartite(graph):
    """
    Check if graph can be colored with 2 colors (bipartite)
    """
    n = len(graph)
    colors = {}
    
    def bfs(start):
        queue = deque([start])
        colors[start] = 0
        
        while queue:
            node = queue.popleft()
            
            for neighbor in graph[node]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[node]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[node]:
                    return False
        
        return True
    
    # Check all components
    for node in range(n):
        if node not in colors:
            if not bfs(node):
                return False
    
    return True

# Time: O(V + E), Space: O(V)
```

### Problem 6: Word Ladder

```python
from collections import deque

def ladder_length(begin_word, end_word, word_list):
    """
    Find shortest transformation sequence from begin_word to end_word
    """
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    
    queue = deque([(begin_word, 1)])
    
    while queue:
        word, steps = queue.popleft()
        
        if word == end_word:
            return steps
        
        # Try changing each character
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                
                if next_word in word_set:
                    word_set.remove(next_word)
                    queue.append((next_word, steps + 1))
    
    return 0

# Time: O(M² × N) where M = word length, N = word count
```

### Problem 7: All Paths from Source to Target

```python
def all_paths_source_target(graph):
    """
    Find all paths from node 0 to node n-1 in DAG
    """
    n = len(graph)
    result = []
    
    def dfs(node, path):
        if node == n - 1:
            result.append(path[:])
            return
        
        for neighbor in graph[node]:
            path.append(neighbor)
            dfs(neighbor, path)
            path.pop()
    
    dfs(0, [0])
    return result

# Time: O(2^V × V), Space: O(V)
```

## Graph Coloring

```python
def graph_coloring(graph, num_colors):
    """
    Color graph with minimum colors such that no adjacent vertices have same color
    """
    n = len(graph)
    colors = [-1] * n
    
    def is_safe(node, color):
        for neighbor in graph[node]:
            if colors[neighbor] == color:
                return False
        return True
    
    def solve(node):
        if node == n:
            return True
        
        for color in range(num_colors):
            if is_safe(node, color):
                colors[node] = color
                if solve(node + 1):
                    return True
                colors[node] = -1
        
        return False
    
    if solve(0):
        return colors
    return None
```

## Summary

**Key Concepts:**
- Graphs model relationships between entities
- Can be directed or undirected, weighted or unweighted
- Two main representations: adjacency matrix and adjacency list
- Two main traversals: DFS (stack/recursion) and BFS (queue)

**DFS Applications:**
- Path finding
- Cycle detection
- Topological sorting
- Connected components
- Solving mazes/puzzles

**BFS Applications:**
- Shortest path (unweighted)
- Level-order traversal
- Finding nearest neighbors
- Network broadcasting

**Time Complexities:**
- DFS/BFS: O(V + E)
- Space: O(V) for visited set and stack/queue

**Practice Tips:**
- Choose right representation (matrix vs list)
- Track visited nodes to avoid cycles
- Use DFS for paths, BFS for shortest distance
- Consider directed vs undirected carefully
- Draw graphs to visualize problems

