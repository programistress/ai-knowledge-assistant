# Advanced Graph Algorithms

## Topological Sort

### Introduction

Topological sorting is a linear ordering of vertices in a Directed Acyclic Graph (DAG) such that for every directed edge (u, v), vertex u comes before v.

**Use cases:**
- Task scheduling with dependencies
- Build systems
- Course prerequisites
- Package dependency resolution

### Kahn's Algorithm (BFS-based)

```python
from collections import defaultdict, deque

def topological_sort_kahn(num_vertices, edges):
    """
    Topological sort using Kahn's algorithm (BFS)
    """
    graph = defaultdict(list)
    in_degree = [0] * num_vertices
    
    # Build graph and calculate in-degrees
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    # Start with vertices that have no dependencies
    queue = deque([i for i in range(num_vertices) if in_degree[i] == 0])
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        # Reduce in-degree for neighbors
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if all vertices were processed (no cycle)
    if len(result) != num_vertices:
        return []  # Graph has cycle
    
    return result

# Time: O(V + E), Space: O(V + E)
```

### DFS-based Approach

```python
def topological_sort_dfs(num_vertices, edges):
    """
    Topological sort using DFS
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    visited = set()
    stack = []
    
    def dfs(vertex):
        visited.add(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)
        
        # Add to stack after processing all descendants
        stack.append(vertex)
    
    for vertex in range(num_vertices):
        if vertex not in visited:
            dfs(vertex)
    
    return stack[::-1]  # Reverse to get correct order

# Time: O(V + E), Space: O(V + E)
```

**Course Schedule II (Real Application):**
```python
def find_order(num_courses, prerequisites):
    """
    Return course order, or empty if impossible
    """
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    result = []
    
    while queue:
        course = queue.popleft()
        result.append(course)
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return result if len(result) == num_courses else []
```

## Shortest Path Algorithms

### Dijkstra's Algorithm (Single Source Shortest Path)

Finds shortest paths from a source to all vertices in weighted graph with non-negative weights.

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    """
    Dijkstra's shortest path algorithm
    graph: dict of dict where graph[u][v] = weight
    """
    # Distance to all vertices
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    
    # Min heap: (distance, vertex)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Check neighbors
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example usage
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'C': 1, 'D': 5},
    'C': {'D': 8, 'E': 10},
    'D': {'E': 2},
    'E': {}
}

distances = dijkstra(graph, 'A')
# Result: {'A': 0, 'B': 4, 'C': 2, 'D': 9, 'E': 11}

# Time: O((V + E) log V) with binary heap
# Space: O(V)
```

**With Path Reconstruction:**
```python
def dijkstra_with_path(graph, start, end):
    """
    Dijkstra with path reconstruction
    """
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    previous = {vertex: None for vertex in graph}
    
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current == end:
            break
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return distances[end], path
```

### Bellman-Ford Algorithm

Finds shortest paths from source, handles negative weights, detects negative cycles.

```python
def bellman_ford(vertices, edges, start):
    """
    Bellman-Ford shortest path algorithm
    edges: list of (u, v, weight) tuples
    """
    # Initialize distances
    distances = {v: float('inf') for v in vertices}
    distances[start] = 0
    
    # Relax edges V-1 times
    for _ in range(len(vertices) - 1):
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
    
    # Check for negative cycles
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            return None  # Negative cycle detected
    
    return distances

# Example
vertices = ['A', 'B', 'C', 'D']
edges = [
    ('A', 'B', 4),
    ('A', 'C', 2),
    ('B', 'C', -3),
    ('C', 'D', 2),
    ('D', 'B', 1)
]

distances = bellman_ford(vertices, edges, 'A')

# Time: O(V × E)
# Space: O(V)
```

### Floyd-Warshall Algorithm (All Pairs Shortest Path)

Finds shortest paths between all pairs of vertices.

```python
def floyd_warshall(graph):
    """
    Floyd-Warshall all pairs shortest path
    graph: adjacency matrix
    """
    n = len(graph)
    dist = [row[:] for row in graph]  # Copy matrix
    
    # Try all intermediate vertices
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # If path through k is shorter
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # Check for negative cycles
    for i in range(n):
        if dist[i][i] < 0:
            return None  # Negative cycle
    
    return dist

# Example
INF = float('inf')
graph = [
    [0, 3, INF, 7],
    [8, 0, 2, INF],
    [5, INF, 0, 1],
    [2, INF, INF, 0]
]

distances = floyd_warshall(graph)

# Time: O(V³)
# Space: O(V²)
```

**Network Delay Time (LeetCode):**
```python
def network_delay_time(times, n, k):
    """
    Find time for all nodes to receive signal from node k
    times = [[u, v, w]] where signal from u to v takes w time
    """
    graph = defaultdict(dict)
    for u, v, w in times:
        graph[u][v] = w
    
    distances = dijkstra(graph, k)
    
    # Check if all nodes reachable
    if len(distances) < n:
        return -1
    
    return max(distances.values())
```

## Minimum Spanning Tree (MST)

### Kruskal's Algorithm (Edge-based)

Builds MST by adding edges in increasing order of weight, avoiding cycles.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        
        return True

def kruskal(n, edges):
    """
    Kruskal's MST algorithm
    edges: list of (weight, u, v) tuples
    """
    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0
    
    # Sort edges by weight
    edges.sort()
    
    for weight, u, v in edges:
        # If adding edge doesn't create cycle
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            
            # Stop when we have n-1 edges
            if len(mst_edges) == n - 1:
                break
    
    return mst_edges, total_weight

# Example
edges = [
    (1, 0, 1),
    (2, 0, 2),
    (3, 1, 2),
    (4, 1, 3),
    (5, 2, 3)
]

mst, weight = kruskal(4, edges)

# Time: O(E log E) for sorting
# Space: O(V)
```

### Prim's Algorithm (Vertex-based)

Builds MST by growing tree one vertex at a time.

```python
import heapq

def prim(graph, start=0):
    """
    Prim's MST algorithm
    graph: adjacency list with weights
    """
    n = len(graph)
    visited = set()
    mst_edges = []
    total_weight = 0
    
    # Min heap: (weight, current_vertex, from_vertex)
    pq = [(0, start, -1)]
    
    while pq and len(visited) < n:
        weight, vertex, from_vertex = heapq.heappop(pq)
        
        if vertex in visited:
            continue
        
        visited.add(vertex)
        
        if from_vertex != -1:
            mst_edges.append((from_vertex, vertex, weight))
            total_weight += weight
        
        # Add edges to unvisited neighbors
        for neighbor, edge_weight in graph[vertex]:
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, neighbor, vertex))
    
    return mst_edges, total_weight

# Example
graph = [
    [(1, 1), (2, 2)],  # 0 -> 1 (weight 1), 0 -> 2 (weight 2)
    [(0, 1), (2, 3), (3, 4)],
    [(0, 2), (1, 3), (3, 5)],
    [(1, 4), (2, 5)]
]

mst, weight = prim(graph)

# Time: O((V + E) log V) with binary heap
# Space: O(V + E)
```

## Union-Find (Disjoint Set)

### Introduction

Union-Find (Disjoint Set Union) is a data structure that tracks elements partitioned into disjoint sets.

**Operations:**
- **Find**: Determine which set an element belongs to
- **Union**: Merge two sets into one

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # Number of components
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x, y):
        """Check if x and y are in same set"""
        return self.find(x) == self.find(y)
    
    def get_count(self):
        """Get number of disjoint sets"""
        return self.count

# Time Complexity:
# - Find: O(α(n)) ≈ O(1) amortized (α is inverse Ackermann function)
# - Union: O(α(n)) ≈ O(1) amortized
# Space: O(n)
```

### Applications

**Problem 1: Number of Connected Components**
```python
def count_components(n, edges):
    uf = UnionFind(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    return uf.get_count()
```

**Problem 2: Redundant Connection**
```python
def find_redundant_connection(edges):
    """
    Find edge that can be removed to make tree
    """
    uf = UnionFind(len(edges) + 1)
    
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]  # This edge creates cycle
    
    return []
```

**Problem 3: Accounts Merge**
```python
def accounts_merge(accounts):
    """
    Merge accounts belonging to same person
    """
    uf = UnionFind(len(accounts))
    email_to_id = {}
    
    # Build union-find
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_to_id:
                uf.union(i, email_to_id[email])
            else:
                email_to_id[email] = i
    
    # Group emails by component
    components = defaultdict(set)
    for email, i in email_to_id.items():
        root = uf.find(i)
        components[root].add(email)
    
    # Build result
    result = []
    for i, emails in components.items():
        name = accounts[i][0]
        result.append([name] + sorted(emails))
    
    return result
```

## Advanced Graph Algorithms

### Strongly Connected Components (Kosaraju's Algorithm)

```python
def kosaraju_scc(graph):
    """
    Find strongly connected components in directed graph
    """
    n = len(graph)
    
    # Step 1: Fill order of vertices (finish times)
    visited = [False] * n
    stack = []
    
    def dfs1(v):
        visited[v] = True
        for neighbor in graph[v]:
            if not visited[neighbor]:
                dfs1(neighbor)
        stack.append(v)
    
    for i in range(n):
        if not visited[i]:
            dfs1(i)
    
    # Step 2: Create reverse graph
    reverse_graph = [[] for _ in range(n)]
    for u in range(n):
        for v in graph[u]:
            reverse_graph[v].append(u)
    
    # Step 3: DFS on reverse graph in order of decreasing finish times
    visited = [False] * n
    sccs = []
    
    def dfs2(v, component):
        visited[v] = True
        component.append(v)
        for neighbor in reverse_graph[v]:
            if not visited[neighbor]:
                dfs2(neighbor, component)
    
    while stack:
        v = stack.pop()
        if not visited[v]:
            component = []
            dfs2(v, component)
            sccs.append(component)
    
    return sccs

# Time: O(V + E), Space: O(V + E)
```

### Bridges in Graph (Tarjan's Algorithm)

```python
def find_bridges(n, connections):
    """
    Find all bridges (critical connections) in graph
    """
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    discovery_time = {}
    low = {}
    bridges = []
    time = [0]
    
    def dfs(node, parent):
        visited.add(node)
        discovery_time[node] = low[node] = time[0]
        time[0] += 1
        
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            
            if neighbor not in visited:
                dfs(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                
                # Bridge condition
                if low[neighbor] > discovery_time[node]:
                    bridges.append([node, neighbor])
            else:
                low[node] = min(low[node], discovery_time[neighbor])
    
    for node in range(n):
        if node not in visited:
            dfs(node, -1)
    
    return bridges

# Time: O(V + E), Space: O(V + E)
```

### Articulation Points

```python
def find_articulation_points(n, edges):
    """
    Find all articulation points (cut vertices) in graph
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    discovery = {}
    low = {}
    parent = {}
    ap = set()
    time = [0]
    
    def dfs(u):
        children = 0
        visited.add(u)
        discovery[u] = low[u] = time[0]
        time[0] += 1
        
        for v in graph[u]:
            if v not in visited:
                children += 1
                parent[v] = u
                dfs(v)
                
                low[u] = min(low[u], low[v])
                
                # u is AP if:
                # 1. u is root and has 2+ children
                if u not in parent and children > 1:
                    ap.add(u)
                
                # 2. u is not root and low[v] >= discovery[u]
                if u in parent and low[v] >= discovery[u]:
                    ap.add(u)
            
            elif v != parent.get(u):
                low[u] = min(low[u], discovery[v])
    
    for node in range(n):
        if node not in visited:
            parent[node] = None
            dfs(node)
    
    return list(ap)
```

## Summary

**Topological Sort:**
- Linear ordering of DAG vertices
- Kahn's (BFS) or DFS approach
- O(V + E) time complexity

**Shortest Path:**
- **Dijkstra**: Non-negative weights, single source, O((V+E) log V)
- **Bellman-Ford**: Handles negative weights, O(V×E)
- **Floyd-Warshall**: All pairs, O(V³)

**Minimum Spanning Tree:**
- **Kruskal**: Edge-based, uses Union-Find, O(E log E)
- **Prim**: Vertex-based, uses priority queue, O((V+E) log V)

**Union-Find:**
- Track disjoint sets
- Near-constant time operations with optimizations
- Used in MST, cycle detection, connectivity

**Advanced:**
- **SCC**: Kosaraju's or Tarjan's algorithm
- **Bridges**: Tarjan's algorithm
- **Articulation Points**: Critical vertices removal

**Key Patterns:**
- Use BFS for shortest path in unweighted graphs
- Use Dijkstra for weighted graphs with non-negative weights
- Use Union-Find for connectivity and cycle detection
- Topological sort for dependency resolution
- MST for network design and clustering

