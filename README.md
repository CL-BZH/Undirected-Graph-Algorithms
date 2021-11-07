# Undirected Graph Algorithms

This depository contains some code I wrote for the course "[C++ For C Programmers, Part A](https://www.coursera.org/learn/c-plus-plus-a)" delivered by the University of California Santa Cruz on Coursera.

The purpose is to run algorithms on undirected graphs.
Currently, Dijkstra's shortest path algorithm is implemented and it is possible to run Monte-Carlo simulation to compute an estimate of the expected shortest path length (number of nodes in the path) and an estimate of the expected shortest path cost (sum of the weights of the edges that form the path).

## Compilation

### Clean
```bash
make clean
```
### Build
```bash
make
```
As can be seen in the Makefile there are some preprocessor flags for tracing.  
_SHOW_EDGES is used to list all edges between nodes.  
_PRINT_MATRIX is used to print the connectivity matrix (where the abscence of edge between two nodes `i` and `j` is indicated by the value `inf` at position `(i,j)` and `(j,i)`.) You need to install [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) for this.


## Usage

```bash
./spmc
```

## Example of results
Below are results obtained when running a Monte-Carlo simulation for estimating the expected shortest path length and cost of a random undirected graph using Dijkstra algorithm.  
Graphs of size 50 were used with density of 0.2 and of 0.4 respectively. The range for the edges' weights is [1.0, 10.0].  
Each test ran 100 simulations in parallele using 4 threads.

```
Monte Carlo simulation for the estimation of the expected path length
 (path length = sum of edges' weight)
Graph size: 50
density: 0.2
Number of run (Optional just press enter to get the default value of 100):

The number of threads is set to 4.
(press enter if you dont want to change it.
Else enter the number of threads you want)

Run simulation with 4 threads
Duration : 27 s 325592 us

Estimation of the mean number of hop in a shortest path: 3.62242
Estimation of the mean distance estimation: 7.0165
Estimation of the variance of the mean distance: 0.0461961
```

```
Monte Carlo simulation for the estimation of the expected path length
 (path length = sum of edges' weight)
Graph size: 50
density: 0.4
Number of run (Optional just press enter to get the default value of 100):

The number of threads is set to 4.
(press enter if you don't want to change it.
Else enter the number of threads you want)

Run simulation with 4 threads
Duration : 31 s 547599 us

Estimation of the mean number of hop in a shortest path: 3.28749
Estimation of the mean distance estimation: 4.70316
Estimation of the variance of the mean distance: 0.0251515

```

## Code details
The code is splitted in four files:

* <span style="color:lime">graph.h</span>
* <span style="color:lime">shortestpath.h</span>
* <span style="color:lime">montecarlo.h</span>
* <span style="color:lime">shortestpathmontecarlo.h</span>
* <span style="color:lime">main.cpp</span>

The undirected graph is defined in graph.h.  
The algorithms for shortest path are implemented in shortestpath.h.  
montecarlo.h defines the generic interface for running Monte-Carlo simulation.  
shortestpathmontecarlo.h implements the Monte-carlo simulation for shortest path algorithms.  
main.cpp just shows example on how to run the algorithms and the Monte-carlo simulation.

Let's detail the content of each files.  


> Note: since I prefer to have the public elements of a class at the beginning of the class I use "struct" keyword instead of "class".

---
### <span style="color:lime">graph.h</span>

This file contains all the definitions needed for undirected graphs:  
* `struct Node`
* `struct Edge`
* `struct Graph`
* `struct RandomGraph: Graph`

`struct Nodes` defines the vertices of a graph.
A node that belongs to a graph is located in the graph by its id, an unsigned integer that is use to index an array. It can also have a value assigned by an algorithm using the graph.  
The '<<' operator is overloaded to print node's information.

`struct Edge` defines the edges of the graph.  
An edge has an index that uniquely identifies the two nodes that can be linked by the edge (see below).  
An edge has a weight which is set to ***infinity*** (`std::numeric_limits<double>::infinity()`) when there is no edge between the two nodes.  
The '<<' operator is overloaded to print edge's information.  

`struct Graph` defines an undirected (weighted) graph.  
The copy constructor and assignment are deleted so that we don't fill the memory with copy of a graph on which instances of an algorithm would run in parallel.  
To save memory, only the lower half of the connectivity matrix is stored as an array. If an edge exists between two nodes then the weight of that edge is store in the array at the index uniquely identifying that edge *(otherwise, **infinity** is stored)*.  
For example, the below connectivity matrix **M** of an undirected graph of size 5 (with no node connected to itself)  

|   | node 0 | node 1 | node 2 | node 3 | node 4 |
| :--------| :--------| :--------| :--------| :--------| :--------|
| **node 0** |inf  |2   |inf |6    |5  |  
| **node 1** |2    |inf |1   |inf  |9  |
| **node 2** |inf  |1   |inf |3    |inf|  
| **node 3** |6    |inf |3   |inf  |inf|  
| **node 4** |5    |9   |inf |inf  |inf|

would be stored in an array as:  
`weights = [M(1,0), M(2,0), M(2,1), M(3,0), M(3,1), M(3,2), M(4,0), M(4,1), M(4,2), M(4,3)]`  
That is:  
`weights = [2, inf, 1, 6, inf, 3, 5, 9, inf, inf]`

For a node $i$ and a node $j$ how is computed the index in the array of the edge between these two nodes?  
Well, it's pretty simple:
* swap $i$ and $j$ if $i < j$ (because we consider only the lower triangular part of the connectivity matrix)
* index `k` is equal to $\frac{i(i-1)}{2} + j$  
**Proof**:   
For a square matrix of size `2` elements of the lower triangular part can indexed this way:  
0  
For a square matrix of size `3` elements of the lower triangular part can indexed this way:  
0  
1 $\space$ 2    
For a square matrix of size `4` elements of the lower triangular part can indexed this way:  
0  
1 $\space$ 2  
3 $\space$ 4 $\space$ 5   
So, on row `i=0` we have `T(0)=1` entry. If `T(i)` is the number of entry on row `i` then the number of entries on row `i+1` is `T(i+1) = T(i) + 1`. Hence, the number of entry on row `l` is `l+1`. If we index the entries as shown above, then the index at the beginning of row `i` `(i>0)` will be the sum of number of entries from row `0` to row `i-1`, i.e. 
$\sum_{l=0}^{i-1}(l+1) = \sum_{l=1}^{i}{l} = \frac{i(i-1)}{2}$.  
Therefore the index at column `j` of row `i` is $\frac{i(i-1)}{2} + j$.

So, in the previous example, the edge between node 4 and 1 is found in the array at index $\frac{4(4-1)}{2} + 1 = 7$. The weight for this edge is `weights[7] = 9`.

In the opposite direction, having an index `k` for the array of edges weights, how can we find the 2 nodes, `i` and `j`, that are linked by this edge?  
Well, we know that $\space k = \frac{i(i-1)}{2} + j \space$ with $\space 0\le j<i$ *(swap `i` and `j` if needed)*.  

So, we have $\frac{i(i-1)}{2} \le k < \frac{i(i-1)}{2} + i$.  

Solving $\space k = \frac{i(i-1)}{2} \space $ gives $\space i = \frac{-1+\sqrt{1+8k}}{2}$  

And solving $\space k = \frac{i(i-1)}{2} + i \space $ gives $\space i = \frac{1+\sqrt{1+8k}}{2}$  
   
Therefore, $\frac{-1+\sqrt{1+8k}}{2} < i \le \frac{1+\sqrt{1+8k}}{2}$ , or again, $\frac{\sqrt{1+8k}}{2} - \frac{1}{2} < i \le \frac{\sqrt{1+8k}}{2} + \frac{1}{2}$.  
Hence, $i = \left\lfloor\frac{\sqrt{1+8k}}{2} + \frac{1}{2}\right\rfloor$ and $j = k - \frac{i(i-1)}{2}$.

Finally, the last structure defined in graph.h is `struct RandomGraph` which inherites from `Graph` since it is a graph. The two main additions are the `density` and the `range`.  
`density` defines the number of connections in the graph (see below).  
`range` defines the range for the weights to be randomly selected in.

The private function `build_edges()` is in charge of randomly selecting edges and allocating their weight.  
Here is how it works:  
We know that in an undirected graph of size `V` *(with no node connected to itself)* we can have a maximum of $\frac{V(V-1)}{2}$ edges between nodes *(then we have a fully connected graph)*.  
Hence, to get a random graph of a given $density$, we must uniformly pick up $\left\lfloor\ density \times \frac{V(V-1)}{2}\right\rfloor$ edges.  
Below is the pseudo-code for doing that *(I used // for comments)*:  
```
//Maximum number of edges in the graph
max_edges = V(V-1)/2
//Number of edges to select
edges = floor(density x max_edges)
//array of indexes of all possible edges in the graph   
available_edges =  [0,..., max_edges-1]
edge_count = 0  
selected_edges = []
selected_edges_weights = []
while edge_count++ < edges:  
    //Randomly select an edge among the available ones
    k = random.int(0,..., available_edges.size() - 1) 
    edge_index = available_edges[k]
    //Store the index of the selected edge
    selected_edges.push(edge_index)
    //Randomly select in the given range a weight for the selected edge
    weight = random.double(range)
    //Store the weight of the selected edge
    selected_edges_weight.push(weight)
    //Rebuild the list of available edges' indexes removing the selected one
    available_edges = [0,..., k - 1] + [k + 1, ..., max_edges - 1]
```  

---
### <span style="color:lime">shortestpath.h</span>
This file contains all the definitions for the shortest path algoritms.
Currently only Dijkstra's algorithm is implemented *(Bellman-Ford algorithm is in the TODO list...)*  

Structured types defined in this file:  
* `struct Path`
* `struct ShortestPath`
* `struct DijkstraShortestPath: ShortestPath`
* `struct BellmanFordShortestPath: ShortestPath` *(Not complete)*  

`struct Path` defines a path between two nodes `n1` and `n2`.

`struct ShortestPath` defines the interface for shortest path algorithm.  
Since it is an interface the functions for computing a shortest path are pure virtual functions and are override in the derived class.
```C++
virtual std::unique_ptr<ShortestPath> clone(const Graph& graph) const = 0;
virtual void compute_path(Path& path) = 0;
```
`struct DijkstraShortestPath` and `struct BellmanFordShortestPath` inherite from `struct ShortestPath` and override the functions `clone()` and `compute_path()`.

`compute_path()` in `struct DijkstraShortestPath` implement the Dijkstra algorithm. When a shortest path between two nodes is found it is added to the known path for memoization *(i.e. `struct DijkstraShortestPath` can be used for dynamic programming)*.  

---
### <span style="color:lime">montecarlo.h</span>
This file contains the generic definition for running a Monte-Carlo simulation:
* `template <typename T>  struct MonteCarlo`

It is the interface for the different type of Monte-Carlo simulations to be run.  
It is a template where `T` is used to set the algorithm (example `ShortestPath`).  
It supports multi-threading and the thread's workload is *defined* by the pure virtual function `thread_work()`.  
The base structure `Stats` can be inherited by the derived class (e.g ShortestPathMonteCarlo::Stats) and provides lock mechanism *(mutex)* for the threads to update the statistics computed in the derived class.

`std::cout` is deactivated when running the simulation but the preprocessor flag _DEBUG_MC *(see Makefile)* can be used to get some trace (e.g. number of thread spawn, number of run per thread, etc.)

---
### <span style="color:lime">shortestpathmontecarlo.h</span>
This file contains the definition for running a Shortest Path Monte-Carlo simulation aimed at computing an estimate of the espected shortest path length and cost of an undirected graph:
* `struct ShortestPathMonteCarlo: MonteCarlo<ShortestPath>`

This is where the real workload for the threads running the Monte-Carlo simulation is defined (i.e. `thread_work()` is overridden).


## License
[MIT](https://choosealicense.com/licenses/mit/)