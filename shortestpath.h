#ifndef SHORTESTPATH_H
#define SHORTESTPATH_H

//#include <memory>
#include <queue>
#include <string>

#include "graph.h"
#include "objectcounter.h"

/*
 * Undirected path definition
 * (let's keep everything public)
 */
struct Path {
  //Undirected path between two nodes n1 and n2
  Node n1;
  Node n2;
  // Store a route (list of nodes) between n1 and n2
  std::vector<Node> route;
  // Distance (sum of edges' weight) between n1 and n2
  double distance{INFINITE_VALUE};
  
  // Get the number of nodes forming the path
  unsigned int hop_count() {
    return route.size();
  }
  
  // Dispaly the path between the 2 nodes (start, end) if it exist
  void show() const {
    for(auto node: route)
      std::cout << node.id << "->";
    // Remove the last arrow and add the path distance (sum of edges' weights)
    std::cout << "\b\b Distance: " << distance << std::endl;
  }
  
  // Test if a route was found between the 2 nodes (start, end)
  bool is_valid() const {
    return ((!route.empty()) && (route[0].id == n1.id));
  }
  
};

/*
 * Interface for shortest path computation.
 * That is any algorithm (Dijkstra, Bellman-Ford, ...) used to compute 
 * shortest paths must comply to this interface (inheritate).
 */
struct ShortestPath {
  
  ShortestPath(const std::string& name=""): graph{nullptr}, name{name} {
    ++obj_count;
  }

  ShortestPath(const Graph& graph, const std::string& name=""):
    graph{&graph}, name{name} {
    ++obj_count;
  }
  
  virtual ~ShortestPath() {
    --obj_count;				     
  };

  // Enable the copy of the derived class such that computation can be run
  // in parallell by multiple threads.
  virtual std::unique_ptr<ShortestPath> clone(const Graph& graph) const = 0;
  
  // Compute the shortest path between 2 nodes
  virtual void compute_path(Path& path) = 0;

  // Get algorithm name
  const std::string get_name() {
    return name;
  };
  
  void set_graph(const Graph& g) {
    graph = &g;
    shortest_paths.clear();
  }
  
  const Graph* get_graph() const {
    return graph;
  }
  
  void add_known_path(const Path& path) {
    Path p{path};
    if(!path_is_known(p.n1, p.n2, p.route))
      shortest_paths.push_back(path);
  }

  void get_shortest_path(Path& path) {
    if(path_is_known(path.n1, path.n2, path.route))
      return;
    else
      compute_path(path);
  }

  // Get the current number of object alive
  unsigned int instances() const {
    return obj_count();
  }
  
private:

  // Avoid computation of already known path
  bool path_is_known(const Node& n1, const Node& n2,
		     std::vector<Node>& route) {
    for(auto shortest_path: shortest_paths) {
      if(((shortest_path.n1.id == n1.id) && (shortest_path.n2.id == n2.id)) ||
	 ((shortest_path.n2.id == n1.id) && (shortest_path.n1.id == n2.id))) {
	if(route.size() == 0)
	  route = shortest_path.route;
	return true;
      }
    }
    return false;
  }

  // Graph on which to run the algorithm
  const Graph *graph;

  // Store all shortest path between all nodes of the graph
  std::vector<Path> shortest_paths;

  // Name of the algo (optional)
  const std::string& name;
  
  // Count the number of instantiation (for debugging purpose)
  static ObjectCounter obj_count;
};

ObjectCounter ShortestPath::obj_count;

/*
 * Use of Dijkstra algorithm to compute shortest path
 */
struct DijkstraShortestPath: ShortestPath {
  
  DijkstraShortestPath(): ShortestPath(dijkstra) {}
  
  DijkstraShortestPath(const Graph& graph): ShortestPath(graph, dijkstra) {}
  
  // Create a new object
  // Note: this is safe since we use unique pointer and move it to the caller.
  virtual std::unique_ptr<ShortestPath> clone(const Graph& graph) const override {
    std::unique_ptr<DijkstraShortestPath>
      p{new DijkstraShortestPath(graph)};
    return std::move(p);
  }
  
private:

  // Compute the shortest path between 2 nodes (path.n1 and path.n2) using
  // Dijkstra algorithm
  void compute_path(Path& path) override {
    
    const Graph* graph{get_graph()};
    unsigned int graph_size{graph->V()};
    
    Node start{path.n1};
    start.value = 0.0;
    Node end{path.n2};

    // Add the starting node to the queue
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> node_queue;
    node_queue.push(start);

    // Keep track of values given to nodes
    std::vector<double> min_value(graph_size, INFINITE_VALUE);
    min_value[0] = 0.0;
    // Store selected parent of each node
    std::vector<Node> parent(graph_size, Node());
    // Tag visited nodes
    std::vector<bool> visited(graph_size, false);
    
    while (!node_queue.empty()) {
      Node current_node{node_queue.top()};
      node_queue.pop();

      // Mark the node as visited to avoid loop
      visited[current_node.id] = true;
      
      if (current_node.value > min_value[current_node.id])
	continue;

      // Visite each neighbor
      std::vector<std::pair<Node,double>> neighbors;
      graph->get_neighbors(current_node, neighbors);
      if(neighbors.size() == 0) {
	std::cout << "Node " << current_node.id << " has no neighbor" << std::endl;
	continue;
      }
      for(auto neighbor: neighbors) {
	double weight{neighbor.second};
	double value{current_node.value + weight};
	Node neighbor_node{neighbor.first};
	unsigned int neighbor_id{neighbor_node.id};
	if(visited[neighbor_id])
	  continue;
	if (value < min_value[neighbor_id]) {
	  min_value[neighbor_id] = value;
	  parent[neighbor_id] = current_node;
	  neighbor_node.value = value;
	  node_queue.push(neighbor_node);
	  //std::cout << "Queue node: " << neighbor_node.id << std::endl;
	}
      }
    }

    if(!parent[end.id].exist()) {
      // There is no path to node 'end'
      std::cout << "There is no path from node " << start.id;
      std::cout << " to node " << end.id << std::endl;
      return;
    }
    
    std::cout << "Building the path from node " << start.id;
    std::cout << " to node " << end.id << std::endl;

    path.route.push_back(end);
    
    Node previous{parent[end.id]};
    
    while(previous.value != INFINITE_VALUE) {
      //std::cout << "Previous (id, value) = ";
      //std::cout << '(' << previous.id << ", " << previous.value << ')' << std::endl;
      path.route.insert(path.route.begin(), previous); 
      previous = parent[previous.id];
    }

    // Set the path distance (sum of the weights of edges that make the path)
    path.distance = min_value[end.id];
 
    // Memoization
    if(path.route[0].id == path.n1.id) {
      add_known_path(path);
    }
  }

  // Algo name
  const std::string dijkstra{"Dijkstra"};

};

/*
 * Use of Bellman-Ford algorithm to compute shortest path
 */
struct BellmanFordShortestPath: ShortestPath {
  
  BellmanFordShortestPath(): ShortestPath(bellman_ford) {}
  
  BellmanFordShortestPath(const Graph& graph):
    ShortestPath(graph, bellman_ford) {}
  
  // Create a new object
  // Note: this is safe since we use unique pointer and move it to the caller.
  virtual std::unique_ptr<ShortestPath> clone(const Graph& graph) const override {
    std::unique_ptr<BellmanFordShortestPath>
      p{new BellmanFordShortestPath(graph)};
    return std::move(p);
  }
  
private:
  void compute_path(Path& path) override {
    //TODO
  }
  
  const std::string bellman_ford{"Bellman-Ford"};

};

#endif //SHORTESTPATH_H
