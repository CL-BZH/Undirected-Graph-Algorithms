#ifndef GRAPH_H
#define GRAPH_H

/*
 * This file implement the random undirected graph.
 * The random undirected graph is an undirected graph where the edges between
 * nodes are randomly selected and so is the weight assiociated with each of
 * these edges.
 * One parameter of the random undirected graph is it density.
 * It is the ratio of existing edges versus the maximum number of edges
 * that the graph could have if it would be fully connected.
 * Hence it gives the probability of having an edge between 2 nodes.
 * To build a graph with a density 'd', then n = int(d x maximum_number_of_edges)
 * edges are randomly selected among all possible edges.
 */

#include <iostream>
#include <vector>
#include <numeric> // std::iota
#include <array>
#include <random>
#include <cmath>
#include <limits>

#ifdef _PRINT_MATRIX
#define _SHOW_EDGES
#include <Eigen/Dense>
using Eigen::MatrixXd;
#endif

const unsigned int DEAD_NODE{std::numeric_limits<unsigned int>::max()};
const double INFINITE_VALUE = std::numeric_limits<double>::infinity();

const unsigned int graph_default_size{50};
const double random_graph_default_density{0.2};
const std::array<double, 2> random_graph_default_range{{1.0, 10.0}};

// Vertex of a graph
struct Node {
  // Identification of the node in a graph
  // Integer value that can be used as index of an array
  unsigned int id{DEAD_NODE};

  // Value given to a node by an algorithm running on the graph
  double value{INFINITE_VALUE};

  // Define the "is greater" operator such that the node can be stored in
  // a priority queue ordered according to the node's value given by the
  // algorithm running on the graph.
  bool operator>(const Node& rhs) const {
    return value > rhs.value;
  }

  // Check is the node is valid
  bool exist() {
    return (id == DEAD_NODE)? false : true;
  }

  // For printing node's information
  friend std::ostream& operator<<(std::ostream& out, const Node& node);
};

// Printing of node information
std::ostream& operator<<(std::ostream& out, const Node& node) {
  // Just print the node's id for the moment
  out << node.id;
  return out;
}

// Edge in an undirected graph
struct Edge {
  // Index given to an edge from which the 2 connecting nodes can be found
  unsigned int index;
  // Nodes linked by the edge
  Node nodes[2];
  // Edge weight (i.e. cost, distance,...)
  double weight;

  Edge(unsigned int index, Node nodes[2], double weight):
    index{index}, nodes{nodes[0], nodes[1]}, weight{weight} {}

  // For printing edge information
  friend std::ostream& operator<<(std::ostream& out, const Edge& edge);
  
};

// Printing of edge information
std::ostream& operator<<(std::ostream& out, const Edge& edge) {
  out << "index: " << edge.index << " -> Edge nodes: ";
  out << '(' << edge.nodes[0] << ',' << edge.nodes[1] << ')';
  out << ", weight: " << edge.weight;
  return out;
}

/*
 * Undirected weighted graph class
 * We consider undirected graph where nodes have no edge to themself.
 */
struct Graph {

  /*
   * Construct an undirected graph with no edge (set edges' weights to infinity)
   * For such a graph of size s, there can be a maximum of s(s-1)/2 edges (we
   * considered that no node has an edge to itself).
   */
  Graph(unsigned int size=graph_default_size, std::string name="NO_NAME"):
    name{name}, vertices{size}, max_edges{(size * (size - 1)) / 2},
    weights(max_edges, INFINITE_VALUE) {
      if(size < 2)
	throw std::runtime_error{"The graph size has to be at least 2"};
      std::cout << "Build a graph of size " << vertices;
      std::cout << ". Maximum number of edges: " << max_edges << '\n'; 
    }

  // Initialization from another graph is done with 'move' not 'copy'
  Graph(const Graph&) = delete;

  Graph(const Graph&& graph) noexcept:
    name{graph.name}, vertices{graph.V()}, edges{graph.E()},
    max_edges{(vertices * (vertices - 1)) / 2},
    weights{graph.get_weights()} {}

  
  // Assigment is done with 'move' not 'copy'
  Graph& operator=(const Graph&) = delete;

  Graph& operator=(Graph&& graph) noexcept {
    return *this;
  }

  // Get the id of the graph
  const std::string& get_name() const {
    return name;
  }
  
  // Add an edge of cost 'weight' between nodes where the 2 nodes
  // are found thanks to the value of 'index'
  void add_edge(unsigned int index, double weight) {
    // Update the number of edges
    edges++;
    // Set the connection weight
    weights[index] = weight;

    #ifdef _SHOW_EDGES
    Node nodes[2];
    get_edge_nodes(index, nodes);
    edges_listing.push_back(Edge(index, nodes, weight));
    #endif
  }

  // Add an edge of cost 'weight' between nodes n1 and n2
  void add_edge(const Node& n1, const Node& n2, double weight) {
    if(n1.id == n2.id)
      throw std::runtime_error{"There is no edge from one node to itself"};
    
    // Get the edge index
    unsigned int index{get_index(n1, n2)};

    add_edge(index, weight);
  }

  // Remove the edge between nodes n1 and n2
  void delete_edge(const Node& n1, const Node& n2) {
    if(n1.id == n2.id)
      return;
    
    unsigned int index = get_index(n1, n2);
    weights[index] = INFINITE_VALUE;
  }

  // Get the value associated to the edge between two nodes
  // (returns infinity if there is no edge)
  double get_edge_value(const Node& n1, const Node& n2) const {
    if(n1.id == n2.id)
      return INFINITE_VALUE;
    
    unsigned int index = get_index(n1, n2);
    return weights[index];
  }
    
  // Get the number of vertices in the graph
  unsigned int V() const {
    return vertices;
  }

  // Get the number of edges in the graph
  unsigned int E() const {
    return edges;
  }
  
  // Tests whether there is an edge between node n1 and node n2
  bool adjacent(const Node& n1, const Node& n2, double *distance=nullptr) const {
    if(n1.id == n2.id) {
      return false;
    }
    
    unsigned int index = get_index(n1, n2);

    double weight{weights[index]};
    
    if (distance != nullptr)
      *distance = weight;
    
    if(weight != INFINITE_VALUE)
      return true;
    return false;
  }
  
  // List all nodes n_i such that there is an edge from node n_1 to n_i
  // and give their distances
  void get_neighbors(const Node& n1,
		     std::vector<std::pair<Node,double>>& neighbors) const {
    // Read the list of index
    Node n_i;
    double distance;
    unsigned int neighbor_count{0};

    for(unsigned int i = 0; i < vertices; i++) {
      n_i.id = i;
      if (adjacent(n1, n_i, &distance)) {
	std::pair<Node,double> neighbor{n_i, distance};		      
	neighbors.push_back(neighbor);
	neighbor_count++;
      }
    }
    //std::cout << "Node " << n1.id << " has " << neighbor_count << " neighbor" << std::endl;
  }

  // Get an access to the edges weights
  const std::vector<double>& get_weights() const {
    return weights;
  }
  
  #ifdef _SHOW_EDGES
  // Show the connections (edges)
  void show() const {
    #ifdef _PRINT_MATRIX
    // In the matrix a 0 means no edge and a non-zero value is the weight of the edge
    MatrixXd connections = MatrixXd::Ones(vertices, vertices) * INFINITE_VALUE;
    for (auto& edge : edges_listing) {
      connections(edge.nodes[0].id, edge.nodes[1].id) = static_cast<double>(edge.weight);
      connections(edge.nodes[1].id, edge.nodes[0].id) = static_cast<double>(edge.weight);
    }
    std::cout << connections << std::endl;
    #else
    for (auto& edge : edges_listing)
      std::cout << edge << '\n';
    #endif
  }
  #endif

  void set_avg_distance(double avg_dist) {
    avg_distance = avg_dist;
  }
  double get_avg_distance() {
    return avg_distance;
  }

  // Return the maximum number of edge that are possible to create in the graph
  unsigned int get_max_edges() {
    return max_edges;
  }

  // Return the graph's density. That is the ratio between the number of edges
  // and the number of edges if the graph was fully connected
  double get_density() {
    return static_cast<double>(E())/get_max_edges();
  }
  
private:
  
  // Maximum number of edges possible in the undirected graph
  // (that is the number of edges of a fully connected graph of the same size)
  const unsigned int max_edges;

  // Actual number of edges in the graph
  unsigned int edges{};
  
  // Number of node in the graph
  const unsigned int vertices;
  
  // Connection matrix represented as an array (since it is an undirectionnal graph
  // with no node connected to itself, the array takes less than half the size
  // of the matrix).
  // For exemple suppose we have the connection matrix
  // [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]]
  // where a, b, ..., p are edges' weights (e.g. if there is no connection between
  // node 1 and 0 then 'e' is equal to INFINITE_VALUE)
  // then the matrix is 'flatten' to [e, i, j, m, n, o]. 
  // And for example, edge between node 2 and 1 (row r=2 column c=1 in the matrix,
  // that is 'j') is found in the array at index ((r - 1)r)/2 + c
  std::vector<double> weights;

  // Average distance between nodes. Where the distance between 2 nodes is the
  // sum of weights of edges that form the shortest path between the 2 nodes.
  double avg_distance{INFINITE_VALUE};
  
  // Optional: give a name to the graph (default is "NO_NAME")
  const std::string name;
  
  // Compute index from nodes id. Then the weight of edge between the 2 nodes
  // is given by weights[index].
  unsigned int get_index(const Node& n1, const Node n2) const {
    if(n1.id == n2.id)
      throw std::runtime_error{"There is no edge from one node to itself"};
    // Since we store only the lower triangular part of the connectivity matrix
    // we may need to exchange the row and column index.
    unsigned int i = n1.id;
    unsigned int j = n2.id;
    if(n2.id > n1.id) {
      i = n2.id;
      j = n1.id;
    }
    return ((i - 1)*i)/2 + j;
  }

  #ifdef _SHOW_EDGES
  // This is just used for visualization of edges
  std::vector<Edge> edges_listing;
  
  // Get id of nodes forming a given edge
  void get_edge_nodes(unsigned int index, Node (&nodes)[2]) const {
    unsigned int i = static_cast<unsigned int>((1 + sqrt(1 + 8*index)) / 2);
    unsigned int j = index - (i*(i-1))/2;
    nodes[0].id = i;
    nodes[1].id = j;
  }
  #endif
  
};

/*
 * Undirected random graph class.
 * An undirected random graph IS a undirected graph, hence the inheritance.
 */
struct RandomGraph: Graph {

  RandomGraph(unsigned int size=graph_default_size,
	      double density=random_graph_default_density,
	      const std::string& name="NO_NAME",
	      std::array<double, 2> range=random_graph_default_range):
    Graph(size, name), density{density}, range{range} {

      // Build edges.
      // We want to have the number of edges equal to int(size x density)
      // and the edges have to be randomly distributed (uniform distribution).
      build_edges();
  }
  
private:
  
  // range of weights for the edges
  std::array<double, 2> range;
  
  // Graph density
  const double density;
  
  /*
   * Helper functions
   */

  /*
   * Building edges between nodes.
   * Knowing that we can have a maximum of edges max_edges (max_edges = V(V-1)/2
   * where V is the number of vetices in the graph) then we must randomly select
   * E = int(density x max_edges) edges.
   * Hence, we list all the edges by there index and pick up E of them using
   * a uniform distribution over the set of available edges (i.e. all edges but
   * the one already selected).
   */
  void build_edges() {
    unsigned long max_edges{get_max_edges()};
    // Indexes of all possible undirected edges
    std::vector<unsigned int> available_edges_indexes(max_edges);
    // fill with [0 .. max_edges)
    std::iota(available_edges_indexes.begin(), available_edges_indexes.end(), 0);
    // Total number of edges we want given the size and density
    int edges_count = static_cast<int>(density * static_cast<double>(max_edges));
     
    //Edges selection
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    for (int i = 0; i < edges_count; ++i) {
      // Randomly select an edge
      unsigned int index{random_edge(available_edges_indexes, gen)};
      // Randomly choose a weight for the selected edge
      double weight{random_weight(gen)};
      // Add the edge (i.e. set the weight with weight < INFINITE_VALUE)
      add_edge(index, weight);
    }
  }

  /*
   * Randomly select an edge in a set of available edges.
   * The set of available edges is given as a vector storing the indexes of
   * the available edges. Once an edge is selected, its index is remove from
   * from the vector (i.e. the set of available edges is updated by removing
   * the selected edge).
   */
  unsigned int random_edge(std::vector<unsigned int>& available_edges_indexes,
			   std::mt19937& gen) {
      std::uniform_int_distribution<> distr(0, available_edges_indexes.size() - 1);
      int idx{distr(gen)};
      unsigned int index = available_edges_indexes[idx];
      // Remove the edge from the available ones
      available_edges_indexes.erase(available_edges_indexes.begin() + idx);
      return index;
  }

  /*
   * Randomly select a weight in a range.
   */
  double random_weight(std::mt19937& gen) const {
    // define the range
    std::uniform_real_distribution<> distr(range[0], range[1]); 
    // Limit to 2 decimals precision
    double val{static_cast<unsigned int>(distr(gen) * 100)/100.0};
    return val;
  }

};

#endif //GRAPH_H
