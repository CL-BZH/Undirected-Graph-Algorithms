#ifndef MST_H
#define MST_H

#include <queue>
#include <string>
#include <sstream>

#include "graph.h"
#include "trace.h"

// Element of the MST
struct MstElement {
  // Value given to a node
  unsigned int value{DEAD_NODE};
  // Mark the node as selected once it is part of the MST
  bool selected{false};
    // Link the node to its parent
  int parent{-1};
};

/*
 * Base class for algorithms that implement Minimum Spanning Tree
 * (Prim, Kruskal, ...)
 * This is an interface. run() is purely virtual and has to be overridden
 * in derived classes.
 */
struct MST {
  
  MST(const Graph& graph): graph{graph}, vertices{graph.V()},
			   mst(vertices) {}
  
  virtual ~MST() { mst.clear();};

  virtual void run(unsigned int root=0 /*Id of the node to select as root*/) = 0;

  /*
   * Print the MST in format:
   * First line: nb of vertice
   * Other lines: edge weight
   * where edge is of the form n1 n2 the ids of the nodes forming the edge
   * and weight is the weight of the edge.
   * E.g:
   * 5
   * 3 0 6
   * 0 1 2
   * 1 2 3
   * 1 4 5
   */
  void print() {
    // Print the number of vertices and then Edge Weight
    std::string str;
    std::cout << mst_string(str);
  }

  /*
   * Generate a command for the Python script draw_mst.py in Tools
   * draw the tree (Python igrap has to be installed)
   */
  void draw() {
    std::string str;
    mst_string(str, ',');
    // Create the command for the Pyhton script
    std::string command = "python ./Tools/draw_mst.py \'";
    command += std::to_string(tree_root) + ",";
    command += str + '\'';
    // Run the Python script
    system(command.c_str());
  }

  // Set a root node for the spanning tree (by default it is node 0)
  void set_root(unsigned int root) {
    if(root >= vertices) {
      std::stringstream err_sstr;
      err_sstr << "Invalid root index " << root << ". Choose a value less than";
      err_sstr << " the number of vertices (" << vertices << ")" << std::endl;
      throw std::runtime_error{err_sstr.str()};
    }
    tree_root = root;
  }

  unsigned int graph_size() {
    return vertices;
  }
  
  // Get all the neighbors of a given node n1
  void get_neighbors(const Node& n1,
		     std::vector<std::pair<Node,double>>& neighbors) {
    neighbors.clear();
    graph.get_neighbors(n1, neighbors);
  }
  
  // Accessor
  const MstElement& operator[](unsigned int index) const {
    if(index >= vertices) {
      std::stringstream err_sstr;
      err_sstr << "Invalid index " << index << std::endl;
      throw std::runtime_error{err_sstr.str()};
    }
    return mst[index]; 
  }
  // Mutator
  MstElement& operator[](unsigned int index)  {
    if(index >= vertices) {
      std::stringstream err_sstr;
      err_sstr << "Invalid index " << index << std::endl;
      throw std::runtime_error{err_sstr.str()};
    }
    return mst[index];
  }

protected:
  // For tracing
  Trace trace;
  
private:
  // Reference to the graph on which to run the MST algo
  const Graph& graph;
  
  // Number of nodes in the graph
  unsigned int vertices;

  // Node selected as root of the tree
  unsigned int tree_root{0};

  // Storage of the MST. Nodes are identified by there index.
  std::vector<MstElement> mst;
  
  /*
   * From mst creates a string that can be used bu print() or draw().
   * By default creates a string with line splitted by newline caratere ('\n').
   * For drawing, newlines are replaced by ','.
   */
  const std::string& mst_string(std::string& str, char line_split='\n') {
    std::stringstream sstr;
    sstr << vertices << line_split;
    for (unsigned int k{0}; k < vertices; ++k) {
      unsigned int parent_id{static_cast<unsigned int>(mst[k].parent)};
      if(parent_id != DEAD_NODE) {
	sstr << parent_id << " " << k << " ";
	sstr << graph.get_edge_value(k, parent_id) << line_split;
      } else if (k != tree_root) {
	// Found a node with no parent despite not being the root
	std::stringstream err_sstr;
	err_sstr << "Check algo. Node " << k << " has no parent!" << std::endl;
	throw std::runtime_error{err_sstr.str()};
      }
    }
    str = sstr.str();
    return str;
  }
    
};


/*
 * Structure for Dijkstra-Jarnik-Prim's algorithm
 * (also known as Prim's algorythm)
 */
struct Prim: MST {
  
  Prim(const Graph& graph): MST(graph) {}

  ~Prim() {}
  
  void run(unsigned int root=0) override {
    
    // Store the id of the root node
    set_root(root);
    
    // Priority Queue for selecting cheapest node
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> node_queue;
    std::vector<Node*> nodes(graph_size(), nullptr);

    // Keep track of number of instances of each node in the queue
    // (for tracing purpose)
    std::vector<unsigned int> node_instances_in_queue(graph_size(), 0);
    
    // Add the root node in the queue
    Node node{root};
    node.value = 0.0;
    node_queue.push(node);
    nodes[root] = &node;
    
    node_instances_in_queue[root] = 1;
    
    while (!node_queue.empty()) {
      // Pop the first node in the queue (i.e the one with the lowest value)
      Node current_node{node_queue.top().id};
      node_queue.pop();

      node_instances_in_queue[current_node.id] -= 1;
      
      if ((*this)[current_node.id].selected == true) {
	trace << "Node " << current_node.id << " already selected => skip";
	trace();
	continue;
      }

      trace << "Selected node " << current_node.id;
      trace();
      
      // Mark the node as selected. i.e. mst[current_node.id].selected = true
      (*this)[current_node.id].selected = true;

      // Get neighbors
      std::vector<std::pair<Node, double>> neighbors;
      get_neighbors(current_node, neighbors);
      
      for(auto& neighbor: neighbors) {
	unsigned int ngbr_id{neighbor.first.id};
	double cost{neighbor.second};
	if (!(*this)[ngbr_id].selected) {
	  // This neighbor is not yet part of the MST (avoid loop)
	  if((*this)[ngbr_id].value > cost) {
	    // A better path to this node is found => update its 
	    // parent in the tree
	    int current_parent{(*this)[ngbr_id].parent};
	    unsigned int new_parent{current_node.id};
	    trace << "\tChange parent of node " << ngbr_id << " from ";
	    trace <<  current_parent << " to " << new_parent;
	    trace();
	    
	    (*this)[ngbr_id].parent = current_node.id;

	    // Update its value in the tree and also in the neighbor since
	    // we will add it to the queue.
	    (*this)[ngbr_id].value = cost;
	    neighbor.first.value = cost;
	  }
	  trace << "\tAdd node " << ngbr_id << " to queue with value ";
	  trace << neighbor.first.value;
	  trace();
	  node_queue.push(neighbor.first);

	  // Just for tracing
	  node_instances_in_queue[neighbor.first.id] += 1;
	  for(size_t i{0}; i < node_instances_in_queue.size(); ++i) {
	    if(node_instances_in_queue[i] > 0) {
	      trace << "Node " << i << " is " << node_instances_in_queue[i];
	      trace << " times in the queue";
	      trace();
	    }
	  }
	}
      }
    }
  }
};

/*
 * Structure for the Kruskal's algorithm
 */
struct Kruskal: MST {
  
  Kruskal(const Graph& graph): MST(graph) {}

  ~Kruskal() {}
  
  void run(unsigned int root=0) override {
    
    // Store the id of the root node
    set_root(root);

    // TO DO
  }
  
};
    
#endif //MST_H
