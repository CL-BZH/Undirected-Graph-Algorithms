#ifndef MST_H
#define MST_H

#include <memory>
#include <queue>
#include <string>
#include <sstream>

#include "graph.h"
#include "trace.h"

// Element of the MST
// Keep everything public for simplicity
struct MstElement {

  MstElement(Node* node_ptr=nullptr): node_ptr{node_ptr} {}
  
  Node* node_ptr;
  
  // Mark the node as selected once it is part of the MST
  bool selected{false};
  
    // Link the node to its parent. -1 means no parent.
  int parent{-1};

  // Define the "is greater" operator such that the element can be stored in
  // a priority queue ordered according to the node's value given by the
  // MST algorithm running on the graph.
  bool operator>(const MstElement& rhs) const {
    return *(this->node_ptr) > *(rhs.node_ptr);
  }
};

/*
 * Base class for algorithms that implement Minimum Spanning Tree
 * (Prim, Kruskal, ...)
 * This is an interface. run() is purely virtual and has to be overridden
 * in derived classes.
 */
struct MST {
  
  MST(const Graph& graph): graph{graph}, vertices{graph.V()} {
    // Instantiate the node of the graph (the graph is only the topology
    // and has no 'real' node in it)
    for(unsigned int i{0}; i < vertices; ++i) {
      MstElement mst_elem{new Node(i)}; 
      mst.push_back(mst_elem);
    }
  }
  
  virtual ~MST() {
    // Delete node instance
    for(unsigned int i{0}; i < vertices; ++i) {
      delete mst[i].node_ptr;
    }
    mst.clear();
  }

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
   * and call the script to draw the tree (Python igrap needed)
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
   * From mst creates a string that can be used to print() or draw().
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
 * (also known as Prim's algorithm)
 */
struct Prim: MST {
  
  Prim(const Graph& graph): MST(graph) {}

  ~Prim() {}
  
  void run(unsigned int root=0) override {
    
    // Store the id of the root node
    set_root(root);
    
    // Priority Queue for selecting cheapest node
    // Only pointers to mst elements (store in MST::mst) are stored in the queue
    std::priority_queue<MstElement*, std::vector<MstElement*>,
			std::greater<MstElement*>> elements_queue;
    
    
    // Keep track of number of instances of each node in the queue
    // (for tracing purpose).
    // Note: nodes are not really stored in the queue but only pointers
    // to element pointing to node.
    std::vector<unsigned int> node_instances_in_queue(graph_size(), 0);
    
    // Set the value of the root node in the MST to 0 and add it to the queue
    MstElement* root_element_ptr{&(*this)[root]};// = &mst[root]
    Node* root_node_ptr{root_element_ptr->node_ptr};
    root_node_ptr->value = 0.0;
    elements_queue.push(root_element_ptr);

    // Set number of instance in the queue of pointers to the root node to 1.
    node_instances_in_queue[root] = 1;

    trace << "\tAdd pointer to element pointing to the root node (id " << root;
    trace << ") to queue with updated value " << root_node_ptr->value;
    trace();
	  
    while (!elements_queue.empty()) {
      // Pop the first node in the queue (i.e the one with the lowest value)
      MstElement* current_element_ptr{elements_queue.top()};
      Node* current_node_ptr{current_element_ptr->node_ptr};
      elements_queue.pop();
      
      // Id of the node currently visited
      unsigned int current_node_id{current_node_ptr->id};
      
      // Decrease the number of instance of pointers to the current node
      // that are currently stored in the queue
      node_instances_in_queue[current_node_id] -= 1;
      
      if (current_element_ptr->selected == true) {
	// See the 'Note' in the for loop below to understand why we need this
	trace << "Node " << current_node_id << " already selected => skip";
	trace();
	continue;
      }

      trace << "Selected node " << current_node_id;
      trace();
      
      // Mark the node as selected. i.e. mst[current_node_id].selected = true
      current_element_ptr->selected = true;

      // Get neighbors
      std::vector<std::pair<Node, double>> neighbors;
      get_neighbors(*(current_node_ptr), neighbors);

      // Go throught all the neighbors and check if the curremtly visited node
      // gives a better path to that neighbor node.
      for(auto& neighbor: neighbors) {
	unsigned int ngbr_id{neighbor.first.id};
	double cost{neighbor.second};
	MstElement* neighbor_element_ptr = &((*this)[ngbr_id]);
	Node* neighbor_node_ptr{neighbor_element_ptr->node_ptr};
	
	if (!neighbor_element_ptr->selected) {
	  // This neighbor is not yet part of the MST (avoid loop)

	  // Get the current value of the neighbor node and check if the
	  // current node is a better parent
	  double neighbor_node_value{neighbor_node_ptr->value};
	  if(neighbor_node_value > cost) {
	    // A better path to this node is found => update its 
	    // parent in the tree with the id of the current node
	    int current_parent{neighbor_element_ptr->parent};
	    trace << "\tChange parent of node " << ngbr_id << " from ";
	    trace <<  current_parent << " to " << current_node_id;
	    trace();
	    
	    neighbor_element_ptr->parent = current_node_id;

	    // Update its value in the tree
	    neighbor_node_ptr->value = cost;

	    // Note: even if the pointer to the element pointing to that neighbor
	    // node is already in the queue we must add it again so that it has
	    // a higher pririority. This is why we have the
	    // "if (current_element_ptr->selected == true) then skip" at the
	    // beginning of the loop
	    trace << "\tAdd pointer to element pointing to node of id " << ngbr_id;
	    trace << " to queue with value updated from " << neighbor_node_value;
	    trace << " to " << neighbor_node_ptr->value;
	    trace();
	    elements_queue.push(neighbor_element_ptr);

	    // Just for tracing purpose
	    node_instances_in_queue[ngbr_id] += 1;
	  } 

	  // Just for tracing purpose
	  for(size_t i{0}; i < node_instances_in_queue.size(); ++i) {
	    if(node_instances_in_queue[i] > 0) {
	      std::string s;
	      if(node_instances_in_queue[i] == 1) {
		trace << "There is ";
		s = " ";
	      } else {
		trace << "There are ";
		s = "s ";
	      }
	      trace << node_instances_in_queue[i] << " ";
	      trace << "element's pointer" << s << "that point to node " << i;
	      trace << " in the queue.";
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
