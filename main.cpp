#include "dfs.h"
#include "mst.h"
#include "shortestpathmontecarlo.h"

int main() {

#ifdef _TEST_RANDOM_GRAPH
  // Example on how to use RandomGraph objects
  
  RandomGraph graph(10, 0.2);

  std::cout << "Graph density: " << graph.get_density() << std::endl;

  std::cout << graph << std::endl;
  
  #ifdef _SHOW_EDGES
  graph.show();
  #endif

  Node n0{5};
  std::vector<std::pair<unsigned int, double>> neighbors;
  graph.get_neighbors(n0, neighbors);

  if(neighbors.size() != 0) {
    if(neighbors.size() == 1) {
      std::cout << "Unique neighbor of node " << n0.id;
      std::cout << " is node (format is: {node id, distance})" << std::endl;
    } else {
      std::cout << "All neighbors of node " << n0.id;
      std::cout << " are nodes (format is: {node id, distance})" << std::endl;
    }
    for (auto neighbor: neighbors) {
      std::cout << '{' << neighbor.first << ", " << neighbor.second << "}, ";
    }
    std::cout << std::endl;
  } else {
    std::cout << "Node " << n0.id << " has no neighbor." << std::endl;
  }
#endif //_TEST_RANDOM_GRAPH
 
 
#ifdef _TEST_DIJKTRA
  // Example on how to use DijkstraShortestPath to run Dijstra algorithm
  // on an undirected graph
  
  RandomGraph ugraph(10, 0.2);

  DijkstraShortestPath dsp{ugraph};
  Node n1{1};
  Path path{n0, n1};
  dsp.get_shortest_path(path);
  if(path.is_valid())
    path.show();
  else
    std::cout << "No path" << std::endl;
#endif //_TEST_DIJKTRA

#ifdef _TEST_SHORTEST_PATH_MONTE_CARLO  
  // MC simulation

  std::cout << "\nMonte Carlo simulation for the estimation of the expected path";
  std::cout << " length\n (path length = sum of edges' weight)" << std::endl;

  unsigned int graph_size;
  double graph_density;
  
  std::cout << "Graph size: ";
  std::cin >> graph_size;
  std::cout << "density: ";
  std::cin >> graph_density;
  
  ShortestPathMonteCarlo spmc{dsp, graph_size, graph_density};

  std::cout << "Number of run ";
  std::cout << "(Optional just press enter to get the default value of ";
  std::cout << spmc.get_runs() << "):" << std::endl;
  
  std::string runs;
  std::cin.ignore();
  getline(std::cin, runs);
  
  std::cout << "The number of threads is set to " << spmc.get_threads() << ".\n";
  std::cout << "(press enter if you don't want to change it." << std::endl;
  std::cout << "Else enter the number of threads you want)" << std::endl;
  
  std::string threads;
  getline(std::cin, threads);
  std::cin.clear();
  
  // Set the number of threads if it was given
  if (!threads.empty())
    spmc.set_threads(atoi(threads.c_str()));

  std::cout << "Run simulation with " << spmc.get_threads() << " threads\n";
  // Run the simulation
  if (runs.empty())
      spmc.run();
  else
    spmc.run(atoi(runs.c_str()));
  
  // print the stats
  spmc.show_stats();

#endif //_TEST_SHORTEST_PATH_MONTE_CARLO  

#ifdef _TEST_MST
  // Build a graph from file
  Graph tree_graph{"./Test/hw3_sampletestdata_mst_data.txt"};
  //Graph tree_graph{"./Test/mst_test.txt"};
  
  // Use DFS to check if the graph is connected
  std::map<unsigned int, bool> visited;
  DFS dfs{tree_graph};
  
  #ifdef _SHOW_EDGES
  tree_graph.show();
  #endif
    
  dfs.get_map(0, visited);
  unsigned int sum{0};
  for(unsigned int i{0}; i < visited.size(); ++i) {
    std::cout << i << " " << visited[i] << std::endl;
    sum += visited[i];
  }
  if(sum == tree_graph.V())
    std::cout << "The graph is connected" << std::endl;
  else
    std::cout << "The graph is not connected" << std::endl;

  // Run Dijkstra-Jarnik-Prim algorithm
  Prim djp{tree_graph};
  unsigned int root{0};
  djp.run(root);
  djp.print();
  djp.draw();
  
#endif //_TEST_MST
  
}
