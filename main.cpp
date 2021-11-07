
#include "shortestpathmontecarlo.h"

int main() {

  RandomGraph graph(10, 0.2);

  std::cout << "Graph density: " << graph.get_density() << std::endl;
  
  #ifdef _SHOW_EDGES
  graph.show();
  #endif

  Node n0{5};
  std::vector<std::pair<Node,double>> neighbors;
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
      std::cout << '{' << neighbor.first.id << ", " << neighbor.second << "}, ";
    }
    std::cout << std::endl;
  } else {
    std::cout << "Node " << n0.id << " has no neighbor." << std::endl;
  }

  DijkstraShortestPath dsp{graph};
  Node n1{1};
  Path path{n0, n1};
  dsp.get_shortest_path(path);
  if(path.is_valid())
    path.show();
  else
    std::cout << "No path" << std::endl;

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
}
