#ifndef SHORTESTPATHMONTECARLO_H
#define SHORTESTPATHMONTECARLO_H

#include <thread>
#include <mutex>
#include <chrono>

#include <cstdio>

#include "shortestpath.h"
#include "montecarlo.h"

/*
 * Structure for running simulations to estimate the expected shortest path
 * cost given a algorithm (e.g. Dijkstra) to compute shortest paths of random
 * undirected weighted graphs.
 */
template <typename Graph_T>
struct ShortestPathMonteCarlo;

template <template<typename> class Graph_T, typename Color_T>
struct ShortestPathMonteCarlo<Graph_T<Color_T>>:
  MonteCarlo<ShortestPath<Graph_T<Color_T>>> {
  
  ShortestPathMonteCarlo(ShortestPath<Graph_T<Color_T>>& algo,
			 unsigned int graph_size,
			 double graph_density, unsigned int threads=0):
    graph_size{graph_size},
    graph_density{graph_density},
    MonteCarlo<ShortestPath<Graph_T<Color_T>>>{algo, threads} {
	// Set the number of threads in case it was not given
	if(threads == 0)
	  this->set_threads(std::thread::hardware_concurrency());
      }

  
  /* 
   * Thread workload.
   * thread_number is used to identify the thread.
   * It's a number in [0, threads).
   * runs is the number of tests to run.
   */
  void thread_work(unsigned int thread_number, unsigned int runs) override {

    for(unsigned int trial{0}; trial < runs; ++trial) {
      // Get a random undirected graph for this run
      auto graph_name{std::to_string(thread_number) + " - " +
			std::to_string(trial+1)};
      RandomGraph graph{graph_size, graph_density, graph_name};
      //Get a new algo object of the same type as the one referenced by algo
      auto sp_algo{this->algo.clone(graph)};
		   		   
      MC_DEBUG_PRINT(("Thread %d - Graph %s\n",
		      thread_number,
		      sp_algo->get_graph()->get_name().c_str()));
			
      unsigned int path_counter{0};
      unsigned int no_path_counter{0};
      double avg_distance{0};
      double avg_hop{0};
      
      //for all nodes
      for(unsigned int i{0}; i < graph.V(); i++) {
	//for all other nodes
	for(unsigned int j{i + 1}; j < graph.V(); j++) {
	  Node start{i};
	  Node end{j};
	  Path path{start, end};
	  // Get the shortest path between the 2 nodes
	  sp_algo->get_shortest_path(path);
	  
	  if(path.is_valid()) {
	    // There exist a path between the 2 nodes
	    path_counter++;
	    avg_distance += path.distance;
	    avg_hop += path.hop_count();
	  } else {
	    no_path_counter++;
	  }
	}
      }
      // Update data for the stats
      if(path_counter != 0) {
	avg_hop /= path_counter;
	avg_distance /= path_counter;
	// Take the lock and update the stats
	stats.lock();
	stats.avg_distances.push_back(avg_distance);
	stats.avg_hops.push_back(avg_hop);
	// Releases the lock
	stats.unlock();
      }
    }
  }

  void compute_stats() override {
    stats.compute(this->get_runs());
  }
  
  void show_stats() {
    stats.show();
  }

private:
  
  // Algo to use to for computing the shortest path
  //ShortestPath& algo;
 
  // Graph size
  unsigned int graph_size;
  
  // Graph density
  double graph_density;

  // Statistics
  struct Stats: MonteCarlo<ShortestPath<Graph_T<Color_T>>>::Stats {
    
    // Store average distance for each trial
    std::vector<double> avg_distances;
    // Store average number of node in path for each trial
    std::vector<double> avg_hops;

    // Estimation of the mean number of hop
    double mean_hop{0};
    // Estimation of the mean distance mean and variance
    double mean_distance{0};
    double mean_distance_var{0};

    // Constructor. For the stats object to be valid the number of run
    // will have to be set
    Stats(): MonteCarlo<ShortestPath<Graph_T<Color_T>>>::Stats() {}
    
    void compute(double runs) override {
      
      if(runs == 0.0)
	throw std::runtime_error{"Number of runs was not set. (use set_runs() function)"};
      
      for(auto avg_hop: avg_hops) {
	mean_hop += avg_hop;
      }
      mean_hop /= runs;
      
      for(auto avg_distance: avg_distances) {
	mean_distance += avg_distance;
      }
      mean_distance /= runs;
      
      if (runs > 1) {
	for(auto avg_distance: avg_distances) {
	  mean_distance_var += pow((avg_distance - mean_distance), 2);
	}
	mean_distance_var = sqrt(mean_distance_var);
	mean_distance_var /= (runs - 1);
      }
    }

    void show() const override {
      std::cout << "\nEstimation of the mean number of hop in a shortest path: "
		<< mean_hop << std::endl
		<< "Estimation of the mean distance estimation: "
		<< mean_distance << std::endl
		<< "Estimation of the variance of the mean distance: "
		<< mean_distance_var << std::endl;
    }
    
  } stats;
  
};

#endif //SHORTESTPATHMONTECARLO_H
