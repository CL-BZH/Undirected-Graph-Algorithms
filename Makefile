#
# 'make depend' uses makedepend to automatically generate dependencies 
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# C++ compiler
CXX = g++

# define any compile-time flags
CXXFLAGS = -Wall -pthread -g -Wno-reorder

# For some debug traces
#CPPFLAGS += -D_DEBUG_MC

# For printing the connections between nodes (when _PRINT_MATRIX is not defined)
#CPPFLAGS += -D_SHOW_EDGES

# For printing the connectivity matrix (if Eigen is installed)
# this will automatically define _SHOW_EDGES
CPPFLAGS += -D_PRINT_MATRIX

# For tracing the MST algorithm
CPPFLAGS += -D_TRACE_MST

# Flag to enable/disable part of the code in main.cpp
#CPPFLAGS += -D_TEST_RANDOM_GRAPH
#CPPFLAGS += -D_TEST_DIJKTRA
#CPPFLAGS += -D_TEST_SHORTEST_PATH_MONTE_CARLO
CPPFLAGS += -D_TEST_MST

# If you installed Eigen and want to use it for showing the connectivity matrix
# add the line below where "path_to_eigen" is where to find Eigen
#INCLUDES = -I/path_to_eigen/
INCLUDES = -I../../eigen/

# define library paths in addition to /usr/lib
LFLAGS =

# define any libraries to link into executable
LIBS = 

# define the C++ source files
SRCS = main.cpp

# define the C++ object files 
OBJS = $(SRCS:.cpp=.o)

# define the executable file 
MAIN = spmc

#
# The following part of the makefile is generic

.PHONY: depend clean

all:    $(MAIN)

$(MAIN): $(OBJS) 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CFLAGS) $(INCLUDES) -c -MMD $<  -o $@ 

clean:
	$(RM) *.o *.d *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

-include *.d
