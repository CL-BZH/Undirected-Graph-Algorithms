import sys
from igraph import *


lines = sys.argv[1:][0]
lines = lines.rstrip(',')
lines = lines.split(',')

root_node = int(lines[0])
vertices = int(lines[1])
edges = []
weights = []

# Create graph
graph = Graph(directed=False)

graph.add_vertices(vertices) 
print(f"Build a tree with {len(graph.vs)} vertices")

# Add ids and labels to vertices
for i in range(len(graph.vs)):
    graph.vs[i]["id"]= i
    graph.vs[i]["label"]= str(i)


# Get the edges and their weights
for line in lines[2:]:
    line = line.split()
    n1 = int(line[0])
    n2 = int(line[1])
    edges += [(n1, n2)]
    weights += [float(line[2])]

# Add the edges
graph.add_edges(edges)

# Add weights and edge labels
graph.es['weight'] = weights
graph.es['label'] = weights

visual_style = {}

#out_name = "graph.png"

# Set bbox and margin
visual_style["bbox"] = (650, 400)
#visual_style["bbox"] = (200, 200)
visual_style["margin"] = 27

# Set vertex colours
visual_style["vertex_color"] = 'white'

# Set vertex size
visual_style["vertex_size"] = 25

# Set vertex lable size
visual_style["vertex_label_size"] = 18

# Don't curve the edges
visual_style["edge_curved"] = False

# Set the layout
my_layout = graph.layout_reingold_tilford(mode="in", root=root_node)
visual_style["layout"] = my_layout

# Plot the graph
plot(graph, **visual_style)
