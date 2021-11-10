
#include "prim_mst_gpu.h"
#include <iostream>

#include "graph.h"

#define V 20
#define E 10

int main() {
	Graph g(V, E);
	prim_mst_hybrid(g);
	g.printEdges();
}
