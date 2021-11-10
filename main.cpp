#include <stdio.h>
#include "prim_mst_gpu.h"
#include <iostream>

#include "graph.h"

#define V 20
#define E 10

int main() {
	Graph g(V, E);
	g.printEdges();
	printf("prim mst hybrid: %d\n", prim_mst_hybrid(g));
}
