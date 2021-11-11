#include <stdio.h>
#include "prim_mst_gpu.h"
#include <iostream>

#include "graph.h"

int main() {
	Graph g5(5);
	g5.addEdge(0,1,5);
	g5.addEdge(0,2,4);
	g5.addEdge(1,2,3);
	g5.addEdge(1,3,7);
	g5.addEdge(2,3,9);
	g5.addEdge(2,4,11);
	g5.addEdge(3,4,2); //should be 16
	
	Graph g8(8);
	g8.addEdge(0,1,4);
	g8.addEdge(0,2,6);
	g8.addEdge(0,3,16);
	g8.addEdge(1,5,24);
	g8.addEdge(2,5,23);
	g8.addEdge(2,4,5);
	g8.addEdge(2,3,8);
	g8.addEdge(3,4,10);
	g8.addEdge(3,7,21);
	g8.addEdge(4,5,18);
	g8.addEdge(4,6,11);
	g8.addEdge(4,7,14);
	g8.addEdge(5,6,9);
	g8.addEdge(6,7,7); //should be 50
	printf("prim mst hybrid on graph g8: %d\n", prim_mst_hybrid(g8));

	Graph g9(9);
	g9.generateConnectedGraph(11);

	Graph g10(10);
	g10.generateConnectedGraphWithDensity(50);

}
