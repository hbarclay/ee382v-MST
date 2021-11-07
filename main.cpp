

#include <iostream>

#include "graph.h"
#include "mstSeq.h"

#define V 20
#define E 10

int main() {
	Graph g(V, E);

	g.printEdges();

	primSeq(g.raw(), g.size());
}
