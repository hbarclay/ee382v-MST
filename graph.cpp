#include <iostream>

#include <stdlib.h>

#include "graph.h"



void Graph::printEdges() {
	std::cout << "list of edges: \n";
	for (int i = 0; i < V; i++) {
		for (int j = i+1; j < V; j++) {
			if (adjMatrix[i*V+j] != INF)
				std::cout << "[" << i << "]-[" << j << "] weight: " << adjMatrix[i*V+j] << std::endl;
		}
	}
}

void Graph::printVertices() {
	std::cout << "list of vertices: \n";
	for (int i = 0; i < V; i++) {
		std::cout << "vertex[" << i << "] connects to: ";
		for (int j = 0; j < V; j++) {
			if (adjMatrix[i*V+j] != INF)
				std::cout << j;
		}
		std::cout << std::endl;
	}		
}

void Graph::generateGraph() {
	srand(time(NULL));

	int i = 0;
	while (i < E) {
		int v1 = rand() % V;
		int v2 = rand() % V;

		if (v1 == v2) continue;
		if (adjMatrix[v1*V+v2] != INF) continue;


		int w = rand() % MAX_WEIGHT + 1;

		adjMatrix[v1*V+v2] = w;
		adjMatrix[v2*V+v1] = w;

		i++;
	}
}
