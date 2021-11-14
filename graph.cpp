#include <iostream>

#include <stdlib.h>

#include "graph.h"


void Graph::printEdges() const {
	std::cout << "list of edges: \n";
	for (int i = 0; i < V; i++) {
		for (int j = i+1; j < V; j++) {
			if (adjMatrix[i*V+j] != INF)
				std::cout << "[" << i << "]-[" << j << "] weight: " << adjMatrix[i*V+j] << std::endl;
		}
	}
}

void Graph::printVertices() const {
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

void Graph::generateConnectedGraph(long long _E) {
	if(_E<V-1)
	{
		printf("number of edge is too low to create a fully connected graph!\n");
	}
	static int seed=0; //use static variable here because we want to have consistent result	between execution
	srand(seed);
	seed+=34;// modify seed becuase if we want to generate 5 different graph of same size&density to test derivation, we don't want them to be the same.

	for(int i =0 ; i< V ; i++ )//first make sure connectivity
	{	
		int already_connected=0;

		for ( int j =0 ; j<V; j++ )
		{
			if(j!=i && adjMatrix[i*V+j]!=INF )
				already_connected=1;
		}
		
		if(!already_connected)
		{
			int target=rand()%V;
			while( target == i )
				target=rand()%V;
			int w = rand() % MAX_WEIGHT + 1;
			addEdge(i, target,w);
		}
	
	}
	//then add more until #edges is  _E
	while (E < _E) {
		int v1 = rand() % V;
		int v2 = rand() % V;

		if (v1 == v2) continue;
		if (adjMatrix[v1*V+v2] != INF) continue;


		int w = rand() % MAX_WEIGHT + 1;
		addEdge(v1,v2,w);

	}
}

void Graph::generateConnectedGraphWithDensity(int density){
	long long _E = (density/100.0)*V*(V-1)/2;
	printf("generating a graph with %d vertices %d%% density(which is %d edges)...\n",V,density, _E );
	generateConnectedGraph(_E);
}
