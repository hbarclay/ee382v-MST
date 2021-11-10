#include <stdio.h>
#include "prim_mst_gpu.h"
void empty_graph(int* graph, int V){
	for(int i =0; i< V*V; i++)
		graph[i]=INT_MAX;
}

void add_edge(int* graph, int V , int v1, int v2, int w)
{
	if(v1==v2)
	{
		printf("failed to add edge: self edge not allowed");
		return;
	}
	graph[v1*V+v2]=w;
	graph[v2*V+v1]=w;
}

void print_graph(int* graph, int V)
{
	printf("\nlist of vertices: \n");
	for(int i =0; i < V; i++)
	{
		printf("vertex[%d] connects to: ", i);
		for(int j =0; j < V; j++)
		{
			if(graph[i*V+j]!=INT_MAX)
			printf("%d ", j);
		}
		printf("\n");
	}		
}

void print_edge(int* graph, int V)
{
	printf("\nlist of edges: \n");
	for(int i =0; i < V; i++)
	{
		for(int j =i+1; j < V; j++)
		{
			if(graph[i*V+j]!=INT_MAX)
			printf("[%d]-[%d] weight:%d\n",i, j, graph[i*V+j]);
		}
	}	
}

int main() {
	/*
	int V=5;
	int* graph=(int*) malloc(V*V*sizeof(int));
	
	empty_graph(graph, V);
	add_edge(graph,V, 0,1,5);
	add_edge(graph,V, 0,2,4);
	add_edge(graph,V, 1,2,3);
	add_edge(graph,V, 1,3,7);
	add_edge(graph,V, 2,3,9);
	add_edge(graph,V, 2,4,11);
	add_edge(graph,V, 3,4,2); //should be 16
	print_graph(graph, V);
	print_edge(graph,V);
	*/
	
	int V=8;
	int* graph=(int*) malloc(V*V*sizeof(int));
	
	empty_graph(graph,V);
	add_edge(graph,V, 0,1,4);
	add_edge(graph,V, 0,2,6);
	add_edge(graph,V, 0,3,16);
	add_edge(graph,V, 1,5,24);
	add_edge(graph,V, 2,5,23);
	add_edge(graph,V, 2,4,5);
	add_edge(graph,V, 2,3,8);
	add_edge(graph,V, 3,4,10);
	add_edge(graph,V, 3,7,21);
	add_edge(graph,V, 4,5,18);
	add_edge(graph,V, 4,6,11);
	add_edge(graph,V, 4,7,14);
	add_edge(graph,V, 5,6,9);
	add_edge(graph,V, 6,7,7); //should be 50

	return 0;
}
