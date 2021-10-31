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
	int V=5;
	int* graph=(int*) malloc(V*V*sizeof(int));
	
	empty_graph(graph, V);
	add_edge(graph,V, 0,2,4);
	add_edge(graph,V, 0,1,2);
	print_graph(graph, V);
	print_edge(graph,V);
	printf("graph size:%d\n",(int)sizeof(graph));
	
	int* graph_gpu;
	cudaMalloc((void**)&graph_gpu, V*V*sizeof(int));
	cudaMemcpy(graph_gpu, graph, V*V*sizeof(int), cudaMemcpyDeviceToHost);
	prim_mst_gpu<<<1,V>>>(graph_gpu,V); 
	cudaDeviceSynchronize();
	return 0;
}
