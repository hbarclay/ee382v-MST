#include <stdio.h>
#include "prim_mst_gpu.h"
#include <iostream>

#include "graph.h"

#define FIXED_DENSITY 80
#define FIXED_DENSITY_COUNT 5
#define V_START 100
#define V_STEP 50

#define FIXED_V 100
#define FIXED_V_COUNT 5
#define DENSITY_START 10
#define DENSITY_STEP 20


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
	int time;
	printf("prim mst hybrid on graph g8: %d\n", prim_mst_hybrid(g8,time));

	int prim_cpu_fv[FIXED_V_COUNT]={0};
	int prim_gpu_fv[FIXED_V_COUNT]={0};
	int boru_cpu_fv[FIXED_V_COUNT]={0};
	int boru_gpu_fv[FIXED_V_COUNT]={0};
	int prim_cpu_fd[FIXED_DENSITY_COUNT]={0};
	int prim_gpu_fd[FIXED_DENSITY_COUNT]={0};
	int boru_cpu_fd[FIXED_DENSITY_COUNT]={0};
	int boru_gpu_fd[FIXED_DENSITY_COUNT]={0};


	//fixed density; increase V
	int V_fd=V_START;
	for(int i = 0; i <FIXED_DENSITY_COUNT;i++ )
	{
		Graph g(V_fd);
		g.generateConnectedGraphWithDensity(FIXED_DENSITY);
		//prim_cpu(g,prim_cpu_fd[i]);
		prim_mst_hybrid(g,prim_gpu_fd[i]);
		//boru_cpu(g,boru_cpu_fd[i]);
		//boru_gpu(g,boru_gpu_fd[i]);
		V_fd+=V_STEP;
		
		printf("prim_cpu finished at %dms\n", prim_cpu_fd[i]);
		printf("prim_gpu finished at %dms\n", prim_gpu_fd[i]);
		printf("boru_cpu finished at %dms\n", boru_cpu_fd[i]);
		printf("boru_gpu finished at %dms\n", boru_gpu_fd[i]);
	}

	//fixed V; increase density
	int V_fv=FIXED_V;
	int density_fv=DENSITY_START;
	for(int i = 0; i <FIXED_V_COUNT;i++ )
	{
		Graph g(V_fv);
		g.generateConnectedGraphWithDensity(density_fv);
		//prim_cpu(g,prim_cpu_fv[i]);
		prim_mst_hybrid(g,prim_gpu_fv[i]);
		//boru_cpu(g,boru_cpu_fv[i]);
		//boru_gpu(g,boru_gpu_fv[i]);
		density_fv+=DENSITY_STEP;
		
		printf("prim_gpu finished at %dms\n", prim_cpu_fv[i]);
		printf("prim_gpu finished at %dms\n", prim_gpu_fv[i]);
		printf("prim_gpu finished at %dms\n", boru_cpu_fv[i]);
		printf("prim_gpu finished at %dms\n", boru_gpu_fv[i]);
	}

}
