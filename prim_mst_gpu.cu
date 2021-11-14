#include <cuda.h>
#include <stdio.h>
#include <limits.h>
#include "graph.h"
#include "min_heap.h"
#include <cuda_runtime_api.h>
#include "prim_mst_gpu.h"
#include <sys/time.h>

#include <thrust/scan.h>
#include <thrust/device_vector.h>

int* graph_gpu;
int* d_gpu;
int* isFixed_gpu;
int* R_gpu;
int* R_next_gpu;
int* Q_gpu;
int* T_gpu;
int* parent_gpu;
int* MWE_gpu;
int* scan_gpu;

int* scan_result_gpu;



int* d;
int* isFixed;
int* R;
__managed__ int R_size;
int* R_next;
int* Q;
int* T;
int* parent;
int* MWE;


__global__ void processEdge1GPU(
int*graph, 
int V,
int* d,
int* isFixed,
int* R,
int* R_next,
int* Q,
int* T,
int* parent,
int* MWE
)
{
	int global_idx=blockDim.x * blockIdx.x + threadIdx.x;
	int z = global_idx/V;
	int k = global_idx-z*V;
	
	cudaDeviceSynchronize();
	if(z<R_size && k<V) //explore every vertex in R
	{
		if (graph[R[z]*V+k]!= INT_MAX && isFixed[k]==0 && R[z]!=k )
		{
			
			//printf("checking edge R[z]:%d-k:%d\n", R[z], k);
			if(graph[R[z]*V+k] == MWE[k] )
			{
				isFixed[k]=true;
				T[k]=graph[R[z]*V+k];
				R_next[k]=1;
				//printf("vertex %d (%d-%d,%d) is now fixed due to MWE\n",k,k,R[z],graph[R[z]*V+k]) ;

			}
			else if (d[k]> graph[R[z]*V+k])
			{
				//printf("add %d-%d,%d to Q (%d<%d)\n", R[z], k, graph[R[z]*V+k], graph[R[z]*V+k],d[k] );
				//d[k] = graph[R[z]*V+k]; //bad code: atomicity will be violated
				//parent[k] = R[z]; //bad code: atomicity will be violated

				int old = atomicMin(&d[k], graph[R[z]*V+k]);
				cudaDeviceSynchronize();//wait for the final d[k] be computed
				if(d[k]==graph[R[z]*V+k]) // if different, means someone else has lower d, do nothing
					parent[k]=R[z];	
				Q[k]=1;
			}
		}	
	}
}

void processEdge1(int*graph, int V)
{
	for(int z = 0 ; z < R_size; z++) //explore every vertex in R
	{
		for(int k=0; k<V; k++) // explore all edge at vertex R[z] 
		{
			if (graph[R[z]*V+k]!= INT_MAX && isFixed[k]==0 && R[z]!=k )
			{
				//printf("checking edge R[z]:%d-k:%d\n", R[z], k);
				if(graph[R[z]*V+k] == MWE[R[z]] || graph[R[z]*V+k] == MWE[k] )
				{
					isFixed[k]=true;
					T[k]=graph[R[z]*V+k];
					R_next[k]=1;
					//printf("vertex %d (%d-%d,%d) is now fixed due to MWE\n",k,k,R[z],graph[R[z]*V+k]) ;
				}
				else if (d[k]> graph[R[z]*V+k])
				{
					d[k] = graph[R[z]*V+k];
					parent[k] = R[z];
					Q[k]=1;
				}
			}	
		}
	}
}


void var_init(int* graph, int V)
{
	d=(int*)malloc(V*sizeof(int));
	isFixed=(int*)malloc(V*sizeof(int));
	Q=(int*)malloc(V*sizeof(int));
	R=(int*)malloc(V*sizeof(int));
	R_next=(int*)malloc(V*sizeof(int));
	T=(int*)malloc(V*sizeof(int));
	parent=(int*)malloc(V*sizeof(int));
	MWE=(int*)malloc(V*sizeof(int));

	for(int i =0; i<V;i++){
		d[i]=INT_MAX;
		isFixed[i]=0;	
		MWE[i]=INT_MAX;
		R[i]=0;
		R_size=0;
		R_next[i]=0;
		Q[i]=0;
		T[i]=0;
		parent[i]=-1;

	}
	//find MWE
	for(int i = 0; i < V; i++ )
	{
		for(int j =0; j <V; j++)
		{
			if(graph[i*V+j] < MWE[i] )
				MWE[i]= graph[i*V+j];
		}
	}
}

void var_free(){
	free(d);
	free(isFixed);
	free(Q);
	free(R);
	free(R_next);
	free(T);
	free(parent);
	free(MWE);
}

void ExclusivePrefixSum(int* input, int* output, int size)
{
	output[0]=0;
	// Adding present element
	// with previous element
	for (int i = 1; i < size; i++)
		output[i] = output[i-1] + input[i-1];
}

__global__ void Q_zeroes_out(int*graph, int V, int* Q)
{
	int global_idx=blockDim.x * blockIdx.x + threadIdx.x;
	if(global_idx<V)
		Q[global_idx]=0;
}
__global__ void hash_to_vector(int*graph, int V, int* R, int*R_next, int*scan_result )
{
	int global_idx=blockDim.x * blockIdx.x + threadIdx.x;
	if(global_idx==1)
	{
		R_size=scan_result[V-1]+R_next[V-1];
	}
	if(global_idx<V)
	{
		if(R_next[global_idx]==1)
			R[scan_result[global_idx]]=global_idx;
	}
	cudaDeviceSynchronize();
	R_next[global_idx]=0; //reinitialize R_next

}


void get_next_R_gpu(int* graph, int V, int num_block, int num_thread)
{
	thrust::device_ptr<int> R_next_thrust(R_next_gpu);
	thrust::device_ptr<int> scan_result_thrust(scan_result_gpu);
	thrust::exclusive_scan(R_next_thrust, R_next_thrust + V, scan_result_thrust);
	hash_to_vector<<<num_block ,num_thread>>>(graph_gpu, V, R_gpu, R_next_gpu, scan_result_gpu);
	cudaDeviceSynchronize();
}


void get_next_R(int* graph, int V)
{
	int temp[V];
	ExclusivePrefixSum(R_next,&temp[0], V);
 	R_size=temp[V-1]+R_next[V-1];
	for(int i =0 ; i<V; i ++ ){
		if(R_next[i]==1)
			R[temp[i]]=i;
	}
	
	//printf("R: ");
	for(int i =0; i <R_size; i++)
	{
		//printf("%d ", R[i]);
	}
	//printf("\n");
	
	//zero out the R_next
	for(int i =0; i<V;i++)
		R_next[i]=0;
}

void gpu_var_init(int* graph, int V){
	cudaMalloc((void**)&graph_gpu,V*V*sizeof(int));
	cudaMalloc((void**)&d_gpu,V*sizeof(int));
	cudaMalloc((void**)&isFixed_gpu,V*sizeof(int));
	cudaMalloc((void**)&Q_gpu,V*sizeof(int));
	cudaMalloc((void**)&R_gpu,V*sizeof(int));
	cudaMalloc((void**)&R_next_gpu,V*sizeof(int));
	cudaMalloc((void**)&T_gpu,V*sizeof(int));
	cudaMalloc((void**)&parent_gpu,V*sizeof(int));
	cudaMalloc((void**)&MWE_gpu,V*sizeof(int));

	cudaMalloc((void**)&scan_result_gpu,V*sizeof(int));

	cudaMemcpy(graph_gpu, graph, V*V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gpu, d, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(isFixed_gpu, isFixed, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Q_gpu, Q, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(R_gpu, R, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(R_next_gpu, R_next, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(T_gpu, T, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(parent_gpu, parent, V*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(MWE_gpu, MWE, V*sizeof(int), cudaMemcpyHostToDevice);

}

void gpu_var_free()
{
	cudaFree(graph_gpu);
	cudaFree(d_gpu);
	cudaFree(isFixed_gpu);
	cudaFree(Q_gpu);
	cudaFree(R_gpu);
	cudaFree(R_next_gpu);
	cudaFree(T_gpu);
	cudaFree(parent_gpu);
	cudaFree(MWE_gpu);

	cudaFree(scan_result_gpu);
}

int prim_mst_hybrid(Graph& g, int& time)
{
	struct timeval start;
	struct timeval end;

	int* graph = g.raw();
	int V = g.size();
	var_init(graph, V);
	d[0]=0;
	MinHeap H(V);
	H.insert(0,d[0]);
	isFixed[0]=1;
	gpu_var_init(graph, V);
	
	gettimeofday(&start, NULL);
	while(!H.empty())
	{
		MinHeapNode min = H.extractMin();
		int j = min.val;
		int j_weight=min.key;
		R[R_size]=j;
		R_size++;
		
		if(parent[j] != -1 && !isFixed[j] ){
			isFixed[j] = true;
			//printf("vertex %d (%d-%d,%d) is now fixed due to min cut\n",j,j,parent[j],j_weight );
			T[j]=graph[j*V+parent[j]];
		}

		while(R_size!=0){
			int block_dim = (V*R_size<512)?V*R_size:512 ;
			int grid_dim = (V*R_size+block_dim-1)/block_dim; //ceiling of V*R_size/block_dim
		
			cudaMemcpy(d_gpu, d, V*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(isFixed_gpu, isFixed, V*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(R_gpu, R, V*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(T_gpu, T, V*sizeof(int), cudaMemcpyHostToDevice);

			processEdge1GPU<<<grid_dim,block_dim>>>(
					graph_gpu,
					V, 
					d_gpu,
					isFixed_gpu,
					R_gpu,
					R_next_gpu,
					Q_gpu,
					T_gpu,
					parent_gpu,
					MWE_gpu
					);
			cudaDeviceSynchronize();
			
			cudaMemcpy(parent, parent_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(d, d_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(T, T_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(Q, Q_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(isFixed, isFixed_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			
			
			//printf("Q: ");
			for(int z =0 ; z<V; z++){
				
				if( !isFixed[z] && Q[z]==1  )
				{
					//printf("%d(%d) ",z,d[z] );
					H.insertOrDecrease(d[z],z);
				}
				Q[z]=0;
			}
			//printf("\n");
			
			block_dim = (V<512)?V:512 ;
			grid_dim = (V+block_dim-1)/block_dim; 

			get_next_R_gpu(graph,V,grid_dim,block_dim);
			cudaMemcpy(R, R_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			Q_zeroes_out<<<grid_dim,block_dim>>>(graph_gpu, V, Q_gpu);
			cudaDeviceSynchronize();
		}

	}
	gettimeofday(&end,NULL);
	unsigned long long startms=(unsigned long long)(start.tv_sec) * 1000 + (unsigned long long)(start.tv_usec) / 1000;
	unsigned long long endms=(unsigned long long)(end.tv_sec) * 1000 + (unsigned long long)(end.tv_usec) / 1000;
	
	time=(int)(endms-startms);

	int MST_total_weight=0;
	//printf("prim_gpu T:");
	for(int i =0; i < V; i++)
	{
		//printf("%d ", T[i]);
		MST_total_weight+=T[i];
	}
	//printf("\n");
	var_free();
	gpu_var_free();
	return MST_total_weight;
}
int prim_mst_hybrid_bad_scan(Graph& g, int& time)
{
	struct timeval start;
	struct timeval end;

	int* graph = g.raw();
	int V = g.size();
	var_init(graph, V);
	d[0]=0;
	MinHeap H(V);
	H.insert(0,d[0]);
	isFixed[0]=1;
	gpu_var_init(graph, V);
	
	gettimeofday(&start, NULL);
	while(!H.empty())
	{
		MinHeapNode min = H.extractMin();
		int j = min.val;
		int j_weight=min.key;
		R[R_size]=j;
		R_size++;
		
		if(parent[j] != -1 && !isFixed[j] ){
			isFixed[j] = true;
			//printf("vertex %d (%d-%d,%d) is now fixed due to min cut\n",j,j,parent[j],j_weight );
			T[j]=graph[j*V+parent[j]];
		}

		while(R_size!=0){
			int block_dim = (V*R_size<512)?V*R_size:512 ;
			int grid_dim = (V*R_size+block_dim-1)/block_dim; //ceiling of V*R_size/block_dim
		
			cudaMemcpy(d_gpu, d, V*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(isFixed_gpu, isFixed, V*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(Q_gpu, Q, V*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(R_gpu, R, V*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(R_next_gpu, R_next, V*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(T_gpu, T, V*sizeof(int), cudaMemcpyHostToDevice);

			processEdge1GPU<<<grid_dim,block_dim>>>(
					graph_gpu,
					V, 
					d_gpu,
					isFixed_gpu,
					R_gpu,
					R_next_gpu,
					Q_gpu,
					T_gpu,
					parent_gpu,
					MWE_gpu
					);
			cudaDeviceSynchronize();
			
			cudaMemcpy(parent, parent_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(d, d_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(T, T_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(Q, Q_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(isFixed, isFixed_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(R_next, R_next_gpu, V*sizeof(int), cudaMemcpyDeviceToHost);
	
			//printf("Q: ");
			for(int z =0 ; z<V; z++){
				
				if( !isFixed[z] && Q[z]==1  )
				{
					//printf("%d(%d) ",z,d[z] );
					H.insertOrDecrease(d[z],z);
				}
				Q[z]=0;
			}
			//printf("\n");
			get_next_R(graph, V);
		}

	}
	gettimeofday(&end,NULL);
	unsigned long long startms=(unsigned long long)(start.tv_sec) * 1000 + (unsigned long long)(start.tv_usec) / 1000;
	unsigned long long endms=(unsigned long long)(end.tv_sec) * 1000 + (unsigned long long)(end.tv_usec) / 1000;
	
	time=(int)(endms-startms);

	int MST_total_weight=0;
	printf("prim_gpu T: ");
	for(int i =0; i < V; i++)
	{
		printf("%d ", T[i]);
		MST_total_weight+=T[i];
	}
	printf("\n");
	var_free();
	gpu_var_free();
	return MST_total_weight;
}



int prim_mst_simulation(int* graph, int V)
{

	var_init(graph, V);	
	d[0]=0;
	MinHeap H(V);
	H.insert(0,d[0]);
	isFixed[0]=1;
	while(!H.empty())
	{
		int j = H.extractMin().val;
		R[R_size]=j;
		R_size++;
		
		if(parent[j] != -1 && !isFixed[j] ){
			isFixed[j] = true;
			printf("vertex %d is now fixed due to min cut\n",j );
			T[j]=graph[j*V+parent[j]];
		}
		while(R_size!=0){
			processEdge1( graph, V);
			
			//printf("Q: ");
			for(int z =0 ; z<V; z++){
				
				if( !isFixed[z] && Q[z]==1  )
				{
					//printf("%d ",z);
					H.insertOrDecrease(d[z],z);
				}
				Q[z]=0;
			}
			//printf("\n");
			get_next_R(graph, V);
		}
	}
	int MST_total_weight=0;
	for(int i =0; i < V; i++)
		MST_total_weight+=T[i];
	return MST_total_weight;
}
