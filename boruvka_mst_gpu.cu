#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include "boruvka_mst_gpu.h"

// Some useful macros
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a < b ? b : a)
#define CEILING_DIV(a, b) ((a + b - 1) / b)
#define MAX_INT ((int) 0x7fffffff)

#define MAX_THREADS 256


// parallel memcpy for device context
/*
__global__ static void myMemcpyHelper(int * dst, int *src) {
    int i = threadIdx.x;
    dst[i] = src[i];
}

__device__ static void myMemcpy(int *dst, int *src, int length) {
    myMemcpyHelper<<<1, length>>>(dst, src);
    cudaDeviceSynchronize();
}
*/

// helper function for findMin()
__global__ static void minReduce(int *src, int *dest, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * i + 1 < len) {
        dest[i] = MIN(src[2 * i], src[2 * i + 1]);
        //printf("at i = %d, min of %d and %d is %d\n", i, src[2 * i], src[2 * i + 1], dest[i]);
    } else if (2 * i < len) {
        dest[i] = src[2 * i];
    }
}

// function to find the min of an array using reduce
__device__ static void findMin(int *arr, int len, int *min) {
    if (len == 1) {
        //printf("found min: %d\n", arr[0]);
        *min = arr[0];
    } else if (len > 1) {
        int *arr2;
        int newLen = CEILING_DIV(len, 2);

        int localNumThreads = newLen < MAX_THREADS ? newLen : MAX_THREADS;
        int localNumBlocks = CEILING_DIV(newLen, localNumThreads);

        cudaMalloc((void **)&arr2, newLen * sizeof(int));
        minReduce<<<localNumBlocks, localNumThreads>>>(arr, arr2, len);
        cudaDeviceSynchronize();
        findMin(arr2, newLen, min);
        cudaFree(arr2);
    } 
}

/*
__device__ static void findMin(int *arr, int len, int *min) {
    int *src;
    int *dest;
    cudaMalloc(&src, len * sizeof(int));
    myMemcpy(src, arr, len);
    cudaMalloc(&dest, len * sizeof(int));
    for (int size = len; size > 1; size = CEILING_DIV(size, 2)) {
        for (int i = 0; i < CEILING_DIV(size, 2); ++i) {
            if (2 * i + 1 < size) {
                dest[i] = MIN(src[2 * i], src[2 * i + 1]);
                printf("at i = %d, min of %d and %d is %d\n", i, src[2 * i], src[2 * i + 1], dest[i]);
            } else {
                dest[i] = src[2 * i];
            }
        }
        myMemcpy(src, dest, CEILING_DIV(size, 2));
        printf("size: %d\n", size);
    }
    *min = MIN(dest[0], dest[1]);
    cudaFree(src);
    cudaFree(dest);
}
*/

__device__ static void findMinSeq(int *arr, int len, int *min) {
    int n = MAX_INT;
    for (int i = 0; i < len; ++i) {
        if (arr[i] < n) {
            n = arr[i];
        }
    }
    *min = n;
}

// function to find the index a specified value appears in an array
__global__ static void findIdx(int *arr, int val, int *idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Searching for %d\n", val);
    if ((i < n) && (arr[i] == val)) {
        *idx = i;
    }
}

// function to create pseudo-trees (first for loop in handout)
__global__ static void getPseudoTree(int *graph, int *T, int *parent, int numVertices, bool *exists,
                                        int * d_numBlocks, int *d_numThreads) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if ((v < numVertices) && exists[v]) {
        // find w such that (v, w) is the minimum weight edge of v.
        int *minEdgeWeight;
        int *minEdgeVertex;

        cudaMalloc(&minEdgeWeight, sizeof(int));
        cudaMalloc(&minEdgeVertex, sizeof(int));

        int *adjacencyList = graph + v * numVertices;

        // solve weird bug with self-loops
        adjacencyList[v] = MAX_INT;

        // for debugging
        *minEdgeWeight = adjacencyList[0];
        *minEdgeVertex = 0;
        
        //findMin(adjacencyList, numVertices, minEdgeWeight);
        findMinSeq(adjacencyList, numVertices, minEdgeWeight);
        //findIdx<<<*d_numBlocks,*d_numThreads>>>(adjacencyList, *minEdgeWeight, minEdgeVertex, numVertices);
        for (int i = 0; i < numVertices; ++i) {
            if (adjacencyList[i] == *minEdgeWeight) {
                *minEdgeVertex = i;
            }
        }
        cudaDeviceSynchronize();

        //printf("parent of %d is %d with weight %d\n", v, *minEdgeVertex, *minEdgeWeight);

        // update the parent
        parent[v] = *minEdgeVertex;
        //printf("set parent of %d to %d\n", v, parent[v]);
        cudaDeviceSynchronize();

        // Update the minimum spanning tree. Since there are two copies of each edge in the matrix, only update the earlier one
        T[MIN(v, *minEdgeVertex) * numVertices + MAX(v, *minEdgeVertex)] = *minEdgeWeight;    // T := T U {(v, w)}
        //printf("adding edge with weight %d between %d and %d\n", *minEdgeWeight, v, *minEdgeVertex);

        // simple version
        //T[v * numVertices + minEdgeVertex] = minEdgeWeight;
        //T[minEdgeVertex * numVertices + v] = minEdgeWeight;
    }
}

__global__ static void getPseudoTreeSeq(int *graph, int *T, int *parent, int numVertices, bool *exists) {
    for (int v = 0; v < numVertices; ++v) {
        if (exists[v]) {
            // find w such that (v, w) is the minimum weight edge of v.
            int *minEdgeWeight;
            int *minEdgeVertex;

            cudaMalloc(&minEdgeWeight, sizeof(int));
            cudaMalloc(&minEdgeVertex, sizeof(int));

            int *adjacencyList = graph + v * numVertices;

            // solve weird bug with self-loops
            adjacencyList[v] = MAX_INT;

            // for debugging
            *minEdgeWeight = adjacencyList[0];
            *minEdgeVertex = 0;
            
            findMinSeq(adjacencyList, numVertices, minEdgeWeight);
            for (int i = 0; i < numVertices; ++i) {
                if (adjacencyList[i] == *minEdgeWeight) {
                    *minEdgeVertex = i;
                }
            }

            // update the parent
            parent[v] = *minEdgeVertex;

            // Update the minimum spanning tree. Since there are two copies of each edge in the matrix, only update the earlier one
            T[MIN(v, *minEdgeVertex) * numVertices + MAX(v, *minEdgeVertex)] = *minEdgeWeight;    // T := T U {(v, w)}
        }
    }
}

// function to convert pseudo trees into rooted trees (second for loop in handout)
__global__ static void makeRootedTrees(int *parent, bool *exists, int numVertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("hello world\n");
    cudaDeviceSynchronize();
    if (parent[v] > numVertices) {
        //printf("bad parent %d\n", parent[v]);
    }
    cudaDeviceSynchronize();
    //printf("making rooted tree at %d\n", v);
    if (v < numVertices) {
        //printf("1\n");
        cudaDeviceSynchronize();
        if (exists[v]) {
            //printf("2\n");
            cudaDeviceSynchronize();
            if (parent[parent[v]] == v) {
                //printf("3\n");
                cudaDeviceSynchronize();
                if (v < parent[v]) {
                    //printf("4\n");
                    cudaDeviceSynchronize();
        //printf("updating parent for %d\n", v);
                    parent[v] = v;
                }
            }
        }
    }
}

__global__ static void makeRootedTreesSeq(int *parent, bool *exists, int numVertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("making rooted tree at %d\n", v);
    for (int v = 0; v < numVertices; ++v) {
        if ((exists[v]) && (parent[parent[v]] == v) && (v < parent[v])) {
            //printf("updating parent for %d\n", v);
            parent[v] = v;
        }
    }
}

// function to convert rooted trees into rooted stars (third for loop in handout)
__global__ static void makeRootedStars(int *parent, bool *exists) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (exists[v]) {
        while (parent[v] != parent[parent[v]]) {
            parent[v] = parent[parent[v]];
        }
    }
}

__global__ static void makeRootedStarsSeq(int *parent, bool *exists, int numVertices) {
    for (int v = 0; v < numVertices; ++v) {
        if (exists[v]) {
            while (parent[v] != parent[parent[v]]) {
                parent[v] = parent[parent[v]];
            }
        }
    }
}

// helper function for contractRootedStars() to remove all edges connected to a specified vertex
__global__ static void removeEdges(int *graph, int v, int numVertices) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < numVertices) {
        graph[u * numVertices + v] = MAX_INT;
        graph[v * numVertices + u] = MAX_INT;
    }
}

__global__ static void transferVertexEdgesToParent(int *graph, int numVertices, int *parent, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // for an edge from vertex v to vertex i

    // parent array => parent[v] is the root of the rooted star that contains v

    // if the weight of the edge between v and i < weight of edge between parent[v] and parent[i], then transfer the edge
    
    if ((v < numVertices) && (i < numVertices) && (graph[v * numVertices + i] != MAX_INT) && (i != parent[v])) {
        // while loop instead of if statement is a hack to fix race condition
        atomicMin(&(graph[parent[v] * numVertices + parent[i]]), graph[v * numVertices + i]);
        atomicMin(&(graph[parent[i] * numVertices + parent[v]]), graph[i * numVertices + v]);
        /*
        while ((graph[parent[v] * numVertices + parent[i]] > graph[v * numVertices + i])  //   
                || (graph[parent[i] * numVertices + parent[v]] > graph[i * numVertices + v])) {
            graph[parent[v] * numVertices + parent[i]] = graph[v * numVertices + i];  
            graph[parent[i] * numVertices + parent[v]] = graph[i * numVertices + v];
        }
        */
    }
}

__global__ static void transferEdgesToParent(int *graph, int *parent, int numVertices, bool *exists, 
                                                int *d_numBlocks, int *d_numThreads) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    // remove vertex if it is not the root of a rooted star
    if ((v < numVertices) && (exists[v]) && (parent[v] != v)) {
        //printf("transferring edges from vertex %d\n", v);
        // TODO: parallelize this
        /*
        for (int i = 0; i < numVertices; ++i) {
            if ((graph[v * numVertices + i] != MAX_INT) && (i != parent[v])) {
                if (graph[parent[v] * numVertices + parent[i]] > graph[v * numVertices + i]) {
                    graph[parent[v] * numVertices + parent[i]] = graph[v * numVertices + i];
                    graph[parent[i] * numVertices + parent[v]] = graph[i * numVertices + v];
                }
            }
        }
        */

        transferVertexEdgesToParent<<<*d_numBlocks,*d_numThreads>>>(graph, numVertices, parent, v);
        cudaDeviceSynchronize();
    }
}

/*
// do the same thing as transferEdgesToParent but without calling transferVertexEdgesToParent
__global__ static void transferAllEdgesToParent(int *graph, int *parent, int numVertices, bool *exists) {
    int v = threadIdx.x / numVertices;
    int i = threadIdx.x % numVertices;

    if ((exists[v]) && (parent[v] != v)) {
        if ((graph[v * numVertices + i] != MAX_INT) && (i != parent[v])) {
            // while loop instead of if statement is a hack to fix race condition
            while ((graph[parent[v] * numVertices + parent[i]] > graph[v * numVertices + i])    
                    || (graph[parent[i] * numVertices + parent[v]] > graph[i * numVertices + v])) {
                graph[parent[v] * numVertices + parent[i]] = graph[v * numVertices + i];
                graph[parent[i] * numVertices + parent[v]] = graph[i * numVertices + v];
            }
            cudaDeviceSynchronize();
        }
    }
}
*/

__global__ static void transferEdgesSeq(int *graph, int *parent, int numVertices, bool *exists) {
    for (int v = 0; v < numVertices; ++v) {
        for (int i = 0; i < numVertices; ++i) {
            if ((exists[v]) && (parent[v] != v) && (graph[v * numVertices + i] != MAX_INT) && (i != parent[v])) {
                if ((graph[parent[v] * numVertices + parent[i]] > graph[v * numVertices + i])    
                    || (graph[parent[i] * numVertices + parent[v]] > graph[i * numVertices + v])) {
                        graph[parent[v] * numVertices + parent[i]] = graph[v * numVertices + i];
                        graph[parent[i] * numVertices + parent[v]] = graph[i * numVertices + v];
                    }
            }
        }
    }
}

// function to contract all rooted stars
__global__ static void contractRootedStars(int *graph, int *parent, int numVertices, bool *exists,
                                            int *d_numBlocks, int *d_numThreads) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    // remove vertex if it is not the root of a rooted star
    if ((v < numVertices) && (exists[v]) && (parent[v] != v)) {
        //printf("contracting vertex %d\n", v);
        exists[v] = false;
        // remove edges that connect to this vertex
        removeEdges<<<*d_numBlocks,*d_numThreads>>>(graph, v, numVertices);
        //cudaDeviceSynchronize();
        /*
        for (int i = 0; i < numVertices; ++i) {
            graph[v * numVertices + i] = MAX_INT;
            graph[i * numVertices + v] = MAX_INT;
        }
        */
    }
}

__global__ static void contractRootedStarsSeq(int *graph, int *parent, int numVertices, bool *exists) {
    for (int v = 0; v < numVertices; ++v) {
        if ((exists[v]) && (parent[v] != v)) {
            //printf("contracting vertex %d\n", v);
            exists[v] = false;
            // remove edges that connect to this vertex
            for (int i = 0; i < numVertices; ++i) {
                graph[v * numVertices + i] = MAX_INT;
                graph[i * numVertices + v] = MAX_INT;
            }
        }
    }
}

// helper function for sum()
template<typename T>
__global__ void sumReduce(T *src, T *dst, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * i + 1 < len) {
        dst[i] = src[2 * i] + src[2 * i + 1];
    } else if (2 * i < len) {
        dst[i] = src[2 * i];
    }
}

// function to find the sum of an array using reduce
template<typename T>
__global__ void sum(T *arr, int len, int *result) {
    if (len > 1) {
        T *arr2;
        int newLen = CEILING_DIV(len, 2);

        int localNumThreads = newLen < MAX_THREADS ? newLen : MAX_THREADS;
        int localNumBlocks = CEILING_DIV(newLen, localNumThreads);

        cudaMalloc((void **)&arr2, newLen * sizeof(T));
        sumReduce<T><<<localNumBlocks, localNumThreads>>>(arr, arr2, len);
        cudaDeviceSynchronize();
        sum<T><<<1,1>>>(arr2, newLen, result);
        cudaFree(arr2);
    } else {
        *result = arr[0];
    }
}


__global__ void sumBool(bool *arr, int len, int *result) {
    int n = 0;
    for (int i = 0; i < len; ++i) {
        if (arr[i]) {
            n++;
        }
        //printf("found %d\n", arr[i]);
    }
    //printf("sum: %d\n", n);
    *result = n;
}

__global__ void sumInt(int *arr, int len, int *result) {
    int n = 0;
    for (int i = 0; i < len; ++i) {
        n += arr[i];
    }
    *result = n;
}

__device__ void printTreeHelper(int *T, int numVertices) {
    for (int v = 0; v < numVertices; ++v) {
        for (int e = 0; e < numVertices; ++e) {
            if (T[v * numVertices + e] != 0) {
                printf("MST edge has weight of size %d between %d and %d\n", T[v * numVertices + e], v, e);
            }
        }
    }
}

__global__ void printTree(int *T, int numVertices) {
    printTreeHelper(T, numVertices);
}

// main function for Boruvka's algorithm
int boruvka(Graph &g, int &time) {
    int numThreads;
    int numBlocks;
    int *d_numThreads;
    int *d_numBlocks;

    // copy graph to GPU
    int *graph;
    int numVertices = g.size();
    size_t graphSize = numVertices * numVertices * sizeof(int);
    cudaMalloc((void **) &graph, graphSize);
    cudaMemcpy(graph, g.raw(), graphSize, cudaMemcpyHostToDevice);

    numThreads = numVertices < MAX_THREADS ? numVertices : MAX_THREADS;
    numBlocks = CEILING_DIV(numVertices, numThreads);
    cudaMalloc(&d_numBlocks, sizeof(int));
    cudaMalloc(&d_numThreads, sizeof(int));
    cudaMemcpy(d_numBlocks, &numBlocks, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numThreads, &numThreads, sizeof(int), cudaMemcpyHostToDevice);


    //printf("copied graph to GPU\n");

    // start timer
    // from https://stackoverflow.com/questions/1952290/how-can-i-get-utctime-in-millisecond-since-january-1-1970-in-c-language
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long startTime =
    (unsigned long long)(tv.tv_sec) * 1000 +
    (unsigned long long)(tv.tv_usec) / 1000;

    // set up array to hold mst
    int *T;
    cudaMalloc((void **) &T, graphSize);
    cudaMemset((void *) T, 0, graphSize);

    // set up parent array
    int *parent;
    cudaMalloc((void **) &parent, g.size() * sizeof(int));
    cudaMemset((void *) parent, 0, g.size() * sizeof(int));

    // set up device vector to mark vertices as existant
    bool *exists;
    cudaMalloc((void **) &exists, g.size() * sizeof(bool));
    cudaMemset((void *) exists, true, g.size() * sizeof(bool));

    //printf("Initialized global arrays\n");

    int numExistingVertices = numVertices;
    int *d_numExistingVertices;
    cudaMalloc((void **)&d_numExistingVertices, sizeof(int));
    while (numExistingVertices > 1) {
        //printf("%d vertices remaining\n", numExistingVertices);
        // get pseudo-tree from the graph
        getPseudoTree<<<numBlocks,numThreads>>>(graph, T, parent, numVertices, exists, d_numBlocks, d_numThreads);
        cudaDeviceSynchronize();
        //getPseudoTreeSeq<<<1,1>>>(graph, T, parent, numVertices, exists);
        //printf("found pseudo trees\n");

        // convert pseudo-trees to rooted trees
        makeRootedTrees<<<numBlocks,numThreads>>>(parent, exists, numVertices);
        //makeRootedTreesSeq<<<1,1>>>(parent, exists, numVertices);
        cudaDeviceSynchronize();
        //printf("made rooted trees\n");

        // convert every rooted tree into a rooted star
        makeRootedStars<<<numBlocks,numThreads>>>(parent, exists);
        //makeRootedStarsSeq<<<1,1>>>(parent, exists, numVertices);
        cudaDeviceSynchronize();
        //printf("made rooted stars\n");

        // contract all rooted stars into a single vertex
        transferEdgesToParent<<<numBlocks,numThreads>>>(graph, parent, numVertices, exists, d_numBlocks, d_numThreads);
        cudaDeviceSynchronize();
        //transferAllEdgesToParent<<<1,numVertices * numVertices>>>(graph, parent, numVertices, exists);
        //transferEdgesSeq<<<1,1>>>(graph, parent, numVertices, exists);
        //printf("transferred edges to parent\n");
        contractRootedStars<<<numBlocks,numThreads>>>(graph, parent, numVertices, exists, d_numBlocks, d_numThreads);
        //contractRootedStarsSeq<<<1,1>>>(graph, parent, numVertices, exists);
        //printf("contracted rooted stars\n");
        cudaDeviceSynchronize();

        // update number of existing vertices
        cudaMemcpy(d_numExistingVertices, &numExistingVertices, sizeof(int), cudaMemcpyHostToDevice);
        //thrust::device_ptr<char> exists_ptr(exists);
        //numExistingVertices = thrust::reduce(exists_ptr, exists_ptr + numVertices, (int)0, thrust::plus<char>());
        sumBool<<<1,1>>>(exists, numVertices, d_numExistingVertices);
        cudaMemcpy(&numExistingVertices, d_numExistingVertices, sizeof(int), cudaMemcpyDeviceToHost);
    }

    //printTree<<<1,1>>>(T, numVertices);

    // return the total weight of the minimum spanning tree
    int result;
    int *d_result;
    cudaMalloc((void **)&d_result, sizeof(int));
    //sumInt<<<1,1>>>(T, numVertices * numVertices, d_result);
    thrust::device_ptr<int> T_ptr(T);
    //thrust::device_vector<int> T_vec(T_ptr, T_ptr + numVertices * numVertices);
    thrust::device_ptr<int> thrust_d_result(d_result);
    result = thrust::reduce(T_ptr, T_ptr + numVertices * numVertices, (int)0, thrust::plus<int>());
    //cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // clean up
    cudaFree(graph);
    cudaFree(T);
    cudaFree(parent);
    cudaFree(exists);
    cudaFree(d_numExistingVertices);
    cudaFree(d_result);
    cudaFree(d_numBlocks);
    cudaFree(d_numThreads);

    // from https://stackoverflow.com/questions/1952290/how-can-i-get-utctime-in-millisecond-since-january-1-1970-in-c-language
    gettimeofday(&tv, NULL);
    unsigned long long endTime =
    (unsigned long long)(tv.tv_sec) * 1000 +
    (unsigned long long)(tv.tv_usec) / 1000;
    time = endTime - startTime;

    return result;
}
