#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include "boruvka_mst_gpu.h"

// Some useful macros
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a < b ? b : a)
#define CEILING_DIV(a, b) ((a + b - 1) / b)
#define MAX_INT ((int) 0x7fffffff)

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
    int i = threadIdx.x;
    if (2 * i + 1 < len) {
        dest[i] = MIN(src[2 * i], src[2 * i + 1]);
        printf("at i = %d, min of %d and %d is %d\n", i, src[2 * i], src[2 * i + 1], dest[i]);
    } else {
        dest[i] = src[2 * i];
    }
}

// function to find the min of an array using reduce
__device__ static void findMin(int *arr, int len, int *min) {
    printf("len = %d\n", len);
    if (len == 1) {
        printf("found min: %d\n", arr[0]);
        *min = arr[0];
    } else if (len > 1) {
        int *arr2;
        int newLen = CEILING_DIV(len, 2);
        cudaMalloc((void **)&arr2, newLen * sizeof(int));
        minReduce<<<1, newLen>>>(arr, arr2, len);
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

/*
__device__ static void findMin(int *arr, int len, int *min) {
    int n = MAX_INT;
    for (int i = 0; i < len; ++i) {
        if (arr[i] < n) {
            n = arr[i];
        }
    }
    *min = n;
}
*/

// function to find the index a specified value appears in an array
__global__ static void findIdx(int *arr, int val, int *idx) {
    int i = threadIdx.x;
    //printf("Searching for %d\n", val);
    if (arr[i] == val) {
        *idx = i;
    }
}

// function to create pseudo-trees (first for loop in handout)
__global__ static void getPseudoTree(int *graph, int *T, int *parent, int numVertices, bool *exists) {
    int v = threadIdx.x;

    // for debugging
    //if (v == 0)
    if (exists[v]) {
        // find w such that (v, w) is the minimum weight edge of v.
        int *minEdgeWeight;
        int *minEdgeVertex;

        cudaMalloc(&minEdgeWeight, sizeof(int));
        cudaMalloc(&minEdgeVertex, sizeof(int));

        int *adjacencyList = graph + v * numVertices;

        // for debugging
        *minEdgeWeight = adjacencyList[0];
        *minEdgeVertex = 0;

        /*
        for (int i = 1; i < numVertices; ++i) {
            if (adjacencyList[i] < minEdgeWeight) {
                minEdgeVertex = i;
                minEdgeWeight = adjacencyList[i];
            }
        }
        */
        
        findMin(adjacencyList, numVertices, minEdgeWeight);
        findIdx<<<1,numVertices>>>(adjacencyList, *minEdgeWeight, minEdgeVertex);
        cudaDeviceSynchronize();

        printf("parent of %d is %d with weight %d\n", v, *minEdgeVertex, *minEdgeWeight);

        // update the parent
        parent[v] = *minEdgeVertex;

        // Update the minimum spanning tree. Since there are two copies of each edge in the matrix, only update the earlier one
        T[MIN(v, *minEdgeVertex) * numVertices + MAX(v, *minEdgeVertex)] = *minEdgeWeight;    // T := T U {(v, w)}

        // simple version
        //T[v * numVertices + minEdgeVertex] = minEdgeWeight;
        //T[minEdgeVertex * numVertices + v] = minEdgeWeight;
    }
}

// function to convert pseudo trees into rooted trees (second for loop in handout)
__global__ static void makeRootedTrees(int *parent, bool *exists) {
    int v = threadIdx.x;
    //printf("making rooted tree at %d\n", v);
    if ((exists[v]) && (parent[parent[v]] == v) && (v < parent[v])) {
        printf("updating parent for %d\n", v);
        parent[v] = v;
    }
}

// function to convert rooted trees into rooted stars (third for loop in handout)
__global__ static void makeRootedStars(int *parent, bool *exists) {
    int v = threadIdx.x;

    if (exists[v]) {
        while (parent[v] != parent[parent[v]]) {
            parent[v] = parent[parent[v]];
        }
    }
}

// helper function for contractRootedStars() to remove all edges connected to a specified vertex
__global__ static void removeEdges(int *graph, int v, int numVertices) {
    int u = threadIdx.x;

    graph[u * numVertices + v] = MAX_INT;
    graph[v * numVertices + u] = MAX_INT;
}

__global__ static void transferVertexEdgesToParent(int *graph, int numVertices, int *parent, int v) {
    int i = threadIdx.x;
    
    if ((graph[v * numVertices + i] != MAX_INT) && (i != parent[v])) {
        // while loop instead of if statement is a hack to fix race condition
        while ((graph[parent[v] * numVertices + parent[i]] > graph[v * numVertices + i])    
                || (graph[parent[i] * numVertices + parent[v]] > graph[i * numVertices + v])) {
            graph[parent[v] * numVertices + parent[i]] = graph[v * numVertices + i];
            graph[parent[i] * numVertices + parent[v]] = graph[i * numVertices + v];
        }
    }
}

__global__ static void transferEdgesToParent(int *graph, int *parent, int numVertices, bool *exists) {
    int v = threadIdx.x;

    // remove vertex if it is not the root of a rooted star
    if ((exists[v]) && (parent[v] != v)) {
        printf("transferring edges from vertex %d\n", v);
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

        transferVertexEdgesToParent<<<1,numVertices>>>(graph, numVertices, parent, v);
        cudaDeviceSynchronize();
    }
}

// function to contract all rooted stars
__global__ static void contractRootedStars(int *graph, int *parent, int numVertices, bool *exists) {
    int v = threadIdx.x;

    // remove vertex if it is not the root of a rooted star
    if ((exists[v]) && (parent[v] != v)) {
        printf("contracting vertex %d\n", v);
        exists[v] = false;
        // remove edges that connect to this vertex
        removeEdges<<<1,numVertices>>>(graph, v, numVertices);
        /*
        for (int i = 0; i < numVertices; ++i) {
            graph[v * numVertices + i] = MAX_INT;
            graph[i * numVertices + v] = MAX_INT;
        }
        */
    }
}

// helper function for sum()
template<typename T>
__global__ void sumReduce(T *src, T *dst, int len) {
    int i = threadIdx.x;
    if (2 * i + 1 < len) {
        dst[i] = src[2 * i] + src[2 * i + 1];
    } else {
        dst[i] = src[2 * i];
    }
}

// function to find the sum of an array using reduce
template<typename T>
__global__ void sum(T *arr, int len, int *result) {
    if (len > 1) {
        T *arr2;
        int newLen = CEILING_DIV(len, 2);
        cudaMalloc((void **)&arr2, newLen * sizeof(T));
        sumReduce<T><<<1, newLen>>>(arr, arr2, len);
        sum<T><<<1,1>>>(arr2, newLen, result);
        cudaFree(arr2);
    } else {
        *result = arr[0];
    }
}

/*
__global__ void sumByte(uint8_t *arr, int len, int *result) {
    int n = 0;
    for (int i = 0; i < len; ++i) {
        n += arr[i];
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
*/

// main function for Boruvka's algorithm
int boruvka(Graph &g) {
    // copy graph to GPU
    int *graph;
    int numVertices = g.size();
    size_t graphSize = numVertices * numVertices * sizeof(int);
    cudaMalloc((void **) &graph, graphSize);
    cudaMemcpy(graph, g.raw(), graphSize, cudaMemcpyHostToDevice);

    printf("copied graph to GPU\n");

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

    printf("Initialized global arrays\n");

    // for debugging
    void *buffer = malloc(graphSize);

    int numExistingVertices = numVertices;
    int *d_numExistingVertices;
    cudaMalloc((void **)&d_numExistingVertices, sizeof(int));
    while (numExistingVertices > 1) {
        printf("%d vertices remaining\n", numExistingVertices);
        // get pseudo-tree from the graph
        getPseudoTree<<<1,numVertices>>>(graph, T, parent, numVertices, exists);

        // convert pseudo-trees to rooted trees
        makeRootedTrees<<<1,numVertices>>>(parent, exists);

        // convert every rooted tree into a rooted star
        makeRootedStars<<<1,numVertices>>>(parent, exists);

        // contract all rooted stars into a single vertex
        transferEdgesToParent<<<1,numVertices>>>(graph, parent, numVertices, exists);
        contractRootedStars<<<1,numVertices>>>(graph, parent, numVertices, exists);

        // update number of existing vertices
        cudaMemcpy(d_numExistingVertices, &numExistingVertices, sizeof(int), cudaMemcpyHostToDevice);
        sum<<<1,1>>>((uint8_t *)exists, numVertices, d_numExistingVertices);
        cudaMemcpy(&numExistingVertices, d_numExistingVertices, sizeof(int), cudaMemcpyDeviceToHost);
    }

    // return the total weight of the minimum spanning tree
    int result;
    int *d_result;
    cudaMalloc((void **)&d_result, sizeof(int));
    sum<<<1,1>>>(T, numVertices * numVertices, d_result);
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // clean up
    cudaFree(graph);
    cudaFree(T);
    cudaFree(parent);
    cudaFree(exists);
    cudaFree(d_numExistingVertices);
    cudaFree(d_result);

    return result;
}
