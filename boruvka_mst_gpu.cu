#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include "boruvka_mst_gpu.h"

// Some useful helper functions
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a < b ? b : a)
#define CEILING_DIV(a, b) ((a + b - 1) / b)
#define MAX_INT ((int) 0x7fffffff)

// helper function for findMin()
__global__ static void minReduce(int *src, int *dest, int len) {
    int i = threadIdx.x;
    if (2 * i + 1 < len) {
        dest[i] = MIN(src[2 * i], src[2 * i + 1]);
    } else {
        dest[i] = src[2 * i];
    }
}

// function to find the min of an array
__device__ static void findMin(int *arr, int len, int *min) {
    if (len > 1) {
        int *arr2;
        int newLen = CEILING_DIV(len, 2);
        cudaMalloc((void **)&arr2, newLen * sizeof(int));
        minReduce<<<1, newLen>>>(arr, arr2, len);
        findMin(arr2, newLen, min);
        cudaFree(arr2);
    } else {
        *min = arr[0];
    }
}

// function to find the index a specified value appears in an array
__global__ static void findIdx(int *arr, int val, int *idx) {
    int i = threadIdx.x;
    if (arr[i] == val) {
        *idx = i;
    }
}

// function to create pseudo-trees (first for loop in handout)
__global__ static void getPseudoTree(int *graph, int *T, int *parent, int numVertices, bool *exists) {
    int v = threadIdx.x;

    if (exists[v]) {
        // find w such that (v, w) is the minimum weight edge of v.
        int *minEdgeWeight;
        int *minEdgeVertex;
        cudaMalloc((void **)&minEdgeWeight, sizeof(int));
        cudaMalloc((void **)&minEdgeVertex, sizeof(int));

        int *adjacencyList = graph + v * numVertices;
        
        findMin(adjacencyList, numVertices, minEdgeWeight);
        findIdx<<<1,numVertices>>>(adjacencyList, *minEdgeWeight, minEdgeVertex);

        // update the parent
        parent[v] = *minEdgeVertex;

        // Update the minimum spanning tree. Since there are two copies of each edge in the matrix, only update the earlier one
        T[MIN(v, *minEdgeVertex) * numVertices + MAX(v, *minEdgeVertex)] = *minEdgeWeight;    // T := T U {(v, w)}
    }
}

// function to convert pseudo trees into rooted trees (second for loop in handout)
__global__ static void makeRootedTrees(int *parent, bool *exists) {
    int v = threadIdx.x;
    if ((exists[v]) && (parent[parent[v]] == v) && (v < parent[v])) {
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

// function to contract all rooted stars into single vertices
__global__ static void contractRootedStars(int *graph, int *parent, int numVertices, bool *exists) {
    int v = threadIdx.x;

    // remove vertex if it is not the root of a rooted star
    if ((exists[v]) && (parent[v] != v)) {
        exists[v] = false;
        // remove edges that connect to this vertex
        removeEdges<<<1,numVertices>>>(graph, v, numVertices);
    }
}

// helper function fo sum()
template<typename T>
__global__ void sumReduce(T *src, T *dst, int len) {
    int i = threadIdx.x;
    if (2 * i + 1 < len) {
        dst[i] = src[2 * i] + src[2 * i + 1];
    } else {
        dst[i] = src[2 * i];
    }
}

// function to find the sum of an array
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

// main function for Boruvka's algorithm
__host__ int boruvka(Graph &g) {
    // copy graph to GPU
    int *graph;
    int numVertices = g.size();
    size_t graphSize = numVertices * numVertices * sizeof(int);
    cudaMalloc((void **) &graph, graphSize);
    cudaMemcpy(graph, g.raw(), graphSize, cudaMemcpyHostToDevice);

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
    cudaMemset((void *) exists, 0, g.size() * sizeof(bool));

    int numExistingVertices = numVertices;
    while (numExistingVertices > 1) {
        // TODO: pass exists to everything
        // get pseudo-tree from the graph
        getPseudoTree<<<1,numVertices>>>(graph, T, parent, numVertices, exists);

        // convert pseudo-trees to rooted trees
        makeRootedTrees<<<1,numVertices>>>(parent, exists);

        // convert every rooted tree into a rooted star
        makeRootedStars<<<1,numVertices>>>(parent, exists);

        // contract all rooted stars into a single vertex
        contractRootedStars<<<1,numVertices>>>(graph, parent, numVertices, exists);

        // update number of existing vertices
        sum<uint8_t><<<1,1>>>((uint8_t *)exists, numVertices, &numExistingVertices);
    }

    // clean up
    cudaFree(graph);
    cudaFree(parent);
    cudaFree(exists);

    int result;
    sum<int><<<1,1>>>(T, numVertices, &result);
    return result;
}
