#include <assert.h>
#include <vector>
#include <iostream>
#include <chrono>

#include "mstSeq.h"
#include "graph.h"

int getMin(const std::vector<int>& d, const auto& fixed) {
	int min = INF;
	int min_v = -1;

	for (int v = 0; v < d.size(); v++) {
		if (!fixed[v] && d[v] < min) {
			min = d[v];
			min_v = v;
		}
	}

	return min_v;
}

// very simple prim's implementation for correctness checks
// this is O(V^2) ! Could be O(E log V) with adj list and min heap
int primSeq(int* adjMap, int V) {
	std::vector<int> d(V, INF);
	std::vector<int> parent(V, -1);
	std::vector<bool> fixed(V, false);
	std::vector<std::pair<int, int>> T(V);

	d[0] = 0;

	for (int i = 0; i < V - 1; i++) {
		int u = getMin(d, fixed);

		fixed[u] = true;

		for (int v = 0; v < V; v++) {
			if (adjMap[u*V+v] && !fixed[v] && adjMap[u*V+v] < d[v]) {
				parent[v] = u;
				d[v] = adjMap[u*V+v];
			}
		}
	}

	int total_weight = 0;
	for (int i = 1; i < V; i++) {
		//std::cout << adjMap[parent[i]*V+i] << " " << parent[i] << " " << i << std::endl;
		total_weight += adjMap[parent[i]*V+i];
		T.push_back(std::make_pair(parent[i], i));
	}

	return total_weight;
}


int prim_cpu(const Graph& G, int& time) {

	auto begin = std::chrono::high_resolution_clock::now();
	int res = primSeq(G.raw(), G.size());
	auto end = std::chrono::high_resolution_clock::now();

	time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	return res;
}

// TODO Wrapper function to report time, borvukas algorithm, optimized versions of both
int boruvkaSeq(int* adjMap, int V) {
	
}

int boruvka_cpu(const Graph& G, int& time) {
	
}
