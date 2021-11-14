#include <assert.h>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <tuple>
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
	
	//int solution[V]={0};
	int total_weight = 0;
	for (int i = 1; i < V; i++) {
		//std::cout << adjMap[parent[i]*V+i] << " " << parent[i] << " " << i << std::endl;
		total_weight += adjMap[parent[i]*V+i];
		T.push_back(std::make_pair(parent[i], i));
		//solution[i] = adjMap[parent[i]*V+i];
	}
	//printf("prim_cpu T:");
	//for (int i = 0; i < V; i++) {
	//	printf("%d ", solution[i]);
	//}
	//printf("\n");

	return total_weight;

	
}


int prim_cpu(const Graph& G, int& time) {

	auto begin = std::chrono::high_resolution_clock::now();
	int res = primSeq(G.raw(), G.size());
	auto end = std::chrono::high_resolution_clock::now();

	time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	return res;
}

// DSU utility
// https://cp-algorithms.com/data_structures/disjoint_set_union.html
int find_set(auto& parent, int i) {
	if (parent[i] == i) 
		return parent[i];

	return parent[i] = find_set(parent, parent[i]);
}


void set_union(auto& rank, auto& parent, int a , int b) {
	a = find_set(parent, a);
	b = find_set(parent, b);

	if (a != b) {
		if (rank[a] < rank[b])
			std::swap(a, b);

		parent[b] = a;

		// rank(a) == rank(b)
		if (rank[a] == rank[b]) {
			rank[a]++;
		}
		
	}
}


int boruvkaSeq(const std::vector<std::tuple<int, int, int>>& edgeList, int V, int E) {
	int total_weight = 0;
	int numtrees = V;

	std::vector<int> rank(V, 0);
	std::vector<int> parent(V);

	for (int i = 0; i < numtrees; i++) {
		parent[i] = i;
	}	

	while(numtrees != 1) {
		std::vector<int> minedge(V, INF);

		for (int i = 0; i < E; i++) {
			int tree_src = find_set(parent, std::get<0>(edgeList[i])); // find tree for each vertex of of edge i
			int tree_dst = find_set(parent, std::get<1>(edgeList[i]));


			if (tree_src == tree_dst) {
				continue;
			}

			minedge[tree_src] = minedge[tree_src] == INF ? i : minedge[tree_src];
			minedge[tree_dst] = minedge[tree_dst] == INF ? i : minedge[tree_dst];

			if (std::get<2>(edgeList[i]) < std::get<2>(edgeList[minedge[tree_src]])) {
				minedge[tree_src] = i;
			}

			if (std::get<2>(edgeList[i]) < std::get<2>(edgeList[minedge[tree_dst]])) {
				minedge[tree_dst] = i;
			}
		}

		for (int i = 0; i < V; i++) {
			if (minedge[i] != INF) {
				int tree_src = find_set(parent, std::get<0>(edgeList[minedge[i]]));
				int tree_dst = find_set(parent, std::get<1>(edgeList[minedge[i]]));

				if (tree_src == tree_dst)
					continue;

				total_weight += std::get<2>(edgeList[minedge[i]]);

				set_union(rank, parent, tree_src, tree_dst);
				numtrees--;

			}
		}
	}

	return total_weight;
}

int boruvka_cpu(const Graph& G, int& time) {
	int* map = G.raw();
	const int V = G.size();

	std::vector<std::tuple<int, int, int>> edgeList;

	for (int i = 0; i < G.size(); i++) {
		for (int j = i+1; j < G.size(); j++) {
			if (map[i*V+j] != INF)
				edgeList.push_back(std::make_tuple(i, j, map[i*V+j]));
		}
	}
	
	auto begin = std::chrono::high_resolution_clock::now();
	int result =  boruvkaSeq(edgeList, G.size(), G.edges());
	auto end = std::chrono::high_resolution_clock::now();
	time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	return result;
}
