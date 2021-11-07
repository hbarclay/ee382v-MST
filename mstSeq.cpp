#include <assert.h>
#include <vector>
#include <iostream>

#include "mstSeq.h"
#include "graph.h"

int* primSeq(int* adjMap, int V) {
	std::vector<int> d(V, INF);
	std::vector<int> parent(V, -1);
	std::vector<bool> fixed(V, false);
	std::vector<std::pair<int, int>> T(V);

	d[0] = 0;

	while (T.size() < V - 1) {
		int v = -1;
		int min = INF;
		for (int i = 0; i < V; i++) {
			if (!fixed[i]) {
				if (d[i] < min) {
					v = i;
					min = d[i];
				}
			}
		}

		assert(v != -1);

		if (d[v] == INF)
			return nullptr;

		if (parent[v] != -1)
			T.push_back(std::make_pair(v, parent[v]));

		fixed[v] = true;

		// Undirected graph -- look at top half of adj matrix
		for (int i = 0; i < V; i++) {
			for (int j = i + 1; j < V; j++) {
				if (adjMap[i*V+j] < d[j]) {
					d[j] = adjMap[i*V+j];
					parent[j] = i;
				}
			}
		}
	}

	std::cout << "DONE" << std::endl;

	for (auto a : T) {
		std::cout << a.first << " " << a.second << " " << std::endl;
	}

	return nullptr;
	
}

