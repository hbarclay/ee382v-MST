#ifndef __GRAPH_H_
#define __GRAPH_H_

#include <cstring>
#include <limits>

#define MAX_WEIGHT 10
#define INF std::numeric_limits<int>::max()

class Graph {

 public:
	Graph() = delete;
	Graph(int _V, int _E) : V(_V), E(_E) {
		adjMatrix = new int[V*V];
		for (int i = 0; i < V*V; i++) {
			adjMatrix[i] = INF;
		}
		generateGraph();
	}

	Graph(const int* w, const int _V) : V(_V){
		adjMatrix = new int[V*V];
		std::memcpy(adjMatrix, w, V*V);
	}

	Graph(const Graph& other) : Graph(other.raw(), other.V) {}

	Graph& operator=(const Graph& other) {
		if (this == &other) return *this;
		V = other.size();
		int* newW = new int[V*V];
		std::memcpy(newW, other.raw(), V*V*sizeof(int));
		delete[] adjMatrix;
		adjMatrix = newW;
		return *this;
	}

	// TODO implement move constr. / assignment for performance

	~Graph() {
		delete [] adjMatrix;
	}

	void addEdge(int src, int dst, int w) {
		adjMatrix[src*V+dst] = w;
	}

	void printEdges();
	void printVertices();

	int* raw() const { return adjMatrix; }
	int size() const { return V; }

 private:
	int V;
	int E;
	int* adjMatrix;

	void generateGraph();
};

#endif // __GRAPH_H_
