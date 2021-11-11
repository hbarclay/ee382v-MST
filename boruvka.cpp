// Test main for Boruvka implementation
#include <iostream>
#include "graph.h"
#include "boruvka_mst_gpu.h"


int main(void) {
    Graph g(8, 14);
    g.addEdge(0, 1, 4);
    g.addEdge(0, 2, 6);
    g.addEdge(0, 3, 16);
    g.addEdge(1, 5, 24);
    g.addEdge(2, 5, 23);
    g.addEdge(2, 4, 5);
    g.addEdge(2, 3, 8);
    g.addEdge(3, 4, 10);
    g.addEdge(3, 7, 21);
    g.addEdge(4, 5, 18);
    g.addEdge(4, 6, 11);
    g.addEdge(4, 7, 14);
    g.addEdge(5, 6, 9);
    g.addEdge(6, 7, 7);

    std::cout << "finished building graph...\n";

    std::cout << boruvka(g) << std::endl;

    return 0;
}
