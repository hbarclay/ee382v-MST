#ifndef __BORUVKA_MST_GPU_H__
#define __BORUVKA_MST_GPU_H__

#include <cuda.h>
#include "graph.h"

__host__ int boruvka(Graph &g);

#endif