executable: mst.cu prim_mst_gpu.cu
	nvcc -o executable mst.cu prim_mst_gpu.cu

clean:
	rm executable
