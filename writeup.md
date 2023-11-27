#Warmup

**Briefly describe how a 4D tensor/array is laid out in memory. Why do you think this convention was chosen and how does it leverage hardware?**
For indices (i, j, k, l), all elements with i=0 come before those with i=1. The pattern repeats: for some fixed i, all elements with j=0 come before those with j=1, ...

This means that the last dimension (l) is stored contiguously in memory. This makes sense because, in a matrix multiplication, this is the dimension we will perform the dot-product along. It leverages hardware because of spatial locality: since cache lines load contiguous memory addresses in bulk, we will encounter cache hits for each iteration of the dot-product.

#Part 2
**What was the optimal tile size for your matrix multiplications? Explain why?**

The optimal tile size was 64x16. The larger the tile size, the greater the data reuse is for the data in the cache. However, once the tile size exceeds 16 in the horizontal dimension, loading a row of a given tile will require loading multiple cache lines (since a single cache line is 16 floats). A tile size of 16 means that, for the QK^T matmul, each dot-product within a tile can be computed using a single cache line from each matrix.

**For a matrix multiply of Q (Nxd) and K^T (dxN), what is the ratio of DRAM accesses in Part 2 versus DRAM acceses in Part 1? (assume 4 byte float primitives, 64 byte cache lines, as well as N and d are very large).**

In Part 1, every element required 2d DRAM accesses. But with a tiled approach, you can compute 64x16 elements with only (64+16)d accesses, since each row and column you load can be reused by every other element in the tile. Thus, the ratio is 2d / ((64+16)d/64x16) = 25.6. Part 1 requires 25.6x more DRAM accesses.

#Part 3
**Why do we use a drastically smaller amount of memory in Part 3 when compared to Parts 1 & 2?**

Instead of materializing a full NxN attention matrix in memory, we only need N x (num threads) memory. This is because each thread just needs scratchpad space for a single row of the attention matrix, due to the fusion.

**Comment out your #pragma omp ... statement, what happens to your cpu time? Record the cpu time in your writeup. Why does fused attention make it easier for us utilize multithreading to a much fuller extent when compared to Part 1?**

The execution time becomes slower by ~8x (260 ms vs 35 ms). Fused attention makes it easier for us to utilize multithreading because it removes unnecessary data dependencies, meaning more of our computation can be done in parallel. In Part 1, the entire NxN attention matrix had to be materialized before computing the output matrix, but in this case, each row of the output matrix only depends on a single row of the attention matrix. Thus, the computation of each row of the attention matrix becomes an independent task in a task pool, which can be worked on in parallel by a thread pool.

#Part 4
**How does the memory usage of Part 4 compare to that of the previous parts? Why is this the case?**

Part 4 uses the least amount of memory compared to all previous parts. This is because it requires the least amount of scratchpad space to store intermediate values. While Part 3 required only a single row of the attention matrix, a single row of size N can be quite large for longer sequence lengths. Part 4, on the other hand, only requires a fixed size square tile of the attention matrix to be in memory at a time. This tile can be e.g. 16x16 = 256, which is much smaller than a whole row for a sequence length of 1024 or 2048.

**Notice that the performance of Part 4 is slower than that of the previous parts. Have we fully optimized Part 4? What other performance improvements can be done? Please list them and describe why they would increase performance.**

We have not fully optimized Part 4. We can still perform the following optimizations:
1. Parallelize the loop over Queries. Right now there is no dependency between each iteration over the query and output tiles (loop "i" in the pseudocode). This should give a speedup of a factor of N, assuming infinite memory bandwidth.
2. Allow for tiling the reduciton dimension. In Part 2 we saw that a tile size of 16 is optimal along the dot product dimension. In Part 4, however, we fix the tile size to be of width d. By reducing the tile width to be just a single cache line, we only have to load one cache line for every tile of the dot product.


