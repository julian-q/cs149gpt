#Warmup

**Briefly describe how a 4D tensor/array is laid out in memory. Why do you think this convention was chosen and how does it leverage hardware?**
For indices (i, j, k, l), all elements with i=0 come before those with i=1. The pattern repeats: for some fixed i, all elements with j=0 come before those with j=1, ...

This means that the last dimension (l) is stored contiguously in memory. This makes sense because, in a matrix multiplication, this is the dimension we will perform the dot-product along. It leverages hardware because of spatial locality: since cache lines load contiguous memory addresses in bulk, we will encounter cache hits for each iteration of the dot-product.


