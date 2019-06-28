
// #include "sort.cu"






/*!
 * Compute the Spearman correlation of a cluster in a pairwise data array.
 *
 * @param x
 * @param y
 * @param labels
 * @param sampleSize
 * @param stride
 * @param cluster
 * @param minSamples
 * @param x_sorted
 * @param y_sorted
 * @param rank
 */
__device__
float Spearman_computeCluster(
   const float *x,
   const float *y,
   const char *labels,
   int sampleSize,
   int stride,
   char cluster,
   int minSamples,
   float *x_sorted,
   float *y_sorted,
   int *rank)
{
   // extract samples in pairwise cluster
   int n = 0;

   for ( int i = 0, j = 0; i < sampleSize; ++i )
   {
      if ( labels[i * stride] == cluster )
      {
         x_sorted[j] = x[i];
         y_sorted[j] = y[i];
         rank[j] = n;
         j += stride;
         n += 1;
      }
   }

   // get power of 2 size
   int N_pow2 = nextPower2(sampleSize);

   for ( int i = n * stride; i < N_pow2 * stride; i += stride )
   {
      x_sorted[i] = INFINITY;
      y_sorted[i] = INFINITY;
      rank[i] = 0;
   }

   // compute correlation only if there are enough samples
   float result = NAN;

   if ( n >= minSamples )
   {
      // execute two sorts that are the beginning of the spearman algorithm
      bitonicSortFF(N_pow2, x_sorted, y_sorted, stride);
      bitonicSortFI(N_pow2, y_sorted, rank, stride);

      // go through spearman sorted rank list and calculate difference from 1,2,3,... list
      int diff = 0;

      for ( int i = 0; i < n; ++i )
      {
         int tmp = i - rank[i * stride];
         diff += tmp*tmp;
      }

      // compute spearman coefficient
      result = 1.0 - 6.0 * diff / (n * (n*n - 1));
   }

   return result;
}






/*!
 * Compute the correlation of each cluster in a pairwise data array. The data array
 * should only contain the clean samples that were extracted from the expression
 * matrix, while the labels should contain all samples.
 *
 * @param globalWorkSize
 * @param expressions
 * @param sampleSize
 * @param in_index
 * @param clusterSize
 * @param in_labels
 * @param minSamples
 * @param out_correlations
 */
__global__
void Spearman_compute(
   int globalWorkSize,
   const float *expressions,
   int sampleSize,
   const int2 *in_index,
   char clusterSize,
   const char *in_labels,
   int minSamples,
   float *work_x,
   float *work_y,
   int *work_rank,
   float *out_correlations)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = gridDim.x * blockDim.x;

   if ( i >= globalWorkSize )
   {
      return;
   }

   // initialize workspace variables
   int N_pow2 = nextPower2(sampleSize);
   int2 index = in_index[i];
   const float *x = &expressions[index.x * sampleSize];
   const float *y = &expressions[index.y * sampleSize];
   const char *labels = &in_labels[i];
   float *x_sorted = &work_x[i];
   float *y_sorted = &work_y[i];
   int *rank = &work_rank[i];
   float *correlations = &out_correlations[i];

   for ( char k = 0; k < clusterSize; ++k )
   {
      correlations[k * stride] = Spearman_computeCluster(x, y, labels, sampleSize, stride, k, minSamples, x_sorted, y_sorted, rank);
   }
}
