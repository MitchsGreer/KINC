
// #include "fetchpair.cu"
// #include "linalg.cu"
// #include "outlier.cu"






struct GMM
{
   float *pi;
   Vector2 *mu;
   Matrix2x2 *sigma;
   Matrix2x2 *sigmaInv;
   float *normalizer;

   int K;
   float logL;
   float entropy;
   Vector2 *_MP;
   int *_counts;
   float *_logpi;
   float *_gamma;
};






/*!
 * Implementation of rand(), taken from POSIX example.
 *
 * @param state
 */
__device__
int myrand(unsigned long *state)
{
   *state = (*state) * 1103515245 + 12345;
   return ((unsigned)((*state)/65536) % 32768);
}






/*!
 * Initialize a mixture component with the given mixture weight and mean.
 *
 * @param gmm
 * @param k
 * @param pi
 * @param mu
 */
__device__
void GMM_Component_initialize(
   GMM *gmm,
   int k,
   float pi,
   const Vector2 *mu)
{
   // initialize mixture weight and mean
   gmm->pi[k] = pi;
   gmm->mu[k] = *mu;

   // initialize covariance to identity matrix
   matrixInitIdentity(&gmm->sigma[k]);
}






/*!
 * Pre-compute the precision matrix and normalizer term for a mixture component.
 *
 * @param gmm
 * @param k
 */
__device__
bool GMM_Component_prepare(GMM *gmm, int k)
{
   const int D = 2;

   // compute precision (inverse of covariance)
   float det;
   matrixInverse(&gmm->sigma[k], &gmm->sigmaInv[k], &det);

   // compute normalizer term for multivariate normal distribution
   gmm->normalizer[k] = -0.5f * (D * logf(2.0f * M_PI) + logf(det));

   // return failure if matrix inverse failed
   return !(det <= 0 || isnan(det));
}






/*!
 * Compute the log of the probability density function of the multivariate normal
 * distribution conditioned on a single component for each point in X:
 *
 *   P(x|k) = exp(-0.5 * (x - mu)^T Sigma^-1 (x - mu)) / sqrt((2pi)^d det(Sigma))
 *
 * Therefore the log-probability is:
 *
 *   log(P(x|k)) = -0.5 * (x - mu)^T Sigma^-1 (x - mu) - 0.5 * (d * log(2pi) + log(det(Sigma)))
 *
 * @param gmm
 * @param k
 * @param X
 * @param N
 * @param logP
 * @param stride
 */
__device__
void GMM_Component_computeLogProbNorm(
   GMM *gmm,
   int k,
   const Vector2 *X,
   int N,
   float *logP,
   int stride)
{
   for ( int i = 0; i < N * stride; i += stride )
   {
      // compute xm = (x - mu)
      Vector2 xm = X[i];
      vectorSubtract(&xm, &gmm->mu[k]);

      // compute Sxm = Sigma^-1 xm
      Vector2 Sxm;
      matrixProduct(&gmm->sigmaInv[k], &xm, &Sxm);

      // compute xmSxm = xm^T Sigma^-1 xm
      float xmSxm = vectorDot(&xm, &Sxm);

      // compute log(P) = normalizer - 0.5 * xm^T * Sigma^-1 * xm
      logP[i] = gmm->normalizer[k] - 0.5f * xmSxm;
   }
}






/*!
 * Initialize the mean of each component in the mixture model using k-means
 * clustering.
 *
 * @param gmm
 * @param X
 * @param N
 * @param stride
 */
__device__
void GMM_initializeMeans(GMM *gmm, const Vector2 *X, int N, int stride)
{
   const int K = gmm->K;

   const int MAX_ITERATIONS = 20;
   const float TOLERANCE = 1e-3f;
   float diff = 0;

   // initialize workspace
   Vector2 *MP = gmm->_MP;
   int *counts = gmm->_counts;

   for ( int t = 0; t < MAX_ITERATIONS && diff > TOLERANCE; ++t )
   {
      // compute mean and sample count for each component
      for ( int k = 0; k < K * stride; k += stride )
      {
         vectorInitZero(&MP[k]);
         counts[k] = 0;
      }

      for ( int i = 0; i < N * stride; i += stride )
      {
         // determine the component mean which is nearest to x_i
         float min_dist = INFINITY;
         int min_k = 0;
         for ( int k = 0; k < K * stride; k += stride )
         {
            float dist = vectorDiffNorm(&X[i], &gmm->mu[k]);
            if ( min_dist > dist )
            {
               min_dist = dist;
               min_k = k;
            }
         }

         // update mean and sample count
         vectorAdd(&MP[min_k], &X[i]);
         ++counts[min_k];
      }

      // scale each mean by its sample count
      for ( int k = 0; k < K * stride; k += stride )
      {
         vectorScale(&MP[k], 1.0f / counts[k]);
      }

      // compute the total change of all means
      diff = 0;
      for ( int k = 0; k < K * stride; k += stride )
      {
         diff += vectorDiffNorm(&MP[k], &gmm->mu[k]);
      }
      diff /= K;

      // update component means
      for ( int k = 0; k < K * stride; k += stride )
      {
         gmm->mu[k] = MP[k];
      }
   }
}






/*!
 * Perform the expectation step of the EM algorithm. In this step we compute
 * gamma, the posterior probabilities for each component in the mixture model
 * and each sample in X, as well as the log-likelihood of the model:
 *
 *   log(p(x_i)) = a + log(sum(exp(log(pi_k) + log(P(x_i|k))) - a))
 *
 *   gamma_ki = exp(log(pi_k) + log(P(x_i|k)) - log(p(x_i)))
 *
 *   log(L) = sum(log(p(x_i)))
 *
 * @param gmm
 * @param X
 * @param N
 * @param stride
 */
__device__
float GMM_computeEStep(GMM *gmm, const Vector2 *X, int N, int stride)
{
   const int K = gmm->K;

   // compute logpi
   for ( int k = 0; k < K * stride; k += stride )
   {
      gmm->_logpi[k] = logf(gmm->pi[k]);
   }

   // compute the log-probability for each component and each point in X
   float *logProb = gmm->_gamma;

   for ( int k = 0; k < K * stride; k += stride )
   {
      GMM_Component_computeLogProbNorm(gmm, k, X, N, &logProb[k * N], stride);
   }

   // compute gamma and log-likelihood
   float logL = 0;

   for ( int i = 0; i < N * stride; i += stride )
   {
      // compute a = argmax(logpi_k + logProb_ki, k)
      float maxArg = -INFINITY;
      for ( int k = 0; k < K * stride; k += stride )
      {
         float arg = gmm->_logpi[k] + logProb[k * N + i];
         if ( maxArg < arg )
         {
            maxArg = arg;
         }
      }

      // compute logpx
      float sum = 0;
      for ( int k = 0; k < K * stride; k += stride )
      {
         sum += expf(gmm->_logpi[k] + logProb[k * N + i] - maxArg);
      }

      float logpx = maxArg + logf(sum);

      // compute gamma_ki
      for ( int k = 0; k < K * stride; k += stride )
      {
         gmm->_gamma[k * N + i] += gmm->_logpi[k] - logpx;
         gmm->_gamma[k * N + i] = expf(gmm->_gamma[k * N + i]);
      }

      // update log-likelihood
      logL += logpx;
   }

   // return log-likelihood
   return logL;
}






/*!
 * Perform the maximization step of the EM algorithm. In this step we update the
 * parameters of the the mixture model using gamma, which is computed during the
 * expectation step:
 *
 *   n_k = sum(gamma_ki)
 *
 *   pi_k = n_k / N
 *
 *   mu_k = sum(gamma_ki * x_i)) / n_k
 *
 *   Sigma_k = sum(gamma_ki * (x_i - mu_k) * (x_i - mu_k)^T) / n_k
 *
 * @param gmm
 * @param X
 * @param N
 * @param stride
 */
__device__
void GMM_computeMStep(GMM *gmm, const Vector2 *X, int N, int stride)
{
   const int K = gmm->K;

   for ( int k = 0; k < K * stride; k += stride )
   {
      // compute n_k = sum(gamma_ki)
      float n_k = 0;

      for ( int i = 0; i < N * stride; i += stride )
      {
         n_k += gmm->_gamma[k * N + i];
      }

      // update mixture weight
      gmm->pi[k] = n_k / N;

      // update mean
      Vector2 *mu = &gmm->mu[k];

      vectorInitZero(mu);

      for ( int i = 0; i < N * stride; i += stride )
      {
         vectorAddScaled(mu, gmm->_gamma[k * N + i], &X[i]);
      }

      vectorScale(mu, 1.0f / n_k);

      // update covariance matrix
      Matrix2x2 *sigma = &gmm->sigma[k];

      matrixInitZero(sigma);

      for ( int i = 0; i < N * stride; i += stride )
      {
         // compute xm = (x_i - mu_k)
         Vector2 xm = X[i];
         vectorSubtract(&xm, mu);

         // compute Sigma_ki = gamma_ki * (x_i - mu_k) (x_i - mu_k)^T
         matrixAddOuterProduct(sigma, gmm->_gamma[k * N + i], &xm);
      }

      matrixScale(sigma, 1.0f / n_k);
   }
}






/*!
 * Compute the cluster labels of a dataset using gamma:
 *
 *   y_i = argmax(gamma_ki, k)
 *
 * @param gamma
 * @param N
 * @param K
 * @param labels
 * @param stride
 */
__device__
void GMM_computeLabels(
   const float *gamma,
   int N,
   int K,
   char *labels,
   int stride)
{
   for ( int i = 0; i < N * stride; i += stride )
   {
      // determine the value k for which gamma_ki is highest
      int max_k = -1;
      float max_gamma = -INFINITY;

      for ( int k = 0; k < K * stride; k += stride )
      {
         if ( max_gamma < gamma[k * N + i] )
         {
            max_k = k;
            max_gamma = gamma[k * N + i];
         }
      }

      // assign x_i to cluster k
      labels[i] = max_k / stride;
   }
}






/*!
 * Compute the entropy of the mixture model for a dataset using gamma
 * and the given cluster labels:
 *
 *   E = -sum(sum(z_ki * log(gamma_ki))), z_ki = (y_i == k)
 *
 * @param gamma
 * @param N
 * @param labels
 * @param stride
 */
__device__
float GMM_computeEntropy(
   const float *gamma,
   int N,
   const char *labels,
   int stride)
{
   float E = 0;

   for ( int i = 0; i < N * stride; i += stride )
   {
      int k = labels[i] * stride;

      E -= logf(gamma[k * N + i]);
   }

   return E;
}






/*!
 * Fit the mixture model to a pairwise data array and compute the output cluster
 * labels for the data. The data array should only contain clean samples.
 *
 * @param gmm
 * @param X
 * @param N
 * @param K
 * @param labels
 * @param stride
 */
__device__
bool GMM_fit(
   GMM *gmm,
   const Vector2 *X,
   int N,
   int K,
   char *labels,
   int stride)
{
   // initialize random state
   unsigned long state = 1;

   // initialize components
   gmm->K = K;

   for ( int k = 0; k < K * stride; k += stride )
   {
      // use uniform mixture weight and randomly sampled mean
      int i = (myrand(&state) % N) * stride;

      GMM_Component_initialize(gmm, k, 1.0f / K, &X[i]);
   }

   // initialize means with k-means
   GMM_initializeMeans(gmm, X, N, stride);

   // run EM algorithm
   const int MAX_ITERATIONS = 100;
   const float TOLERANCE = 1e-8f;
   float prevLogL = -INFINITY;
   float currLogL = -INFINITY;

   for ( int t = 0; t < MAX_ITERATIONS; ++t )
   {
      // pre-compute precision matrix and normalizer term
      bool success = true;

      for ( int k = 0; k < K * stride; k += stride )
      {
         success = success && GMM_Component_prepare(gmm, k);
      }

      // return failure if matrix inverse failed
      if ( !success )
      {
         return false;
      }

      // perform E step
      prevLogL = currLogL;
      currLogL = GMM_computeEStep(gmm, X, N, stride);

      // check for convergence
      if ( fabs(currLogL - prevLogL) < TOLERANCE )
      {
         break;
      }

      // perform M step
      GMM_computeMStep(gmm, X, N, stride);
   }

   // save outputs
   gmm->logL = currLogL;
   GMM_computeLabels(gmm->_gamma, N, K, labels, stride);
   gmm->entropy = GMM_computeEntropy(gmm->_gamma, N, labels, stride);

   return true;
}






typedef enum
{
   AIC,
   BIC,
   ICL
} Criterion;






/*!
 * Compute the Akaike Information Criterion of a Gaussian mixture model.
 *
 * @param K
 * @param D
 * @param logL
 */
__device__
float GMM_computeAIC(int K, int D, float logL)
{
   int p = K * (1 + D + D * D);

   return 2 * p - 2 * logL;
}






/*!
 * Compute the Bayesian Information Criterion of a Gaussian mixture model.
 *
 * @param K
 * @param D
 * @param logL
 * @param N
 */
__device__
float GMM_computeBIC(int K, int D, float logL, int N)
{
   int p = K * (1 + D + D * D);

   return logf((float) N) * p - 2 * logL;
}






/*!
 * Compute the Integrated Completed Likelihood of a Gaussian mixture model.
 *
 * @param K
 * @param D
 * @param logL
 * @param N
 * @param E
 */
__device__
float GMM_computeICL(int K, int D, float logL, int N, float E)
{
   int p = K * (1 + D + D * D);

   return logf((float) N) * p - 2 * logL + 2 * E;
}






/*!
 * Determine the number of clusters in a pairwise data array. Several sub-models,
 * each one having a different number of clusters, are fit to the data and the
 * sub-model with the best criterion value is selected.
 *
 * @param numPairs
 * @param expressions
 * @param sampleSize
 * @param in_index
 * @param minSamples
 * @param minClusters
 * @param maxClusters
 * @param criterion
 * @param out_K
 * @param out_labels
 */
__global__
void GMM_compute(
   int numPairs,
   const float *expressions,
   int sampleSize,
   const int2 *in_index,
   int minSamples,
   char minClusters,
   char maxClusters,
   Criterion criterion,
   Vector2 *work_X,
   int *work_N,
   char *work_labels,
   float *work_gmm_pi,
   Vector2 *work_gmm_mu,
   Matrix2x2 *work_gmm_sigma,
   Matrix2x2 *work_gmm_sigmaInv,
   float *work_gmm_normalizer,
   Vector2 *work_gmm_MP,
   int *work_gmm_counts,
   float *work_gmm_logpi,
   float *work_gmm_gamma,
   char *out_K,
   char *out_labels)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = gridDim.x * blockDim.x;

   if ( i >= numPairs )
   {
      return;
   }

   // initialize workspace variables
   int2 index = in_index[i];
   const float *x = &expressions[index.x * sampleSize];
   const float *y = &expressions[index.y * sampleSize];

   Vector2 *X = &work_X[i];
   int numSamples = work_N[i];
   char *labels = &work_labels[i];

   float *     gmm_pi = &work_gmm_pi[i];
   Vector2 *   gmm_mu = &work_gmm_mu[i];
   Matrix2x2 * gmm_sigma = &work_gmm_sigma[i];
   Matrix2x2 * gmm_sigmaInv = &work_gmm_sigmaInv[i];
   float *     gmm_normalizer = &work_gmm_normalizer[i];
   Vector2 *   gmm_MP = &work_gmm_MP[i];
   int *       gmm_counts = &work_gmm_counts[i];
   float *     gmm_logpi = &work_gmm_logpi[i];
   float *     gmm_gamma = &work_gmm_gamma[i];

   char *bestK = &out_K[i];
   char *bestLabels = &out_labels[i];

   // initialize GMM struct
   GMM gmm = {
      gmm_pi,
      gmm_mu,
      gmm_sigma,
      gmm_sigmaInv,
      gmm_normalizer,
      0, 0, 0,
      gmm_MP,
      gmm_counts,
      gmm_logpi,
      gmm_gamma
   };

   // perform clustering only if there are enough samples
   *bestK = 0;

   if ( numSamples >= minSamples )
   {
      // extract clean samples from data array
      for ( int i = 0, j = 0; i < sampleSize; ++i )
      {
         if ( bestLabels[i * stride] >= 0 )
         {
            X[j] = make_float2(x[i], y[i]);
            j += stride;
         }
      }

      // determine the number of clusters
      float bestValue = INFINITY;

      for ( char K = minClusters; K <= maxClusters; ++K )
      {
         // run each clustering sub-model
         bool success = GMM_fit(&gmm, X, numSamples, K, labels, stride);

         if ( !success )
         {
            continue;
         }

         // compute the criterion value of the sub-model
         float value = INFINITY;

         switch (criterion)
         {
         case AIC:
            value = GMM_computeAIC(K, 2, gmm.logL);
            break;
         case BIC:
            value = GMM_computeBIC(K, 2, gmm.logL, numSamples);
            break;
         case ICL:
            value = GMM_computeICL(K, 2, gmm.logL, numSamples, gmm.entropy);
            break;
         }

         // save the sub-model with the lowest criterion value
         if ( value < bestValue )
         {
            *bestK = K;
            bestValue = value;

            // save labels for clean samples
            for ( int i = 0, j = 0; i < sampleSize * stride; i += stride )
            {
               if ( bestLabels[i] >= 0 )
               {
                  bestLabels[i] = labels[j];
                  j += stride;
               }
            }
         }
      }
   }
}
