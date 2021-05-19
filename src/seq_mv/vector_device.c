/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

__global__ void
hypreCUDAKernel_MassAxpy(const HYPRE_Int k, const HYPRE_Int n, const HYPRE_Complex * __restrict__ alpha, const HYPRE_Complex * __restrict__ x, HYPRE_Complex *y)
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();
   HYPRE_Int j = 0;
   HYPRE_Complex sum = 0;

   if (i < n)
   {
      for (j = 0; j < k; ++j)
      {
        HYPRE_Complex xx = x[j*n+i];
        sum += alpha[j] * xx;
      }
      y[i] += sum;
   }
}

HYPRE_Int
hypreDevice_MassAxpy(HYPRE_Int k, HYPRE_Int n, HYPRE_Complex *alpha, HYPRE_Complex *x, HYPRE_Complex *y)
{
   /* trivial case */
   if (n <= 0 || k<=0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(n, "thread", bDim);
   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_MassAxpy, gDim, bDim, k, n, alpha, x, y);
   return hypre_error_flag;
}

template<HYPRE_Int B>
__global__ void
hypreCUDAKernel_MassInnerProd_kernel1(const HYPRE_Int k, const HYPRE_Int n, const HYPRE_Complex * __restrict__ x, const HYPRE_Complex * __restrict__ y, HYPRE_Complex *z)
{
  HYPRE_Int i=0, j=0;
   volatile __shared__ HYPRE_Complex shmem[B/32];
   HYPRE_Int tidx=threadIdx.x+blockDim.x*blockIdx.x;
   HYPRE_Complex xx=0.0;
   if (tidx<n)
      xx = x[tidx];

   if (threadIdx.x<B/32)
      shmem[threadIdx.x] = 0.0;
   __syncthreads();

   HYPRE_Int warp_lane=threadIdx.x%32;
   HYPRE_Int warp_id = threadIdx.x/32;

   HYPRE_Int ind1 = tidx;
   HYPRE_Int ind2 = blockIdx.x;

   for (j=0; j<k; ++j) 
   {
      HYPRE_Complex sum=0;
      if (tidx<n)
         sum =  xx*y[ind1];

      // Warp shuffle reduce
      for (i = 16; i > 0; i >>= 1)
         sum += __shfl_down_sync(32, sum, i);

      // Combine across warps through shmem
      __syncthreads();
      if (warp_lane==0)
         shmem[warp_id] = sum;
      __syncthreads();

      // Put it back in a register to finish the reduction
      if (threadIdx.x<B/32)
	sum = shmem[threadIdx.x];

      // Warp shuffle reduce
      for (i = B/64; i > 0; i >>= 1)
         sum += __shfl_down_sync(B/32, sum, i);
      
      /* Write to global memory */
      if (threadIdx.x==0)
         z[ind2] = sum;

      // update indices
      ind1+=n;
      ind2+=gridDim.x;
   }
}

template<HYPRE_Int B>
__global__ void
hypreCUDAKernel_MassInnerProd_kernel2(const HYPRE_Int n, const HYPRE_Complex * __restrict__ z, HYPRE_Complex *w)
{
   HYPRE_Int i = 0;

   volatile __shared__ HYPRE_Complex shmem[B];

   HYPRE_Complex sum = 0;
   for (i=threadIdx.x+blockDim.x*blockIdx.x; i<n; i+=gridDim.x*blockDim.x)
   {
      HYPRE_Complex zz = z[blockIdx.y*n + i];
      sum += zz;
    }

   HYPRE_Int warp_lane=threadIdx.x%32;
   HYPRE_Int warp_id = threadIdx.x/32;

   // Warp shuffle reduce
   for (i = 16; i > 0; i >>= 1)
     sum += __shfl_down_sync(32, sum, i);

   // Combine across warps through shmem
   __syncthreads();
   if (warp_lane==0)
      shmem[warp_id] = sum;
   __syncthreads();

   // Put it back in a register to finish the reduction
   if (threadIdx.x<B/32)
      sum = shmem[threadIdx.x];

   // Warp shuffle reduce
   for (i = B/64; i > 0; i >>= 1)
      sum += __shfl_down_sync(B/32, sum, i);

   /* Write to global memory */
   if (threadIdx.x==0)
      w[blockIdx.y*gridDim.x + blockIdx.x] = sum;

}

__global__ void
hypreCUDAKernel_MassInnerProd_kernel3(const HYPRE_Int n, const HYPRE_Complex * __restrict__ z, HYPRE_Complex *w)
{
   HYPRE_Int i = 0;

   HYPRE_Complex sum = 0.0;

   if (threadIdx.x<n) 
      sum = z[n*blockIdx.x + threadIdx.x];

   /* reduce */
   for (i = 8; i > 0; i >>= 1)
      sum += __shfl_down_sync(8, sum, i);

   /* Write to global memory */
   if (threadIdx.x==0)
      w[blockIdx.x] = sum;
}



HYPRE_Int
hypreDevice_MassInnerProd(HYPRE_Int k, HYPRE_Int n, HYPRE_Complex *x, HYPRE_Complex *y, HYPRE_Complex *result)
{
   /* trivial case */
   if (n <= 0 || k<=0)
   {
      return hypre_error_flag;
   }

   const HYPRE_Int N1 = 512;
   HYPRE_Int num_blocks = ((n+N1-1)/N1);

   const HYPRE_Int N2 = 512;
   HYPRE_Int nb = ((num_blocks+N2-1)/N2);
   HYPRE_Int num_blocks2 = nb < 16 ? nb : 16;

   HYPRE_Complex * d_result;
   if (num_blocks>32) {
     HYPRE_Int total_mem = k*num_blocks + k*num_blocks2 + k;
     d_result = hypre_CTAlloc(HYPRE_Complex,total_mem,HYPRE_MEMORY_DEVICE);
   } else {
     HYPRE_Int total_mem = k*num_blocks + k;
     d_result = hypre_CTAlloc(HYPRE_Complex,total_mem,HYPRE_MEMORY_DEVICE);
   }

   /* Kernel 1 : Initial Reduction */
   hypreCUDAKernel_MassInnerProd_kernel1<N1><<<num_blocks,N1>>>(k, n, x, y, d_result);

   /* Kernel 2/3 : Final Reduction */
   if (num_blocks>16) {
     dim3 gDim(num_blocks2,k,1);
     dim3 bDim(N2,1,1);
     hypreCUDAKernel_MassInnerProd_kernel2<N2><<<gDim,bDim>>>(num_blocks, d_result, d_result+k*num_blocks);
     hypreCUDAKernel_MassInnerProd_kernel3<<<k, 16>>>(num_blocks2, d_result+k*num_blocks, d_result+k*(num_blocks+num_blocks2));
     hypre_TMemcpy(result, d_result+k*(num_blocks+num_blocks2), HYPRE_Complex, k, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   } else {
     hypreCUDAKernel_MassInnerProd_kernel3<<<k, 16>>>(num_blocks, d_result, d_result+k*num_blocks);
     hypre_TMemcpy(result, d_result+k*num_blocks, HYPRE_Complex, k, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   hypre_TFree(d_result, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP)
