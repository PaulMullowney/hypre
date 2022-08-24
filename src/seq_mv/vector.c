/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_hypre_utilities.hpp" //RL: TODO vector_device.c, include cuda there

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCreate
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCreate( HYPRE_Int size )
{
   hypre_Vector  *vector;

   vector = hypre_CTAlloc(hypre_Vector, 1, HYPRE_MEMORY_HOST);

   hypre_VectorData(vector) = NULL;
   hypre_VectorSize(vector) = size;
   hypre_VectorNumVectors(vector) = 1;
   hypre_VectorMultiVecStorageMethod(vector) = 0;

   /* set defaults */
   hypre_VectorOwnsData(vector) = 1;

   hypre_VectorMemoryLocation(vector) = hypre_HandleMemoryLocation(hypre_handle());

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultiVectorCreate
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqMultiVectorCreate( HYPRE_Int size, HYPRE_Int num_vectors )
{
   hypre_Vector *vector = hypre_SeqVectorCreate(size);
   hypre_VectorNumVectors(vector) = num_vectors;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorDestroy( hypre_Vector *vector )
{
   if (vector)
   {
      HYPRE_MemoryLocation memory_location = hypre_VectorMemoryLocation(vector);

      if (hypre_VectorOwnsData(vector))
      {
         hypre_TFree(hypre_VectorData(vector), memory_location);
      }

	  hypre_TFree(vector, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInitialize_v2
 *
 * Initialize a vector at a given memory location
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorInitialize_v2( hypre_Vector *vector, HYPRE_MemoryLocation memory_location )
{
   HYPRE_Int  size = hypre_VectorSize(vector);
   HYPRE_Int  num_vectors = hypre_VectorNumVectors(vector);
   HYPRE_Int  multivec_storage_method = hypre_VectorMultiVecStorageMethod(vector);

   hypre_VectorMemoryLocation(vector) = memory_location;

   /* Caveat: for pre-existing data, the memory location must be guaranteed
    * to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered
    * when being used, and freed */
   if (!hypre_VectorData(vector))
   {
      hypre_VectorData(vector) = hypre_CTAlloc(HYPRE_Complex, num_vectors * size, memory_location);
   }

   if (multivec_storage_method == 0)
   {
      hypre_VectorVectorStride(vector) = size;
      hypre_VectorIndexStride(vector)  = 1;
   }
   else if (multivec_storage_method == 1)
   {
      hypre_VectorVectorStride(vector) = 1;
      hypre_VectorIndexStride(vector)  = num_vectors;
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid multivec storage method!\n");
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorInitialize( hypre_Vector *vector )
{
   return hypre_SeqVectorInitialize_v2(vector, hypre_VectorMemoryLocation(vector));
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetDataOwner( hypre_Vector *vector,
                             HYPRE_Int     owns_data   )
{
   hypre_VectorOwnsData(vector) = owns_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetSize( hypre_Vector *vector,
                        HYPRE_Int     size   )
{
   HYPRE_Int  multivec_storage_method = hypre_VectorMultiVecStorageMethod(vector);

   hypre_VectorSize(vector) = size;
   if (multivec_storage_method == 0)
   {
      hypre_VectorVectorStride(vector) = size;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * ReadVector
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorRead( char *file_name )
{
   hypre_Vector  *vector;

   FILE    *fp;

   HYPRE_Complex *data;
   HYPRE_Int      size;

   HYPRE_Int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   hypre_fscanf(fp, "%d", &size);

   vector = hypre_SeqVectorCreate(size);

   hypre_VectorMemoryLocation(vector) = HYPRE_MEMORY_HOST;

   hypre_SeqVectorInitialize(vector);

   data = hypre_VectorData(vector);
   for (j = 0; j < size; j++)
   {
      hypre_fscanf(fp, "%le", &data[j]);
   }

   fclose(fp);

   /* multivector code not written yet */
   hypre_assert( hypre_VectorNumVectors(vector) == 1 );

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPrint( hypre_Vector *vector,
                      char         *file_name )
{
   FILE          *fp;

   HYPRE_Complex *data;
   HYPRE_Int      size, num_vectors, vecstride, idxstride;

   HYPRE_Int      i, j;
   HYPRE_Complex  value;

   num_vectors = hypre_VectorNumVectors(vector);
   vecstride = hypre_VectorVectorStride(vector);
   idxstride = hypre_VectorIndexStride(vector);

   /*----------------------------------------------------------
    * Print in the data
    *----------------------------------------------------------*/

   data = hypre_VectorData(vector);
   size = hypre_VectorSize(vector);

   fp = fopen(file_name, "w");

   if ( hypre_VectorNumVectors(vector) == 1 )
   {
      hypre_fprintf(fp, "%d\n", size);
   }
   else
   {
      hypre_fprintf(fp, "%d vectors of size %d\n", num_vectors, size );
   }

   if ( num_vectors > 1 )
   {
      for ( j = 0; j < num_vectors; ++j )
      {
         hypre_fprintf(fp, "vector %d\n", j );
         for (i = 0; i < size; i++)
         {
            value = data[ j * vecstride + i * idxstride ];
#ifdef HYPRE_COMPLEX
            hypre_fprintf(fp, "%.14e , %.14e\n",
                          hypre_creal(value), hypre_cimag(value));
#else
            hypre_fprintf(fp, "%.14e\n", value);
#endif
         }
      }
   }
   else
   {
      for (i = 0; i < size; i++)
      {
#ifdef HYPRE_COMPLEX
         hypre_fprintf(fp, "%.14e , %.14e\n",
                       hypre_creal(data[i]), hypre_cimag(data[i]));
#else
         hypre_fprintf(fp, "%.14e\n", data[i]);
#endif
      }
   }

   fclose(fp);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetConstantValues( hypre_Vector *v,
                                  HYPRE_Complex value )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Complex *vector_data = hypre_VectorData(v);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(v);
   HYPRE_Int      size        = hypre_VectorSize(v);
   HYPRE_Int      total_size  = size * num_vectors;

   /* Trivial case */
   if (total_size <= 0)
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_VectorMemoryLocation(v));

   //hypre_SeqVectorPrefetch(v, HYPRE_MEMORY_DEVICE);
   if (exec == HYPRE_EXEC_DEVICE)
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      hypreDevice_ComplexFilln(vector_data, total_size, value);

#elif defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL(std::fill_n, vector_data, total_size, value);

#elif defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_Int i;

      #pragma omp target teams distribute parallel for private(i) is_device_ptr(vector_data)
      for (i = 0; i < total_size; i++)
      {
         vector_data[i] = value;
      }
#endif

      hypre_SyncComputeStream(hypre_handle());
   }
   else
#endif /* defined(HYPRE_USING_GPU) */
   {
      HYPRE_Int i;

#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < total_size; i++)
      {
         vector_data[i] = value;
      }
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetRandomValues
 *
 * returns vector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetRandomValues( hypre_Vector *v,
                                HYPRE_Int     seed )
{
   HYPRE_Complex *vector_data = hypre_VectorData(v);
   HYPRE_Int      size        = hypre_VectorSize(v);
   HYPRE_Int      i;

   hypre_SeedRand(seed);
   size *= hypre_VectorNumVectors(v);

   if (hypre_GetActualMemLocation(hypre_VectorMemoryLocation(v)) == hypre_MEMORY_HOST)
   {
      /* RDF: threading this loop may cause problems because of hypre_Rand() */
      for (i = 0; i < size; i++)
      {
         vector_data[i] = 2.0 * hypre_Rand() - 1.0;
      }
   }
   else
   {
      HYPRE_Complex *h_data = hypre_TAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
      for (i = 0; i < size; i++)
      {
         h_data[i] = 2.0 * hypre_Rand() - 1.0;
      }
      hypre_TMemcpy(vector_data, h_data, HYPRE_Complex, size, hypre_VectorMemoryLocation(v),
                    HYPRE_MEMORY_HOST);
      hypre_TFree(h_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCopy
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SeqVectorCopy( hypre_Vector *x,
                     hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   size_t size = hypre_min( hypre_VectorSize(x), hypre_VectorSize(y) ) * hypre_VectorNumVectors(x);

   hypre_TMemcpy( hypre_VectorData(y),
                  hypre_VectorData(x),
                  HYPRE_Complex,
                  size,
                  hypre_VectorMemoryLocation(y),
                  hypre_VectorMemoryLocation(x) );

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneDeep
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

hypre_Vector*
hypre_SeqVectorCloneDeep_v2( hypre_Vector *x, HYPRE_MemoryLocation memory_location )
{
   HYPRE_Int      size          = hypre_VectorSize(x);
   HYPRE_Int      num_vectors   = hypre_VectorNumVectors(x);

   hypre_Vector *y = hypre_SeqMultiVectorCreate( size, num_vectors );

   hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
   hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
   hypre_VectorIndexStride(y) = hypre_VectorIndexStride(x);

   hypre_SeqVectorInitialize_v2(y, memory_location);
   hypre_SeqVectorCopy( x, y );

   return y;
}

hypre_Vector*
hypre_SeqVectorCloneDeep( hypre_Vector *x )
{
   return hypre_SeqVectorCloneDeep_v2(x, hypre_VectorMemoryLocation(x));
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneShallow
 * Returns a complete copy of x - a shallow copy, pointing the data of x
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCloneShallow( hypre_Vector *x )
{
   HYPRE_Int      size   = hypre_VectorSize(x);
   HYPRE_Int      num_vectors   = hypre_VectorNumVectors(x);
   hypre_Vector * y = hypre_SeqMultiVectorCreate( size, num_vectors );

   hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
   hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
   hypre_VectorIndexStride(y) = hypre_VectorIndexStride(x);

   hypre_VectorMemoryLocation(y) = hypre_VectorMemoryLocation(x);

   hypre_VectorData(y) = hypre_VectorData(x);
   hypre_SeqVectorSetDataOwner(y, 0);
   hypre_SeqVectorInitialize(y);

   return y;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorScale( HYPRE_Complex alpha,
                      hypre_Vector *y )
{
   /* special cases */
   if (alpha == 1.0)
   {
      return 0;
   }

   if (alpha == 0.0)
   {
      return hypre_SeqVectorSetConstantValues(y, 0.0);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Int      size   = hypre_VectorSize(y);

   size *= hypre_VectorNumVectors(y);

   //hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_CUBLAS)
   HYPRE_CUBLAS_CALL( hypre_cublas_scal(hypre_HandleCublasHandle(hypre_handle()), size, &alpha, y_data,
                                        1) );
#else
   hypreDevice_ComplexScalen( y_data, size, y_data, alpha );
#endif // #if defined(HYPRE_USING_CUBLAS)

#elif defined(HYPRE_USING_SYCL) // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_ONEMKLBLAS)
   HYPRE_ONEMKL_CALL( oneapi::mkl::blas::scal(*hypre_HandleComputeStream(hypre_handle()),
                                              size, alpha,
                                              y_data, 1).wait() );
#else
   HYPRE_ONEDPL_CALL( std::transform, y_data, y_data + size,
                      y_data, [alpha](HYPRE_Complex y) -> HYPRE_Complex { return alpha * y; } );
#endif // #if defined(HYPRE_USING_ONEMKL)

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#else // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

   HYPRE_Int i;
#if defined(HYPRE_USING_DEVICE_OPENMP)
   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data)
#elif defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      y_data[i] *= alpha;
   }

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream(hypre_handle());
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpy
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SeqVectorAxpy( HYPRE_Complex alpha,
                     hypre_Vector *x,
                     hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Int      size   = hypre_VectorSize(x);

   size *= hypre_VectorNumVectors(x);

   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_CUBLAS)
   HYPRE_CUBLAS_CALL( hypre_cublas_axpy(hypre_HandleCublasHandle(hypre_handle()), size, &alpha, x_data,
                                        1,
                                        y_data, 1) );
#else
   hypreDevice_ComplexAxpyn(x_data, size, y_data, y_data, alpha);
#endif // #if defined(HYPRE_USING_CUBLAS)

#elif defined(HYPRE_USING_SYCL) // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_ONEMKLBLAS)
   HYPRE_ONEMKL_CALL( oneapi::mkl::blas::axpy(*hypre_HandleComputeStream(hypre_handle()),
                                              size, alpha,
                                              x_data, 1, y_data, 1).wait() );
#else
   HYPRE_ONEDPL_CALL( std::transform, x_data, x_data + size, y_data, y_data,
                      [alpha](HYPRE_Complex x, HYPRE_Complex y) -> HYPRE_Complex { return alpha * x + y; } );
#endif // #if defined(HYPRE_USING_ONEMKL)

#endif // #if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP)

#else // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

   HYPRE_Int i;
#if defined(HYPRE_USING_DEVICE_OPENMP)
   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data, x_data)
#elif defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      y_data[i] += alpha * x_data[i];
   }

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream(hypre_handle());
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorElmdivpy
 *
 * Computes: y = y + x ./ b
 *
 * Notes:
 *    1) y and b must have the same sizes
 *    2) x_size can be larger than y_size
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorElmdivpy( hypre_Vector *x,
                         hypre_Vector *b,
                         hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Complex *x_data        = hypre_VectorData(x);
   HYPRE_Complex *b_data        = hypre_VectorData(b);
   HYPRE_Complex *y_data        = hypre_VectorData(y);
   HYPRE_Int      num_vectors_x = hypre_VectorNumVectors(x);
   HYPRE_Int      num_vectors_y = hypre_VectorNumVectors(y);
   HYPRE_Int      num_vectors_b = hypre_VectorNumVectors(b);
   HYPRE_Int      size          = hypre_VectorSize(y);

   /* Sanity checks */
   if (hypre_VectorSize(y) != hypre_VectorSize(b))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: sizes of y and b do not match!\n");
      return hypre_error_flag;
   }

   if (hypre_VectorSize(x) < hypre_VectorSize(y))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: x_size is smaller than y_size!\n");
      return hypre_error_flag;
   }

   /* row-wise multivec is not supported */
   hypre_assert(hypre_VectorMultiVecStorageMethod(x) == 0);
   hypre_assert(hypre_VectorMultiVecStorageMethod(b) == 0);
   hypre_assert(hypre_VectorMultiVecStorageMethod(y) == 0);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)
   //HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_VectorMemoryLocation(x), hypre_VectorMemoryLocation(b) );
   //RL: TODO back to hypre_GetExecPolicy2 later
   HYPRE_ExecutionPolicy exec = HYPRE_EXEC_DEVICE;
   if (exec == HYPRE_EXEC_DEVICE)
   {
      //TODO
      //hypre_SeqVectorElmdivpyDevice(x, b, y);
      /*
      #if defined(HYPRE_USING_DEVICE_OPENMP)
      #pragma omp target teams distribute parallel for private(i) is_device_ptr(u_data,v_data,l1_norms)
      #endif
      */

      if (num_vectors_b == 1)
      {
         if (num_vectors_x == 1)
         {
            hypreDevice_IVAXPY(size, b_data, x_data, y_data);
         }
         else if (num_vectors_x == num_vectors_y)
         {
            hypreDevice_IVAMXPMY(num_vectors_x, size, b_data, x_data, y_data);
         }
         else
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported combination of num_vectors!\n");
            return hypre_error_flag;
         }
      }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported combination of num_vectors!\n");
         return hypre_error_flag;
      }
   }
   else
#endif
   {
      HYPRE_Int i, j;

      if (num_vectors_b == 1)
      {
         if (num_vectors_x == 1 && num_vectors_y == 1)
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               y_data[i] += x_data[i] / b_data[i];
            }
         }
         else if (num_vectors_x == 2 && num_vectors_y == 2)
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               HYPRE_Complex  val = 1.0 / b_data[i];

               y_data[i]        += x_data[i]        * val;
               y_data[i + size] += x_data[i + size] * val;
            }
         }
         else if (num_vectors_x == num_vectors_y)
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i, j) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               HYPRE_Complex  val = 1.0 / b_data[i];

               for (j = 0; j < num_vectors_x; j++)
               {
                  y_data[i + size * j] += x_data[i + size * j] * val;
               }
            }
         }
         else
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported combination of num_vectors!\n");
            return hypre_error_flag;
         }
      }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported combination of num_vectors!\n");
         return hypre_error_flag;
      }
   }

#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream(hypre_handle());
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/* y[i] += x[i] / b[i] where marker[i] == marker_val */
HYPRE_Int
hypre_SeqVectorElmdivpyMarked( hypre_Vector *x,
                               hypre_Vector *b,
                               hypre_Vector *y,
                               HYPRE_Int    *marker,
                               HYPRE_Int     marker_val)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *b_data = hypre_VectorData(b);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Int      size   = hypre_VectorSize(b);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_VectorMemoryLocation(x),
                                                      hypre_VectorMemoryLocation(b) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypreDevice_IVAXPYMarked(size, b_data, x_data, y_data, marker, marker_val);
   }
   else
#endif
   {
      HYPRE_Int i;
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         if (marker[i] == marker_val)
         {
            y_data[i] += x_data[i] / b_data[i];
         }
      }
   }

#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream(hypre_handle());
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProd
 *--------------------------------------------------------------------------*/

template<HYPRE_Int SIZE>
__global__ void
hypreGPUKernel_InnerProd(hypre_DeviceItem &item,
						 const HYPRE_Int n,
						 const HYPRE_Complex * __restrict__ x,
						 const HYPRE_Complex * __restrict__ y,
						 HYPRE_Complex * __restrict__ z)
{
   volatile __shared__ HYPRE_Complex shmem[SIZE];
   const HYPRE_Int lane = threadIdx.x%warpSize;
   const HYPRE_Int warp = threadIdx.x/warpSize;
   const HYPRE_Int s = blockDim.x*gridDim.x;
   HYPRE_Int tid = hypre_gpu_get_grid_thread_id<1, 1>(item);
   HYPRE_Complex sum = 0.0;

   double z1=0;
   if (tid<n) z1=x[tid]*y[tid];
   for (tid+=s; tid<n; tid+=s)
   {
      double z2=x[tid]*y[tid];
      sum += z1;
      z1=z2;
   }
   sum += z1;

#pragma unroll
   for (HYPRE_Int i = warpSize>>1; i > 0; i >>= 1)
      sum += __shfl_down_sync(warpSize, sum, i);

   if (lane==0) shmem[warp] = sum;
   __syncthreads();

   if (SIZE>=32) { if(threadIdx.x<16) shmem[threadIdx.x]+=shmem[threadIdx.x+16]; }
   if (SIZE>=16) { if(threadIdx.x<8) shmem[threadIdx.x]+=shmem[threadIdx.x+8]; }
   if (SIZE>=8) { if(threadIdx.x<4) shmem[threadIdx.x]+=shmem[threadIdx.x+4]; }
   if (SIZE>=4) { if(threadIdx.x<2) shmem[threadIdx.x]+=shmem[threadIdx.x+2]; }
   if (threadIdx.x==0) z[blockIdx.x] = shmem[0]+shmem[1];
}


template<HYPRE_Int SIZE>
__global__ void
hypreGPUKernel_InnerProd2(hypre_DeviceItem &item,
						  const HYPRE_Int n,
						  const HYPRE_Complex * __restrict__ x,
						  HYPRE_Complex * __restrict__ z)
{
   volatile __shared__ HYPRE_Complex shmem[SIZE];
   const HYPRE_Int lane = threadIdx.x%warpSize;
   const HYPRE_Int warp = threadIdx.x/warpSize;
   const HYPRE_Int s = blockDim.x*gridDim.x;
   HYPRE_Int tid = hypre_gpu_get_grid_thread_id<1, 1>(item);
   HYPRE_Complex sum = 0.0;

   double z1=0;
   if (tid<n) z1=x[tid];
   for (tid+=s; tid<n; tid+=s)
   {
      double z2=x[tid];
      sum += z1;
      z1=z2;
   }
   sum += z1;

#pragma unroll
   for (HYPRE_Int i = warpSize>>1; i > 0; i >>= 1)
      sum += __shfl_down_sync(warpSize, sum, i);

   if (lane==0) shmem[warp] = sum;
   __syncthreads();

   if (SIZE>=32) { if(threadIdx.x<16) shmem[threadIdx.x]+=shmem[threadIdx.x+16]; }
   if (SIZE>=16) { if(threadIdx.x<8) shmem[threadIdx.x]+=shmem[threadIdx.x+8]; }
   if (SIZE>=8) { if(threadIdx.x<4) shmem[threadIdx.x]+=shmem[threadIdx.x+4]; }
   if (SIZE>=4) { if(threadIdx.x<2) shmem[threadIdx.x]+=shmem[threadIdx.x+2]; }
   if (threadIdx.x==0) z[blockIdx.x] = shmem[0]+shmem[1];
}


HYPRE_Real
hypreDevice_InnerProdPinned( hypre_Vector *x,
							 hypre_Vector *y )
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Int      n      = hypre_VectorSize(x);
   HYPRE_Real     result = 0.0;

   n *= hypre_VectorNumVectors(x);

   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   HYPRE_Int dev;
#if defined(HYPRE_USING_CUDA)
   struct cudaDeviceProp deviceProp;
   HYPRE_CUDA_CALL( cudaGetDevice(&dev) );
   HYPRE_CUDA_CALL( cudaGetDeviceProperties(&deviceProp, dev) );
#endif

#if defined(HYPRE_USING_HIP)
   hipDeviceProp_t deviceProp;
   HYPRE_HIP_CALL( hipGetDevice(&dev) );
   HYPRE_HIP_CALL( hipGetDeviceProperties(&deviceProp, dev) );
#endif

   const HYPRE_Int warpSize = deviceProp.warpSize;
   const HYPRE_Int numSMs = deviceProp.multiProcessorCount;
   const HYPRE_Int maxThreads = deviceProp.maxThreadsPerMultiProcessor;
   const HYPRE_Int numThreads = 512;
   const HYPRE_Int numBlocks = min(4*(maxThreads/numThreads)*numSMs, (n+numThreads-1)/numThreads);

   HYPRE_Complex * hwork = hypre_HandlePinnedWork(hypre_handle());

   if (!hwork)
   {
	  const HYPRE_Int N = 4*(maxThreads/numThreads)*numSMs;
	  hipHostMalloc((void **)&hwork, N*sizeof(HYPRE_Complex), hipHostRegisterMapped);
	  hypre_HandlePinnedWork(hypre_handle()) = hwork;
	  //hypre_printf("%s %s %d : host=%p, device=%p\n",__FILE__,__FUNCTION__,__LINE__,hwork,dwork);
   }
   HYPRE_Complex * dwork = (HYPRE_Complex *) hypre_HostGetDevicePointer(hwork);

   /* Do the first part on device. This step writes to Pinned memory on the host. */
   if (numThreads==128)
   {
	   HYPRE_GPU_LAUNCH( hypreGPUKernel_InnerProd<2>, dim3(numBlocks,1,1), dim3(numThreads,1,1), n, x_data, y_data, dwork );
   }
   else if (numThreads==256)
   {
	   HYPRE_GPU_LAUNCH( hypreGPUKernel_InnerProd<4>, dim3(numBlocks,1,1), dim3(numThreads,1,1), n, x_data, y_data, dwork );
   }
   else if (numThreads==512)
   {
	   HYPRE_GPU_LAUNCH( hypreGPUKernel_InnerProd<8>, dim3(numBlocks,1,1), dim3(numThreads,1,1), n, x_data, y_data, dwork );
   }
   hypre_ForceSyncComputeStream(hypre_handle());

   for (int i=0; i<numBlocks; ++i) result += hwork[i];

   return result;
}


HYPRE_Real
hypre_SeqVectorInnerProdPinned( hypre_Vector *x,
								hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Real     result = 0.0;

#ifndef HYPRE_COMPLEX

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

   result = hypreDevice_InnerProdPinned(x, y);

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#else // #ifndef HYPRE_COMPLEX

#error "Complex inner product"

#endif // #ifndef HYPRE_COMPLEX

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return result;
}




HYPRE_Int
hypreDevice_InnerProdDevice( hypre_Vector *x,
							 hypre_Vector *y,
							 HYPRE_Real * result )
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Int      n      = hypre_VectorSize(x);

   n *= hypre_VectorNumVectors(x);

   /* trivial case */
   if (n <= 0)
   {
      return hypre_error_flag;
   }

   HYPRE_Int dev;
#if defined(HYPRE_USING_CUDA)
   struct cudaDeviceProp deviceProp;
   HYPRE_CUDA_CALL( cudaGetDevice(&dev) );
   HYPRE_CUDA_CALL( cudaGetDeviceProperties(&deviceProp, dev) );
#endif

#if defined(HYPRE_USING_HIP)
   hipDeviceProp_t deviceProp;
   HYPRE_HIP_CALL( hipGetDevice(&dev) );
   HYPRE_HIP_CALL( hipGetDeviceProperties(&deviceProp, dev) );
#endif

   const HYPRE_Int warpSize = deviceProp.warpSize;
   const HYPRE_Int numSMs = deviceProp.multiProcessorCount;
   const HYPRE_Int maxThreads = deviceProp.maxThreadsPerMultiProcessor;
   const HYPRE_Int numThreads = 512;
   const HYPRE_Int numBlocks = min(4*(maxThreads/numThreads)*numSMs, (n+numThreads-1)/numThreads);

   HYPRE_Complex * dwork = hypre_HandleDeviceWork(hypre_handle());

   if (!dwork)
   {
	  const HYPRE_Int N = 4*(maxThreads/numThreads)*numSMs;
	  dwork =  hypre_CTAlloc(HYPRE_Real, N, HYPRE_MEMORY_DEVICE);
	  hypre_HandleDeviceWork(hypre_handle()) = dwork;
   }

   /* Do the first part on device. This step writes to Pinned memory on the host. */
   if (numThreads==128)
   {
	   HYPRE_GPU_LAUNCH( hypreGPUKernel_InnerProd<2>, dim3(numBlocks,1,1), dim3(numThreads,1,1), n, x_data, y_data, dwork );
	   HYPRE_GPU_LAUNCH( hypreGPUKernel_InnerProd2<2>, dim3(1,1,1), dim3(numThreads,1,1), numBlocks, dwork, result );
   }
   else if (numThreads==256)
   {
	   HYPRE_GPU_LAUNCH( hypreGPUKernel_InnerProd<4>, dim3(numBlocks,1,1), dim3(numThreads,1,1), n, x_data, y_data, dwork );
	   HYPRE_GPU_LAUNCH( hypreGPUKernel_InnerProd2<4>, dim3(1,1,1), dim3(numThreads,1,1), numBlocks, dwork, result );
   }
   else if (numThreads==512)
   {
	   HYPRE_GPU_LAUNCH( hypreGPUKernel_InnerProd<8>, dim3(numBlocks,1,1), dim3(numThreads,1,1), n, x_data, y_data, dwork );
	   HYPRE_GPU_LAUNCH( hypreGPUKernel_InnerProd2<8>, dim3(1,1,1), dim3(numThreads,1,1), numBlocks, dwork, result );
   }
   hypre_ForceSyncComputeStream(hypre_handle());

   return hypre_error_flag;
}


HYPRE_Int
hypre_SeqVectorInnerProdDevice( hypre_Vector *x,
								hypre_Vector *y,
								HYPRE_Real * result )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

#ifndef HYPRE_COMPLEX

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

   hypreDevice_InnerProdDevice(x, y, result);

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#else // #ifndef HYPRE_COMPLEX

#error "Complex inner product"

#endif // #ifndef HYPRE_COMPLEX

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}






HYPRE_Real
hypre_SeqVectorInnerProd( hypre_Vector *x,
                          hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Int      size   = hypre_VectorSize(x);
   HYPRE_Real     result = 0.0;

   size *= hypre_VectorNumVectors(x);

   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

#ifndef HYPRE_COMPLEX

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_CUBLAS)
   HYPRE_CUBLAS_CALL( hypre_cublas_dot(hypre_HandleCublasHandle(hypre_handle()), size, x_data, 1,
                                       y_data, 1,
                                       &result) );
#else
   result = HYPRE_THRUST_CALL( inner_product, x_data, x_data + size, y_data, 0.0 );
#endif // #if defined(HYPRE_USING_CUBLAS)

#elif defined(HYPRE_USING_SYCL) // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_ONEMKLBLAS)
   HYPRE_Real *result_dev = hypre_CTAlloc(HYPRE_Real, 1, HYPRE_MEMORY_DEVICE);
   HYPRE_ONEMKL_CALL( oneapi::mkl::blas::dot(*hypre_HandleComputeStream(hypre_handle()),
                                             size, x_data, 1,
                                             y_data, 1, result_dev).wait() );
   hypre_TMemcpy(&result, result_dev, HYPRE_Real, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TFree(result_dev, HYPRE_MEMORY_DEVICE);
#else
   result = HYPRE_ONEDPL_CALL( std::transform_reduce, x_data, x_data + size, y_data, 0.0 );
#endif // #if defined(HYPRE_USING_ONEMKLBLAS)

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#else // #ifndef HYPRE_COMPLEX
   /* TODO */
#error "Complex inner product"
#endif // #ifndef HYPRE_COMPLEX

#else // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

   HYPRE_Int i;
#if defined(HYPRE_USING_DEVICE_OPENMP)
   #pragma omp target teams  distribute  parallel for private(i) reduction(+:result) is_device_ptr(y_data,x_data) map(result)
#elif defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) reduction(+:result) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      result += hypre_conj(y_data[i]) * x_data[i];
   }

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream(hypre_handle());
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return result;
}


//TODO

/*--------------------------------------------------------------------------
 * hypre_VectorSumElts:
 * Returns the sum of all vector elements.
 *--------------------------------------------------------------------------*/

HYPRE_Complex hypre_SeqVectorSumElts( hypre_Vector *vector )
{
   HYPRE_Complex  sum = 0;
   HYPRE_Complex *data = hypre_VectorData( vector );
   HYPRE_Int      size = hypre_VectorSize( vector );
   HYPRE_Int      i;

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:sum) HYPRE_SMP_SCHEDULE
#endif
   for ( i = 0; i < size; ++i ) { sum += data[i]; }

   return sum;
}

HYPRE_Int
hypre_SeqVectorPrefetch( hypre_Vector *x, HYPRE_MemoryLocation memory_location)
{
#ifdef HYPRE_USING_UNIFIED_MEMORY
   if (hypre_VectorMemoryLocation(x) != HYPRE_MEMORY_DEVICE)
   {
      /* hypre_error_w_msg(HYPRE_ERROR_GENERIC," Error! CUDA Prefetch with non-unified momory\n");*/
      return 1;
   }

   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Int      size   = hypre_VectorSize(x) * hypre_VectorNumVectors(x);

   if (size == 0)
   {
      return hypre_error_flag;
   }

   hypre_MemPrefetch(x_data, sizeof(HYPRE_Complex)*size, memory_location);
#endif

   return hypre_error_flag;
}

#if 0
/* y[i] = max(alpha*x[i], beta*y[i]) */
HYPRE_Int
hypre_SeqVectorMax( HYPRE_Complex alpha,
                    hypre_Vector *x,
                    HYPRE_Complex beta,
                    hypre_Vector *y     )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Int      size   = hypre_VectorSize(x);

   size *= hypre_VectorNumVectors(x);

   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);

   thrust::maximum<HYPRE_Complex> mx;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_THRUST_CALL( transform,
                      thrust::make_transform_iterator(x_data,        alpha * _1),
                      thrust::make_transform_iterator(x_data + size, alpha * _1),
                      thrust::make_transform_iterator(y_data,        beta  * _1),
                      y_data,
                      mx );
#else
   HYPRE_Int i;
#if defined(HYPRE_USING_DEVICE_OPENMP)
   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data, x_data)
#elif defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      y_data[i] += hypre_max(alpha * x_data[i], beta * y_data[i]);
   }

#endif /* defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */

   hypre_SyncComputeStream(hypre_handle());

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}
#endif
