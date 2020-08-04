/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * COGMRESv2 gmres
 *
 *****************************************************************************/

#ifndef hypre_KRYLOV_COGMRESv2_HEADER
#define hypre_KRYLOV_COGMRESv2_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic COGMRESv2 Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic COGMRESv2 linear solver interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2Data and hypre_COGMRESv2Functions
 *--------------------------------------------------------------------------*/

/**
 * @name COGMRESv2 structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_COGMRESv2Functions} object ...
 **/

typedef struct
{
   void *       (*CAlloc)        ( size_t count, size_t elt_size );
   HYPRE_Int    (*Free)          ( void *ptr );
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   void *       (*CreateVectorArray)  ( HYPRE_Int size, void *vectors );
   HYPRE_Int    (*DestroyVector) ( void *vector );
   void *       (*MatvecCreate)  ( void *A, void *x );
   HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                   void *x, HYPRE_Complex beta, void *y );
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   HYPRE_Real   (*InnerProd)     ( void *x, void *y );
   HYPRE_Int    (*MassInnerProd) ( void *x, void **p, HYPRE_Int k, HYPRE_int unroll, void *result);
   HYPRE_Int    (*MassDotpTwo)( void *x, void *y, void **p, HYPRE_Int k, void *result_x, HYPRE_int unroll, void *result_y);
   HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   HYPRE_Int    (*ClearVector)   ( void *x );
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x );
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y );
   HYPRE_Int    (*MassAxpy)      ( HYPRE_Complex *alpha, void **x, void *y, HYPRE_Int k, HYPRE_Int unroll);
   HYPRE_Int    (*precond)       ();
   HYPRE_Int    (*precond_setup) ();

   HYPRE_Int    (*modify_pc)(void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm );
} hypre_COGMRESv2Functions;

/**
 * The {\tt hypre\_COGMRESv2Data} object ...
 **/

typedef struct
{
   HYPRE_Int      k_dim;
   HYPRE_Int      unroll;
   HYPRE_Int      cgs;
   HYPRE_Int      min_iter;
   HYPRE_Int      max_iter;
   HYPRE_Int      rel_change;
   HYPRE_Int      skip_real_r_check;
   HYPRE_Int      converged;
   HYPRE_Real     tol;
   HYPRE_Real     cf_tol;
   HYPRE_Real     a_tol;
   HYPRE_Real     rel_residual_norm;

   void  *A;
   void  *r;
   void  *w;
   void  *w_2;
   void  **p;

   void    *matvec_data;
   void    *precond_data;

   hypre_COGMRESv2Functions * functions;

   /* log info (always logged) */
   HYPRE_Int      num_iterations;
 
   HYPRE_Int     print_level; /* printing when print_level>0 */
   HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   HYPRE_Real  *norms;
   char    *log_file_name;

} hypre_COGMRESv2Data;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic COGMRESv2 Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

hypre_COGMRESv2Functions *
hypre_COGMRESv2FunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size ),
   HYPRE_Int    (*Free)          ( void *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   void *       (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                   void *x, HYPRE_Complex beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*MassInnerProd) ( void *x, void **p, HYPRE_Int k, HYPRE_Int unroll, void *result),
   HYPRE_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, HYPRE_Int k, HYPRE_Int unroll, void *result_x, void *result_y),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
   HYPRE_Int    (*MassAxpy)      ( HYPRE_Complex *alpha, void **x, void *y, HYPRE_Int k, HYPRE_Int unroll),   
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );

/**
 * Description...
 *
 * @param param [IN] ...
 **/

void *
hypre_COGMRESv2Create( hypre_COGMRESv2Functions *cogmres_v2_functions );

#ifdef __cplusplus
}
#endif
#endif
