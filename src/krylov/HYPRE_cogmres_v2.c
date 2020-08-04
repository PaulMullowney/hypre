/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_COGMRESv2 interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2Destroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
HYPRE_Int 
HYPRE_ParCSRCOGMRESv2Destroy( HYPRE_Solver solver )
{
   return( hypre_COGMRESv2Destroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2Setup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_COGMRESv2Setup( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_COGMRESv2Setup( solver,
                             A,
                             b,
                             x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2Solve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_COGMRESv2Solve( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_COGMRESv2Solve( solver,
                             A,
                             b,
                             x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetKDim, HYPRE_COGMRESv2GetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetKDim( HYPRE_Solver solver,
                          HYPRE_Int             k_dim    )
{
   return( hypre_COGMRESv2SetKDim( (void *) solver, k_dim ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetKDim( HYPRE_Solver solver,
                          HYPRE_Int           * k_dim    )
{
   return( hypre_COGMRESv2GetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetUnroll, HYPRE_COGMRESv2GetUnroll
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetUnroll( HYPRE_Solver solver,
                          HYPRE_Int             unroll    )
{
   return( hypre_COGMRESv2SetUnroll( (void *) solver, unroll ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetUnroll( HYPRE_Solver solver,
                          HYPRE_Int           * unroll    )
{
   return( hypre_COGMRESv2GetUnroll( (void *) solver, unroll ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetCGS, HYPRE_COGMRESv2GetCGS
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetCGS( HYPRE_Solver solver,
                          HYPRE_Int             cgs    )
{
   return( hypre_COGMRESv2SetCGS( (void *) solver, cgs ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetCGS( HYPRE_Solver solver,
                          HYPRE_Int           * cgs    )
{
   return( hypre_COGMRESv2GetCGS( (void *) solver, cgs ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetTol, HYPRE_COGMRESv2GetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetTol( HYPRE_Solver solver,
                         HYPRE_Real         tol    )
{
   return( hypre_COGMRESv2SetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetTol( HYPRE_Solver solver,
                         HYPRE_Real       * tol    )
{
   return( hypre_COGMRESv2GetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetAbsoluteTol, HYPRE_COGMRESv2GetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetAbsoluteTol( HYPRE_Solver solver,
                         HYPRE_Real         a_tol    )
{
   return( hypre_COGMRESv2SetAbsoluteTol( (void *) solver, a_tol ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetAbsoluteTol( HYPRE_Solver solver,
                         HYPRE_Real       * a_tol    )
{
   return( hypre_COGMRESv2GetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetConvergenceFactorTol, HYPRE_COGMRESv2GetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetConvergenceFactorTol( HYPRE_Solver solver,
                         HYPRE_Real         cf_tol    )
{
   return( hypre_COGMRESv2SetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetConvergenceFactorTol( HYPRE_Solver solver,
                         HYPRE_Real       * cf_tol    )
{
   return( hypre_COGMRESv2GetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetMinIter, HYPRE_COGMRESv2GetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetMinIter( HYPRE_Solver solver,
                             HYPRE_Int          min_iter )
{
   return( hypre_COGMRESv2SetMinIter( (void *) solver, min_iter ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetMinIter( HYPRE_Solver solver,
                             HYPRE_Int        * min_iter )
{
   return( hypre_COGMRESv2GetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetMaxIter, HYPRE_COGMRESv2GetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int          max_iter )
{
   return( hypre_COGMRESv2SetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int        * max_iter )
{
   return( hypre_COGMRESv2GetMaxIter( (void *) solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToSolverFcn  precond,
                             HYPRE_PtrToSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return( hypre_COGMRESv2SetPrecond( (void *) solver,
									  (HYPRE_Int (*)(void*, void*, void*, void*))precond,
									  (HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
									  (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2GetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2GetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_COGMRESv2GetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetPrintLevel, HYPRE_COGMRESv2GetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetPrintLevel( HYPRE_Solver solver,
                        HYPRE_Int          level )
{
   return( hypre_COGMRESv2SetPrintLevel( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetPrintLevel( HYPRE_Solver solver,
                        HYPRE_Int        * level )
{
   return( hypre_COGMRESv2GetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetLogging, HYPRE_COGMRESv2GetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2SetLogging( HYPRE_Solver solver,
                     HYPRE_Int          level )
{
   return( hypre_COGMRESv2SetLogging( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_COGMRESv2GetLogging( HYPRE_Solver solver,
                     HYPRE_Int        * level )
{
   return( hypre_COGMRESv2GetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2GetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2GetNumIterations( HYPRE_Solver  solver,
                                   HYPRE_Int                *num_iterations )
{
   return( hypre_COGMRESv2GetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2GetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2GetConverged( HYPRE_Solver  solver,
                         HYPRE_Int                *converged )
{
   return( hypre_COGMRESv2GetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2GetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESv2GetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               HYPRE_Real         *norm   )
{
   return( hypre_COGMRESv2GetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2GetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_COGMRESv2GetResidual( HYPRE_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return hypre_COGMRESv2GetResidual( (void *) solver, (void **) residual );

}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESv2SetModifyPC
 *--------------------------------------------------------------------------*/
 

HYPRE_Int HYPRE_COGMRESv2SetModifyPC( HYPRE_Solver  solver,
             HYPRE_Int (*modify_pc)(HYPRE_Solver, HYPRE_Int, HYPRE_Real) )
{
   return hypre_COGMRESv2SetModifyPC( (void *) solver, (HYPRE_Int(*)(void*, HYPRE_Int, HYPRE_Real))modify_pc);
   
}


