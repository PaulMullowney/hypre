/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2Create
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2Create( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_COGMRESv2Functions * cogmres_v2_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   cogmres_v2_functions =
      hypre_COGMRESv2FunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovMassInnerProd, 
         hypre_ParKrylovMassDotpTwo, hypre_ParKrylovCopyVector,
         //hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,hypre_ParKrylovMassAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
   *solver = ( (HYPRE_Solver) hypre_COGMRESv2Create( cogmres_v2_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2Destroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRCOGMRESv2Destroy( HYPRE_Solver solver )
{
   return( hypre_COGMRESv2Destroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2Setup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRCOGMRESv2Setup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_COGMRESv2Setup( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2Solve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRCOGMRESv2Solve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_COGMRESv2Solve( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetKDim( HYPRE_Solver solver,
                          HYPRE_Int             k_dim    )
{
   return( HYPRE_COGMRESv2SetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetUnroll
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetUnroll( HYPRE_Solver solver,
                          HYPRE_Int             unroll    )
{
   return( HYPRE_COGMRESv2SetUnroll( solver, unroll ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetCGS
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetCGS( HYPRE_Solver solver,
                          HYPRE_Int             cgs    )
{
   return( HYPRE_COGMRESv2SetCGS( solver, cgs ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetTol( HYPRE_Solver solver,
                         HYPRE_Real         tol    )
{
   return( HYPRE_COGMRESv2SetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetAbsoluteTol( HYPRE_Solver solver,
                                 HYPRE_Real         a_tol    )
{
   return( HYPRE_COGMRESv2SetAbsoluteTol( solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetMinIter( HYPRE_Solver solver,
                             HYPRE_Int          min_iter )
{
   return( HYPRE_COGMRESv2SetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int          max_iter )
{
   return( HYPRE_COGMRESv2SetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToParSolverFcn  precond,
                             HYPRE_PtrToParSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return( HYPRE_COGMRESv2SetPrecond( solver,
                                  (HYPRE_PtrToSolverFcn) precond,
                                  (HYPRE_PtrToSolverFcn) precond_setup,
                                  precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2GetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2GetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_COGMRESv2GetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetLogging( HYPRE_Solver solver,
                             HYPRE_Int logging)
{
   return( HYPRE_COGMRESv2SetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2SetPrintLevel( HYPRE_Solver solver,
                                HYPRE_Int print_level)
{
   return( HYPRE_COGMRESv2SetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2GetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2GetNumIterations( HYPRE_Solver  solver,
                                   HYPRE_Int    *num_iterations )
{
   return( HYPRE_COGMRESv2GetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2GetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2GetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               HYPRE_Real   *norm   )
{
   return( HYPRE_COGMRESv2GetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2GetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESv2GetResidual( HYPRE_Solver  solver,
                                HYPRE_ParVector *residual)
{
   return( HYPRE_COGMRESv2GetResidual( solver, (void *) residual ) );
}
