/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRCOGMRESv2 Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif
    
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2Create
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2create, HYPRE_PARCSRCOGMRESv2CREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *solver,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2Create(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2Destroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcogmresv2destroy, HYPRE_PARCSRCOGMRESv2DESTROY)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2Destroy(
           hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2Setup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcogmresv2setup, HYPRE_PARCSRCOGMRESv2SETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2Setup(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassObj (HYPRE_ParVector, b),
           hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2Solve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcogmresv2solve, HYPRE_PARCSRCOGMRESv2SOLVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2Solve(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassObj (HYPRE_ParVector, b),
           hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2setkdim, HYPRE_PARCSRCOGMRESv2SETKDIM)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *kdim,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2SetKDim(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (kdim)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetUnroll
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2setunroll, HYPRE_PARCSRCOGMRESv2SETUNROLL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *unroll,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2SetUnroll(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (unroll)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetCGS
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2setcgs, HYPRE_PARCSRCOGMRESv2SETCGS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *cgs,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2SetCGS(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (cgs)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2settol, HYPRE_PARCSRCOGMRESv2SETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Real *tol,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2SetTol(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2setabsolutet, HYPRE_PARCSRCOGMRESv2SETABSOLUTET)
   ( hypre_F90_Obj *solver,
     hypre_F90_Real *tol,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2SetAbsoluteTol(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2setminiter, HYPRE_PARCSRCOGMRESv2SETMINITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *min_iter,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2SetMinIter(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2setmaxiter, HYPRE_PARCSRCOGMRESv2SETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2SetMaxIter(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (max_iter) ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2setprecond, HYPRE_PARCSRCOGMRESv2SETPRECOND)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *precond_id,
     hypre_F90_Obj *precond_solver,
     hypre_F90_Int *ierr          )
{
/*------------------------------------------------------------
 * The precond_id flags mean :
 * 0 - no preconditioner
 * 1 - set up a ds preconditioner
 * 2 - set up an amg preconditioner
 * 3 - set up a pilut preconditioner
 * 4 - set up a parasails preconditioner
 * 5 - set up a Euclid preconditioner
 *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCOGMRESv2SetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_ParCSRDiagScale,
              HYPRE_ParCSRDiagScaleSetup,
              NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCOGMRESv2SetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_BoomerAMGSolve,
              HYPRE_BoomerAMGSetup,
              (HYPRE_Solver)       *precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCOGMRESv2SetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_ParCSRPilutSolve,
              HYPRE_ParCSRPilutSetup,
              (HYPRE_Solver)       *precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCOGMRESv2SetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_ParCSRParaSailsSolve,
              HYPRE_ParCSRParaSailsSetup,
              (HYPRE_Solver)       *precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCOGMRESv2SetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_EuclidSolve,
              HYPRE_EuclidSetup,
              (HYPRE_Solver)       *precond_solver ) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2GetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2getprecond, HYPRE_PARCSRCOGMRESv2GETPRECOND)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *precond_solver_ptr,
     hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2GetPrecond(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObjRef (HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2setlogging, HYPRE_PARCSRCOGMRESv2SETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2SetLogging(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2SetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2setprintleve, HYPRE_PARCSRCOGMRESv2SETPRINTLEVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *print_level,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2SetPrintLevel(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2GetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2getnumiterat, HYPRE_PARCSRCOGMRESv2GETNUMITERAT)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_iterations,
     hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2GetNumIterations(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESv2GetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresv2getfinalrela, HYPRE_PARCSRCOGMRESv2GETFINALRELA)
   ( hypre_F90_Obj *solver,
     hypre_F90_Real *norm,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCOGMRESv2GetFinalRelativeResidualNorm(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassRealRef (norm)    ) );
}
    
#ifdef __cplusplus
}
#endif
