/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/


#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"	

#define DEBUG_FAC 0
#define DEBUGGING_MESSAGES 0

HYPRE_Int
FAC_Cycle(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int first_iteration);

HYPRE_Int 
FAC_FCycle(void *amg_vdata, HYPRE_Int first_iteration);

HYPRE_Int
FAC_Cycle_timed(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int time_part);

HYPRE_Int 
FAC_FCycle_timed(void *amg_vdata, HYPRE_Int time_part);

HYPRE_Int
FAC_Interpolate( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c );

HYPRE_Int
FAC_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int first_iteration );

HYPRE_Int
FAC_Simple_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c );

HYPRE_Int
FAC_Relax( hypre_ParAMGData *amg_data, hypre_ParCompGrid *compGrid, HYPRE_Int type, HYPRE_Int level );

HYPRE_Int
FAC_Jacobi( hypre_ParAMGData *amg_data, hypre_ParCompGrid *compGrid, HYPRE_Int level );

HYPRE_Int
FAC_GaussSeidel( hypre_ParCompGrid *compGrid );

HYPRE_Int
FAC_Cheby( hypre_ParAMGData *amg_data, hypre_ParCompGrid *compGrid, HYPRE_Int level );

HYPRE_Int
hypre_BoomerAMGDD_FAC_Cycle( void *amg_vdata, HYPRE_Int first_iteration )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int cycle_type = hypre_ParAMGDataFACCycleType(amg_data);

   if (cycle_type == 1 || cycle_type == 2) FAC_Cycle(amg_vdata, hypre_ParAMGDataAMGDDStartLevel(amg_data), cycle_type, first_iteration);
   else if (cycle_type == 3) FAC_FCycle(amg_vdata, first_iteration);
   else
   {
      if (myid == 0) hypre_printf("Error: unknown cycle type\n");
   }

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_Cycle_timed( void *amg_vdata, HYPRE_Int time_part )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int cycle_type = hypre_ParAMGDataFACCycleType(amg_data);

   if (cycle_type == 1 || cycle_type == 2) FAC_Cycle_timed(amg_vdata, hypre_ParAMGDataAMGDDStartLevel(amg_data), cycle_type, time_part);
   else if (cycle_type == 3) FAC_FCycle_timed(amg_vdata, time_part);
   else
   {
      if (myid == 0) hypre_printf("Error: unknown cycle type\n");
   }

   return 0;
}

HYPRE_Int FAC_Cycle(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int first_iteration)
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int i;

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);


   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // Relax on the real nodes
   #if DEBUGGING_MESSAGES
   printf("Rank %d, relax on level %d\n", myid, level);
   #endif
   for (i = 0; i < numRelax[1]; i++) FAC_Relax( amg_data, compGrid[level], relax_type, level );

   // Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
   if (level < transition_level)
   {
      #if DEBUGGING_MESSAGES
      printf("Rank %d, restrict on level %d\n", myid, level);
      #endif
      FAC_Restrict( compGrid[level], compGrid[level+1], first_iteration );
      hypre_SeqVectorSetConstantValues( hypre_ParCompGridS(compGrid[level]), 0.0 );
      hypre_SeqVectorSetConstantValues( hypre_ParCompGridT(compGrid[level]), 0.0 );
   }
   else FAC_Simple_Restrict( compGrid[level], compGrid[level+1] );

   //  Either solve on the coarse level or recurse
   if (level+1 == num_levels-1)
   {
      #if DEBUGGING_MESSAGES
      printf("Rank %d, coarse solve on level %d\n", myid, num_levels-1);
      #endif
      for (i = 0; i < numRelax[3]; i++) FAC_Relax( amg_data, compGrid[num_levels-1], relax_type, num_levels-1 );
   }
   else for (i = 0; i < cycle_type; i++)
   {
      #if DEBUGGING_MESSAGES
      printf("Rank %d, recurse on level %d\n", myid, level);
      #endif
      FAC_Cycle(amg_vdata, level+1, cycle_type, first_iteration);
      first_iteration = 0;
   }

   // Interpolate up and relax
   #if DEBUGGING_MESSAGES
   printf("Rank %d, interpolate on level %d\n", myid, level);
   #endif
   FAC_Interpolate( compGrid[level], compGrid[level+1] );

   #if DEBUGGING_MESSAGES
   printf("Rank %d, relax on level %d\n", myid, level);
   #endif
   for (i = 0; i < numRelax[2]; i++) FAC_Relax( amg_data, compGrid[level], relax_type, level );

   return 0;
}

HYPRE_Int FAC_FCycle(void *amg_vdata, HYPRE_Int first_iteration)
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

   HYPRE_Int level, i;

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // ... work down to coarsest ... Note: proper restricted values already stored on and above transition level
   if (!first_iteration)
   {
      for (level = hypre_ParAMGDataAMGDDStartLevel(amg_data); level < num_levels - 1; level++)
      {
         // Restrict down from the transition level
         if (level < transition_level)
         {
            FAC_Restrict( compGrid[level], compGrid[level+1], 0 );
            hypre_SeqVectorSetConstantValues( hypre_ParCompGridS(compGrid[level]), 0.0 );
            hypre_SeqVectorSetConstantValues( hypre_ParCompGridT(compGrid[level]), 0.0 );
         }
         else FAC_Simple_Restrict( compGrid[level], compGrid[level+1] );
      }
   }

   //  ... solve on coarsest level ...
   for (i = 0; i < numRelax[3]; i++) FAC_Relax( amg_data, compGrid[num_levels-1], relax_type, num_levels-1 );

   // ... and work back up to the finest
   for (level = num_levels - 2; level > hypre_ParAMGDataAMGDDStartLevel(amg_data)-1; level--)
   {
      // Interpolate up and relax
      FAC_Interpolate( compGrid[level], compGrid[level+1] );

      // V-cycle
      FAC_Cycle( amg_vdata, level, 1, 0 );
   }

   return 0;
}

HYPRE_Int FAC_Cycle_timed(void *amg_vdata, HYPRE_Int level, HYPRE_Int cycle_type, HYPRE_Int time_part)
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int i, j; // loop variables

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // Relax on the real nodes
   if (time_part == 1) for (i = 0; i < numRelax[1]; i++) FAC_Relax( amg_data, compGrid[level], relax_type, level );

   // Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
   if (time_part == 2)
   { 
      if (level < transition_level)
      {
         FAC_Restrict( compGrid[level], compGrid[level+1], 1 );
         hypre_SeqVectorSetConstantValues( hypre_ParCompGridS(compGrid[level]), 0.0 );
         hypre_SeqVectorSetConstantValues( hypre_ParCompGridT(compGrid[level]), 0.0 );
      }
      else FAC_Simple_Restrict( compGrid[level], compGrid[level+1] ); // !!! Todo: I don't use s and t here, right?
   }

   //  Either solve on the coarse level or recurse
   if (level+1 == num_levels-1) for (i = 0; i < numRelax[3]; i++) FAC_Relax( amg_data, compGrid[num_levels-1], relax_type, num_levels-1 );
   else for (i = 0; i < cycle_type; i++) FAC_Cycle_timed(amg_vdata, level+1, cycle_type, time_part);

   // Interpolate up and relax
   if (time_part == 3) FAC_Interpolate( compGrid[level], compGrid[level+1] );

   if (time_part == 1) for (i = 0; i < numRelax[2]; i++) FAC_Relax( amg_data, compGrid[level], relax_type, level );

   return 0;
}

HYPRE_Int FAC_FCycle_timed(void *amg_vdata, HYPRE_Int time_part)
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );

   HYPRE_Int level, i;

   // Get the AMG structure
   hypre_ParAMGData   *amg_data = (hypre_ParAMGData*) amg_vdata;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;
   HYPRE_Int relax_type = hypre_ParAMGDataFACRelaxType(amg_data);
   HYPRE_Int *numRelax = hypre_ParAMGDataNumGridSweeps(amg_data);

   // Get the composite grid
   hypre_ParCompGrid          **compGrid = hypre_ParAMGDataCompGrid(amg_data);

   // ... work down to coarsest ... Note: proper restricted values already stored on and above transition level
   for (level = hypre_ParAMGDataAMGDDStartLevel(amg_data); level < num_levels - 1; level++)
   {
      // Restrict down from the transition level
      if (time_part == 2)
      {
         if (level < transition_level)
         {
            FAC_Restrict( compGrid[level], compGrid[level+1], 0 );
            hypre_SeqVectorSetConstantValues( hypre_ParCompGridS(compGrid[level]), 0.0 );
            hypre_SeqVectorSetConstantValues( hypre_ParCompGridT(compGrid[level]), 0.0 );
         }
         else FAC_Simple_Restrict( compGrid[level], compGrid[level+1] );
      }
   }

   //  ... solve on coarsest level ...
   if (time_part == 1) for (i = 0; i < numRelax[3]; i++) FAC_Relax( amg_data, compGrid[num_levels-1], relax_type, level );

   // ... and work back up to the finest
   for (level = num_levels - 2; level > hypre_ParAMGDataAMGDDStartLevel(amg_data)-1; level--)
   {
      // Interpolate up and relax
      if (time_part == 3) FAC_Interpolate( compGrid[level], compGrid[level+1] );

      // V-cycle
      FAC_Cycle_timed( amg_vdata, level, 1, time_part );
   }

   return 0;
}

HYPRE_Int
FAC_Interpolate( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c )
{
   hypre_CSRMatrixMatvec(1.0, hypre_ParCompGridP(compGrid_f), hypre_ParCompGridU(compGrid_c), 1.0, hypre_ParCompGridU(compGrid_f));
   return 0;
}

HYPRE_Int
FAC_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c, HYPRE_Int first_iteration )
{
   // Recalculate residual on coarse grid
   if (!first_iteration) hypre_CSRMatrixMatvec(-1.0, hypre_ParCompGridA(compGrid_c), hypre_ParCompGridU(compGrid_c), 1.0, hypre_ParCompGridF(compGrid_c));

   // Get update: s_l <- A_lt_l + s_l (NOTE: I'm assuming here that A is symmetric and computing s_l <- A_l^Tt_l + s_l)
   // hypre_CSRMatrixMatvecT(1.0, hypre_ParCompGridA(compGrid_f), hypre_ParCompGridT(compGrid_f), 1.0, hypre_ParCompGridS(compGrid_f));
   hypre_CSRMatrixMatvec(1.0, hypre_ParCompGridAT(compGrid_f), hypre_ParCompGridT(compGrid_f), 1.0, hypre_ParCompGridS(compGrid_f));

   // If we need to preserve the updates on the next level !!! Do we need this if statement? Implications? Still need to generally make sure transition level stuff still works...
   if (hypre_ParCompGridS(compGrid_c))
   {
      // hypre_CSRMatrixMatvecT(1.0, hypre_ParCompGridP(compGrid_f), hypre_ParCompGridS(compGrid_f), 0.0, hypre_ParCompGridS(compGrid_c));
      hypre_CSRMatrixMatvec(1.0, hypre_ParCompGridR(compGrid_f), hypre_ParCompGridS(compGrid_f), 0.0, hypre_ParCompGridS(compGrid_c));

      // Subtract restricted update from recalculated residual: f_{l+1} <- f_{l+1} - s_{l+1}
      hypre_SeqVectorAxpy(-1.0, hypre_ParCompGridS(compGrid_c), hypre_ParCompGridF(compGrid_c));
   }
   else
   {
      // Restrict and subtract update from recalculated residual: f_{l+1} <- f_{l+1} - P_l^Ts_l
      // hypre_CSRMatrixMatvecT(-1.0, hypre_ParCompGridP(compGrid_f), hypre_ParCompGridS(compGrid_f), 1.0, hypre_ParCompGridF(compGrid_c));
      hypre_CSRMatrixMatvec(-1.0, hypre_ParCompGridR(compGrid_f), hypre_ParCompGridS(compGrid_f), 1.0, hypre_ParCompGridF(compGrid_c));
   }

   // Zero out initial guess on coarse grid
   hypre_SeqVectorSetConstantValues(hypre_ParCompGridU(compGrid_c), 0.0);

   return 0;
}

HYPRE_Int
FAC_Simple_Restrict( hypre_ParCompGrid *compGrid_f, hypre_ParCompGrid *compGrid_c )
{
   // Calculate fine grid residuals and restrict
   if (!hypre_ParCompGridTemp(compGrid_f))
   {      
      hypre_ParCompGridTemp(compGrid_f) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid_f));
      hypre_SeqVectorInitialize(hypre_ParCompGridTemp(compGrid_f));
   }
   hypre_CSRMatrixMatvecOutOfPlace(-1.0, hypre_ParCompGridA(compGrid_f), hypre_ParCompGridU(compGrid_f), 1.0, hypre_ParCompGridF(compGrid_f), hypre_ParCompGridTemp(compGrid_f), 0);
   // hypre_CSRMatrixMatvecT(1.0, hypre_ParCompGridP(compGrid_f), hypre_ParCompGridTemp(compGrid_f), 0.0, hypre_ParCompGridF(compGrid_c));
   hypre_CSRMatrixMatvec(1.0, hypre_ParCompGridR(compGrid_f), hypre_ParCompGridTemp(compGrid_f), 0.0, hypre_ParCompGridF(compGrid_c));
   
   // Zero out initial guess on coarse grid
   hypre_SeqVectorSetConstantValues(hypre_ParCompGridU(compGrid_c), 0.0);

   return 0;
}

HYPRE_Int
FAC_Relax( hypre_ParAMGData *amg_data, hypre_ParCompGrid *compGrid, HYPRE_Int type, HYPRE_Int level )
{
   if (type == 0) FAC_Jacobi(amg_data, compGrid, level);
   else if (type == 1) FAC_GaussSeidel(compGrid);
   else if (type == 2) FAC_Cheby(amg_data, compGrid, level);
	return 0;
}

HYPRE_Int
FAC_Jacobi( hypre_ParAMGData *amg_data, hypre_ParCompGrid *compGrid, HYPRE_Int level )
{
   HYPRE_Int i,j; 
   HYPRE_Real relax_weight = hypre_ParAMGDataRelaxWeight(amg_data)[level];

   // Calculate l1_norms if necessary
   if (!hypre_ParCompGridL1Norms(compGrid))
   {
      hypre_ParCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_SHARED);
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
      {
         for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
         {
            // hypre_ParCompGridL1Norms(compGrid)[i] += fabs(hypre_ParCompGridAData(compGrid)[j]);
            if (hypre_ParCompGridAColInd(compGrid)[j] == i) hypre_ParCompGridL1Norms(compGrid)[i] = hypre_ParCompGridAData(compGrid)[j];
         }
      }
   }

   // Allocate temporary vector if necessary
   if (!hypre_ParCompGridTemp(compGrid))
   {      
      hypre_ParCompGridTemp(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
      hypre_SeqVectorInitialize(hypre_ParCompGridTemp(compGrid));
   }

   // !!! Adapted from par_relax case 7. Right now, I'm doing straight ahead unweighted Jacobi.
   /*-----------------------------------------------------------------
   * Copy f into temporary vector.
   *-----------------------------------------------------------------*/
   #if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
   hypre_SeqVectorPrefetchToDevice(hypre_ParCompGridTemp(compGrid));
   hypre_SeqVectorPrefetchToDevice(hypre_ParCompGridF(compGrid));
   VecCopy(hypre_VectorData(hypre_ParCompGridTemp(compGrid)),hypre_VectorData(hypre_ParCompGridF(compGrid)),hypre_VectorSize(hypre_ParCompGridF(compGrid)),HYPRE_STREAM(4));
   #else
   hypre_SeqVectorCopy(hypre_ParCompGridF(compGrid),hypre_ParCompGridTemp(compGrid));
   #endif
   hypre_CSRMatrixMatvec(-relax_weight, hypre_ParCompGridA(compGrid), hypre_ParCompGridU(compGrid), relax_weight, hypre_ParCompGridTemp(compGrid));
   #if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
   VecScale(hypre_VectorData(hypre_ParCompGridU(compGrid)),hypre_VectorData(hypre_ParCompGridTemp(compGrid)),hypre_ParCompGridL1Norms(compGrid),hypre_ParCompGridNumRealNodes(compGrid),HYPRE_STREAM(4));
   VecScale(hypre_VectorData(hypre_ParCompGridT(compGrid)),hypre_VectorData(hypre_ParCompGridTemp(compGrid)),hypre_ParCompGridL1Norms(compGrid),hypre_ParCompGridNumRealNodes(compGrid),HYPRE_STREAM(4));
   #else
   for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++)
   {
      hypre_VectorData(hypre_ParCompGridU(compGrid))[i] += hypre_VectorData(hypre_ParCompGridTemp(compGrid))[i] / hypre_ParCompGridL1Norms(compGrid)[i];
   }
   for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++)
   {
      hypre_VectorData(hypre_ParCompGridT(compGrid))[i] += hypre_VectorData(hypre_ParCompGridTemp(compGrid))[i] / hypre_ParCompGridL1Norms(compGrid)[i];
   }
   #endif

   return 0;
}

HYPRE_Int
FAC_GaussSeidel( hypre_ParCompGrid *compGrid )
{
   HYPRE_Int               i, j; // loop variables
   HYPRE_Int               is_real;
   HYPRE_Complex           diag; // placeholder for the diagonal of A
   HYPRE_Complex           u_before;

   // Do Gauss-Seidel relaxation on the real nodes
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      if (hypre_ParCompGridARowPtr(compGrid)[i+1] - hypre_ParCompGridARowPtr(compGrid)[i] > 0)
      {
         u_before = hypre_VectorData(hypre_ParCompGridU(compGrid))[i];

         // Initialize u as RHS
         hypre_VectorData(hypre_ParCompGridU(compGrid))[i] = hypre_VectorData(hypre_ParCompGridF(compGrid))[i];
         diag = 0.0;

         // Loop over entries in A
         for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
         {
            #if DEBUG_FAC
            if (hypre_ParCompGridAColInd(compGrid)[j] < 0) printf("Real node doesn't have its full stencil in A! row %d, entry %d\n",i,j);
            #endif
            // If this is the diagonal, store for later division
            if (hypre_ParCompGridAColInd(compGrid)[j] == i) diag = hypre_ParCompGridAData(compGrid)[j];
            // Else, subtract off A_ij*u_j
            else
            {
               hypre_VectorData(hypre_ParCompGridU(compGrid))[i] -= hypre_ParCompGridAData(compGrid)[j] * hypre_VectorData(hypre_ParCompGridU(compGrid))[ hypre_ParCompGridAColInd(compGrid)[j] ];
            }
         }

         // Divide by diagonal
         if (diag == 0.0) printf("Tried to divide by zero diagonal!\n");
         hypre_VectorData(hypre_ParCompGridU(compGrid))[i] /= diag;

         if (hypre_ParCompGridT(compGrid)) hypre_VectorData(hypre_ParCompGridT(compGrid))[i] += hypre_VectorData(hypre_ParCompGridU(compGrid))[i] - u_before;
      }
   }

   return 0;
}

HYPRE_Int 
FAC_Cheby( hypre_ParAMGData *amg_data, hypre_ParCompGrid *compGrid, HYPRE_Int level )
{
   HYPRE_Real    *coefs = hypre_ParAMGDataChebyCoefs(amg_data)[level];
   HYPRE_Int     scale = hypre_ParAMGDataChebyScale(amg_data);
   HYPRE_Int     order = hypre_ParAMGDataChebyOrder(amg_data);

   HYPRE_Int i,j;
   
   HYPRE_Int cheby_order;

   /* u = u + p(A)r */

   if (order > 4)
      order = 4;
   if (order < 1)
      order = 1;

   /* we are using the order of p(A) */
   cheby_order = order -1;
   
   hypre_CSRMatrix *A = hypre_ParCompGridA(compGrid);
   hypre_Vector *u = hypre_ParCompGridU(compGrid);
   hypre_Vector *t = hypre_ParCompGridT(compGrid);
   hypre_Vector *f = hypre_ParCompGridF(compGrid);



   // !!! NOTE: is this correct??? If I'm doing a bunch of matvecs that include the ghost dofs, is that right?
   // I think this is fine for now because I don't store the rows associated with ghost dofs, so their values shouldn't change at all, but this may change in a later version.



   // Calculate diagonal scaling values if necessary
   if (!hypre_ParCompGridL1Norms(compGrid) && scale)
   {
      hypre_ParCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_SHARED);
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
      {
         for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
         {
            if (hypre_ParCompGridAColInd(compGrid)[j] == i) hypre_ParCompGridL1Norms(compGrid)[i] = 1.0/sqrt(hypre_ParCompGridAData(compGrid)[j]);
         }
      }
   }

   // Setup temporary/auxiliary vectors
   if (!hypre_ParCompGridTemp(compGrid))
   {      
      hypre_ParCompGridTemp(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
      hypre_SeqVectorInitialize(hypre_ParCompGridTemp(compGrid));
   }
   hypre_Vector *r = hypre_ParCompGridTemp(compGrid);
   if (!hypre_ParCompGridTemp2(compGrid))
   {      
      hypre_ParCompGridTemp2(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
      hypre_SeqVectorInitialize(hypre_ParCompGridTemp2(compGrid));
   }
   hypre_Vector *u_before = hypre_ParCompGridTemp2(compGrid);
   if (!hypre_ParCompGridTemp3(compGrid))
   {      
      hypre_ParCompGridTemp3(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
      hypre_SeqVectorInitialize(hypre_ParCompGridTemp3(compGrid));
   }
   hypre_Vector *v = hypre_ParCompGridTemp3(compGrid);

   hypre_SeqVectorCopy(u, u_before);

   if (!scale)
   {
      /* get residual: r = f - A*u */
      hypre_SeqVectorCopy(f, r); 
      hypre_CSRMatrixMatvec(-1.0, A, u, 1.0, r);

      hypre_SeqVectorCopy(r, u);
      hypre_SeqVectorScale(coefs[cheby_order], u);

      for (i = cheby_order - 1; i >= 0; i--) 
      {
         hypre_CSRMatrixMatvec(1.0, A, u, 0.0, v);

         hypre_SeqVectorAxpy(coefs[i], r, v);
         hypre_SeqVectorCopy(v, u);
      }
   }
   else /* scaling! */
   {      
    /* get scaled residual: r = D^(-1/2)f -
       * D^(-1/2)A*u */
      hypre_SeqVectorCopy(f, r); 
      hypre_CSRMatrixMatvec(-1.0, A, u, 1.0, r);
      VecComponentwiseScale(hypre_VectorData(r), hypre_ParCompGridL1Norms(compGrid), hypre_ParCompGridNumNodes(compGrid), HYPRE_STREAM(4));

      /* save original u, then start 
         the iteration by multiplying r by the cheby coef.*/
      hypre_SeqVectorCopy(u, u_before);
      hypre_SeqVectorCopy(r, u);
      hypre_SeqVectorScale(coefs[cheby_order], u);

      /* now do the other coefficients */   
      for (i = cheby_order - 1; i >= 0; i--) 
      {
         /* v = D^(-1/2)AD^(-1/2)u */
         VecComponentwiseScale(hypre_VectorData(u), hypre_ParCompGridL1Norms(compGrid), hypre_ParCompGridNumNodes(compGrid), HYPRE_STREAM(4));
         hypre_CSRMatrixMatvec(1.0, A, u, 0.0, v);
         VecComponentwiseScale(hypre_VectorData(v), hypre_ParCompGridL1Norms(compGrid), hypre_ParCompGridNumNodes(compGrid), HYPRE_STREAM(4));

         /* u_new = coef*r + v*/
         hypre_SeqVectorAxpy(coefs[i], r, v);
         hypre_SeqVectorCopy(v, u);         
      } /* end of cheby_order loop */

      /* now we have to scale u_data before adding it to u_orig*/
      VecComponentwiseScale(hypre_VectorData(u), hypre_ParCompGridL1Norms(compGrid), hypre_ParCompGridNumNodes(compGrid), HYPRE_STREAM(4));

   }/* end of scaling code */


   // Update only over real dofs by adjusting size of vectors
   hypre_VectorSize(u) = hypre_ParCompGridNumRealNodes(compGrid);
   hypre_VectorSize(t) = hypre_ParCompGridNumRealNodes(compGrid);
   hypre_VectorSize(u_before) = hypre_ParCompGridNumRealNodes(compGrid);

   hypre_SeqVectorAxpy(1.0, u, t);
   hypre_SeqVectorAxpy(1.0, u_before, u);
  
   hypre_VectorSize(u) = hypre_ParCompGridNumNodes(compGrid);
   hypre_VectorSize(t) = hypre_ParCompGridNumNodes(compGrid);
   hypre_VectorSize(u_before) = hypre_ParCompGridNumNodes(compGrid);

   return hypre_error_flag;
}

