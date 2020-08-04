/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * COGMRESv2 cogmres
 *
 *****************************************************************************/

#include "krylov.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2FunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_COGMRESv2Functions *
hypre_COGMRESv2FunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
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
   HYPRE_Int    (*MassInnerProd) (void *x, void **y, HYPRE_Int k, HYPRE_Int unroll, void *result),
   HYPRE_Int    (*MassDotpTwo)   (void *x, void *y, void **z, HYPRE_Int k, HYPRE_Int unroll, void *result_x, void *result_y),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),      
   HYPRE_Int    (*MassAxpy)      ( HYPRE_Complex *alpha, void **x, void *y, HYPRE_Int k, HYPRE_Int unroll),   
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   )
{
   hypre_COGMRESv2Functions * cogmres_v2_functions;
   cogmres_v2_functions = (hypre_COGMRESv2Functions *)
    CAlloc( 1, sizeof(hypre_COGMRESv2Functions), HYPRE_MEMORY_HOST );

   cogmres_v2_functions->CAlloc            = CAlloc;
   cogmres_v2_functions->Free              = Free;
   cogmres_v2_functions->CommInfo          = CommInfo; /* not in PCGFunctionsCreate */
   cogmres_v2_functions->CreateVector      = CreateVector;
   cogmres_v2_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   cogmres_v2_functions->DestroyVector     = DestroyVector;
   cogmres_v2_functions->MatvecCreate      = MatvecCreate;
   cogmres_v2_functions->Matvec            = Matvec;
   cogmres_v2_functions->MatvecDestroy     = MatvecDestroy;
   cogmres_v2_functions->InnerProd         = InnerProd;
   cogmres_v2_functions->MassInnerProd     = MassInnerProd;
   cogmres_v2_functions->MassDotpTwo       = MassDotpTwo;
   cogmres_v2_functions->CopyVector        = CopyVector;
   cogmres_v2_functions->ClearVector       = ClearVector;
   cogmres_v2_functions->ScaleVector       = ScaleVector;
   cogmres_v2_functions->Axpy              = Axpy;
   cogmres_v2_functions->MassAxpy          = MassAxpy;
   /* default preconditioner must be set here but can be changed later... */
   cogmres_v2_functions->precond_setup     = PrecondSetup;
   cogmres_v2_functions->precond           = Precond;

   return cogmres_v2_functions;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2Create
 *--------------------------------------------------------------------------*/

void *
hypre_COGMRESv2Create( hypre_COGMRESv2Functions *cogmres_v2_functions )
{
   hypre_COGMRESv2Data *cogmres_v2_data;

   cogmres_v2_data = hypre_CTAllocF(hypre_COGMRESv2Data, 1, cogmres_v2_functions, HYPRE_MEMORY_HOST);

   cogmres_v2_data->functions = cogmres_v2_functions;

   /* set defaults */
   (cogmres_v2_data -> k_dim)          = 5;
   (cogmres_v2_data -> cgs)            = 1; /* 0 == Classical Gram Schmidt, 1 == 1Sync, 2 == 2Sync */
   (cogmres_v2_data -> tol)            = 1.0e-06; /* relative residual tol */
   (cogmres_v2_data -> cf_tol)         = 0.0;
   (cogmres_v2_data -> a_tol)          = 0.0; /* abs. residual tol */
   (cogmres_v2_data -> min_iter)       = 0;
   (cogmres_v2_data -> max_iter)       = 1000;
   (cogmres_v2_data -> rel_change)     = 0;
   (cogmres_v2_data -> skip_real_r_check) = 0;
   (cogmres_v2_data -> converged)      = 0;
   (cogmres_v2_data -> precond_data)   = NULL;
   (cogmres_v2_data -> print_level)    = 0;
   (cogmres_v2_data -> logging)        = 0;
   (cogmres_v2_data -> p)              = NULL;
   (cogmres_v2_data -> r)              = NULL;
   (cogmres_v2_data -> w)              = NULL;
   (cogmres_v2_data -> w_2)            = NULL;
   (cogmres_v2_data -> matvec_data)    = NULL;
   (cogmres_v2_data -> norms)          = NULL;
   (cogmres_v2_data -> log_file_name)  = NULL;
   (cogmres_v2_data -> unroll)         = 0;

   return (void *) cogmres_v2_data;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2Destroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2Destroy( void *cogmres_v2_vdata )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   HYPRE_Int i;

   if (cogmres_v2_data)
   {
      hypre_COGMRESv2Functions *cogmres_v2_functions = cogmres_v2_data->functions;
      if ( (cogmres_v2_data->logging>0) || (cogmres_v2_data->print_level) > 0 )
      {
         if ( (cogmres_v2_data -> norms) != NULL )
         hypre_TFreeF( cogmres_v2_data -> norms, cogmres_v2_functions );
      }

      if ( (cogmres_v2_data -> matvec_data) != NULL )
         (*(cogmres_v2_functions->MatvecDestroy))(cogmres_v2_data -> matvec_data);

      if ( (cogmres_v2_data -> r) != NULL )
         (*(cogmres_v2_functions->DestroyVector))(cogmres_v2_data -> r);
      if ( (cogmres_v2_data -> w) != NULL )
         (*(cogmres_v2_functions->DestroyVector))(cogmres_v2_data -> w);
      if ( (cogmres_v2_data -> w_2) != NULL )
         (*(cogmres_v2_functions->DestroyVector))(cogmres_v2_data -> w_2);


      if ( (cogmres_v2_data -> p) != NULL )
      {
         for (i = 0; i < (cogmres_v2_data -> k_dim+1); i++)
         {
            if ( (cogmres_v2_data -> p)[i] != NULL )
            (*(cogmres_v2_functions->DestroyVector))( (cogmres_v2_data -> p) [i]);
         }
         hypre_TFreeF( cogmres_v2_data->p, cogmres_v2_functions );
      }
      hypre_TFreeF( cogmres_v2_data, cogmres_v2_functions );
      hypre_TFreeF( cogmres_v2_functions, cogmres_v2_functions );
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2GetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_COGMRESv2GetResidual( void *cogmres_v2_vdata, void **residual )
{
   /* returns a pointer to the residual vector */

   hypre_COGMRESv2Data  *cogmres_v2_data     = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *residual = cogmres_v2_data->r;
   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2Setup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2Setup( void *cogmres_v2_vdata,
                    void *A,
                    void *b,
                    void *x         )
{
   hypre_COGMRESv2Data *cogmres_v2_data     = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   hypre_COGMRESv2Functions *cogmres_v2_functions = cogmres_v2_data->functions;

   HYPRE_Int k_dim            = (cogmres_v2_data -> k_dim);
   HYPRE_Int max_iter         = (cogmres_v2_data -> max_iter);
   HYPRE_Int (*precond_setup)(void*,void*,void*,void*) = (cogmres_v2_functions->precond_setup);
   void       *precond_data   = (cogmres_v2_data -> precond_data);
   HYPRE_Int rel_change       = (cogmres_v2_data -> rel_change);

   (cogmres_v2_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   if ((cogmres_v2_data -> p) == NULL)
      (cogmres_v2_data -> p) = (void**)(*(cogmres_v2_functions->CreateVectorArray))(k_dim+1,x);
   if ((cogmres_v2_data -> r) == NULL)
      (cogmres_v2_data -> r) = (*(cogmres_v2_functions->CreateVector))(b);

   /* printf("%s %s : pointer = %p\n",__FILE__,__FUNCTION__,hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *)(cogmres_v2_data -> r)))); */
   /* printf("%s %s : size = %d\n",__FILE__,__FUNCTION__,hypre_VectorSize(hypre_ParVectorLocalVector((hypre_ParVector *)(cogmres_v2_data -> r)))); */
   /* printf("%s %s : num vectors = %d\n",__FILE__,__FUNCTION__,hypre_VectorNumVectors(hypre_ParVectorLocalVector((hypre_ParVector *)(cogmres_v2_data -> r)))); */
   /* printf("%s %s : storage = %d\n",__FILE__,__FUNCTION__,hypre_VectorMultiVecStorageMethod(hypre_ParVectorLocalVector((hypre_ParVector *)(cogmres_v2_data -> r)))); */
   /* printf("%s %s : vstride = %d\n",__FILE__,__FUNCTION__,hypre_VectorVectorStride(hypre_ParVectorLocalVector((hypre_ParVector *)(cogmres_v2_data -> r)))); */
   /* printf("%s %s : istride = %d\n",__FILE__,__FUNCTION__,hypre_VectorIndexStride(hypre_ParVectorLocalVector((hypre_ParVector *)(cogmres_v2_data -> r)))); */

   if ((cogmres_v2_data -> w) == NULL)
      (cogmres_v2_data -> w) = (*(cogmres_v2_functions->CreateVector))(b);

   if (rel_change)
   {  
      if ((cogmres_v2_data -> w_2) == NULL)
         (cogmres_v2_data -> w_2) = (*(cogmres_v2_functions->CreateVector))(b);
   }


   if ((cogmres_v2_data -> matvec_data) == NULL)
      (cogmres_v2_data -> matvec_data) = (*(cogmres_v2_functions->MatvecCreate))(A, x);

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ( (cogmres_v2_data->logging)>0 || (cogmres_v2_data->print_level) > 0 )
   {
      if ((cogmres_v2_data -> norms) == NULL)
         (cogmres_v2_data -> norms) = hypre_CTAllocF(HYPRE_Real, max_iter + 1,cogmres_v2_functions, HYPRE_MEMORY_HOST);
   }
   if ( (cogmres_v2_data->print_level) > 0 ) 
   {
      if ((cogmres_v2_data -> log_file_name) == NULL)
         (cogmres_v2_data -> log_file_name) = (char*)"cogmres.out.log";
   }

   return hypre_error_flag;
}


/*-----------------------------------------------------
 * Aux function for Hessenberg matrix storage
 *-----------------------------------------------------*/

HYPRE_Int idx(HYPRE_Int r, HYPRE_Int c, HYPRE_Int n) {
  return r*n+c;
}

void GramSchmidt_Classical(HYPRE_Int my_id,
			   HYPRE_Int i,
                           HYPRE_Int k_dim,
			   HYPRE_Int unroll,
                           void ** p,
                           HYPRE_Real * hh,
                           hypre_COGMRESv2Functions *cf){

  HYPRE_Int j;

  /* These calls can be done with cublas dgemv calls ... need to be configured properly  */
  HYPRE_Int index = (i-1)*(k_dim+1); //idx(i-1, 0, k_dim+1);
  (*(cf->MassInnerProd))((void *) p[i], p, i, unroll, &hh[index]);
  for (j=0; j<i; j++) hh[index+j]  = -hh[index+j];

  (*(cf->MassAxpy))(&hh[index], p, p[i], i, unroll);
  for (j=0; j<i; j++) hh[index+j]  = -hh[index+j];

  HYPRE_Real t = sqrt((*(cf->InnerProd))(p[i], p[i]));
  hh[index + i] = t;
  if (t != 0) {
    t = 1.0/t;
    (*(cf->ScaleVector))(t, p[i]);
  }
}



void GramSchmidt_2SyncTriSolveNoDelayedNorm(HYPRE_Int my_id,
					    HYPRE_Int i,
					    HYPRE_Int k_dim,
					    HYPRE_Int unroll,
					    void ** p,
					    HYPRE_Real * hh,
					    HYPRE_Real * rv,
					    HYPRE_Real * temprv,
					    HYPRE_Real * L,
					    hypre_COGMRESv2Functions *cf){

    HYPRE_Int index = (i-1)*(k_dim+1); //idx(i-1, 0, k_dim+1);
    (*(cf->MassDotpTwo))((void *) p[i-1], p[i], p, i, unroll, temprv, temprv+i);

    //FIRST COLUMN OF rvGPU, i.e., Q(:,1:j-1)'*Q(:,j-1), goes to COLUMN of L  in original code
    memcpy(L+index, temprv, i*sizeof(HYPRE_Real));

    //SECOND COLUMN OF rvGPU, i.e., Q(:,1:j-1)'*Q(:,j), goes to r_i in original code
    memcpy(rv, temprv+i, i*sizeof(HYPRE_Real));

    //triangular solve (I + L)^(-1)
    for (int j=0; j<i; ++j) {
      HYPRE_Int index_rhs = j*(k_dim+1);
      HYPRE_Real RV = rv[j];
      for (int k=0; k<j; ++k) {
        hh[index+j] -= L[index_rhs]*RV;
	index_rhs++;
      }
      hh[index+j] = RV;
    }

    for (int j=0; j<i; ++j) hh[index+j] *= -1.0;
    (*(cf->MassAxpy))(&hh[index], p, p[i], i, unroll);
    for (int j=0; j<i; ++j) hh[index+j] *= -1.0;

    HYPRE_Real t  = sqrt((*(cf->InnerProd))(p[i], p[i]));
    hh[index+i] = t;

    if (t != 0.0) {
      t = 1.0/t;
      (*(cf->ScaleVector))(t, p[i]);
    }
}

void GramSchmidt_1SyncTriSolveDelayedNorm(HYPRE_Int my_id,
					  HYPRE_Int i,
					  HYPRE_Int k_dim,
					  HYPRE_Int unroll,
					  void ** p,
					  HYPRE_Real * hh,
					  HYPRE_Real * rv,
					  HYPRE_Real * temprv,
                                          HYPRE_Real * L,
					  hypre_COGMRESv2Functions *cf){

    HYPRE_Int index = (i-1)*(k_dim+1); //idx(i-1, 0, k_dim+1);
    (*(cf->MassDotpTwo))((void *) p[i-1], p[i], p, i, unroll, temprv, temprv+i);

    HYPRE_Real q_im1_norm_inv = 1.0;
    if (i>=2) {
      q_im1_norm_inv = 1.0/sqrt(temprv[i-1]);
      for (int k=i-1; k<2*i; ++k) temprv[k] *= (q_im1_norm_inv);
      temprv[i-1] *= (q_im1_norm_inv);
      temprv[2*i-1] *= (q_im1_norm_inv);
    }

    //FIRST COLUMN OF rvGPU, i.e., Q(:,1:j-1)'*Q(:,j-1), goes to COLUMN of L  in original code
    memcpy(L+index, temprv, i*sizeof(HYPRE_Real));

    //SECOND COLUMN OF rvGPU, i.e., Q(:,1:j-1)'*Q(:,j), goes to r_i in original code
    memcpy(rv, temprv+i, i*sizeof(HYPRE_Real));

    /* Scale the vectors */
    (*(cf->ScaleVector))(q_im1_norm_inv, p[i-1]);
    (*(cf->ScaleVector))(q_im1_norm_inv, p[i]);

    //triangular solve (I + L)^(-1)
    for (int j=0; j<i; ++j) {
      HYPRE_Int index_rhs = j*(k_dim+1);
      HYPRE_Real RV = rv[j];
      for (int k=0; k<j; ++k) {
        hh[index+j] -= L[index_rhs]*RV;
	index_rhs++;
      }
      hh[index+j] = RV;
    }
    // for (int j=0; j<i; ++j) {
    //   for (int k=0; k<=j; ++k) {
    //     if (j==k) hh[index+j] = rv[j];
    //     else      hh[index+j] -= L[idx(j, k, k_dim+1)]*rv[j];
    //   }
    // }

    for (int j=0; j<i; ++j) hh[index+j] *= -1.0;
    (*(cf->MassAxpy))(&hh[index], p, p[i], i, unroll);
    for (int j=0; j<i; ++j) hh[index+j] *= -1.0;

    /* modify this entry to the householder matrix
       ... similar to what happens at the end of the 2 sync verison */
    if (i>=2) hh[(i-2)*(k_dim+1)+i-1] = 1.0/q_im1_norm_inv;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2Solve
 *-------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2Solve(void  *cogmres_v2_vdata,
                   void  *A,
                   void  *b,
                   void  *x)
{

   hypre_COGMRESv2Data      *cogmres_v2_data      = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   hypre_COGMRESv2Functions *cogmres_v2_functions = cogmres_v2_data->functions;
   HYPRE_Int     k_dim             = (cogmres_v2_data -> k_dim);
   HYPRE_Int     unroll            = (cogmres_v2_data -> unroll);
   HYPRE_Int     cgs               = (cogmres_v2_data -> cgs);
   HYPRE_Int     min_iter          = (cogmres_v2_data -> min_iter);
   HYPRE_Int     max_iter          = (cogmres_v2_data -> max_iter);
   HYPRE_Int     rel_change        = (cogmres_v2_data -> rel_change);
   HYPRE_Int     skip_real_r_check = (cogmres_v2_data -> skip_real_r_check);
   HYPRE_Real    r_tol             = (cogmres_v2_data -> tol);
   HYPRE_Real    cf_tol            = (cogmres_v2_data -> cf_tol);
   HYPRE_Real    a_tol             = (cogmres_v2_data -> a_tol);
   void         *matvec_data       = (cogmres_v2_data -> matvec_data);

   void         *r                 = (cogmres_v2_data -> r);
   void         *w                 = (cogmres_v2_data -> w);
   /* note: w_2 is only allocated if rel_change = 1 */
   void         *w_2               = (cogmres_v2_data -> w_2); 

   void        **p                 = (cogmres_v2_data -> p);

   HYPRE_Int (*precond)(void*,void*,void*,void*) = (cogmres_v2_functions -> precond);
   HYPRE_Int  *precond_data       = (HYPRE_Int*)(cogmres_v2_data -> precond_data);

   HYPRE_Int print_level = (cogmres_v2_data -> print_level);
   HYPRE_Int logging     = (cogmres_v2_data -> logging);

   HYPRE_Real     *norms          = (cogmres_v2_data -> norms);
  /* not used yet   char           *log_file_name  = (cogmres_v2_data -> log_file_name);*/
  /*   FILE           *fp; */

   HYPRE_Int  break_value = 0;
   HYPRE_Int  i, j, k;
  /*KS: rv is the norm history */
   HYPRE_Real *rs, *hh, *uu, *c, *s, *rs_2, *rv;
  //, *tmp; 
   HYPRE_Int  iter; 
   HYPRE_Int  my_id, num_procs;
   HYPRE_Real epsilon, gamma, t, r_norm, b_norm, den_norm, x_norm;
   HYPRE_Real w_norm;

   HYPRE_Real epsmac = 1.e-16; 
   HYPRE_Real ieee_check = 0.;

   HYPRE_Real guard_zero_residual; 
   HYPRE_Real cf_ave_0 = 0.0;
   HYPRE_Real cf_ave_1 = 0.0;
   HYPRE_Real weight;
   HYPRE_Real r_norm_0;
   HYPRE_Real relative_error = 1.0;

   HYPRE_Int        rel_change_passed = 0, num_rel_change_check = 0;
   HYPRE_Int    itmp = 0;

   HYPRE_Real real_r_norm_old, real_r_norm_new;

   printf("%s %s %d : cgs=%d\n",__FILE__,__FUNCTION__,__LINE__,cgs);

   (cogmres_v2_data -> converged) = 0;
   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

   (*(cogmres_v2_functions->CommInfo))(A,&my_id,&num_procs);
   if ( logging>0 || print_level>0 )
   {
      norms          = (cogmres_v2_data -> norms);
   }

   /* initialize work arrays */
   rs = hypre_CTAllocF(HYPRE_Real,k_dim+1,cogmres_v2_functions, HYPRE_MEMORY_HOST);
   c  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_v2_functions, HYPRE_MEMORY_HOST);
   s  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_v2_functions, HYPRE_MEMORY_HOST);
   if (rel_change) rs_2 = hypre_CTAllocF(HYPRE_Real,k_dim+1,cogmres_v2_functions, HYPRE_MEMORY_HOST); 

   rv = hypre_CTAllocF(HYPRE_Real, k_dim+1, cogmres_v2_functions, HYPRE_MEMORY_HOST);

   hh = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_v2_functions, HYPRE_MEMORY_HOST);
   uu = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_v2_functions, HYPRE_MEMORY_HOST);
   HYPRE_Real * temprv = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*2, cogmres_v2_functions, HYPRE_MEMORY_HOST);
   HYPRE_Real * L = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_v2_functions, HYPRE_MEMORY_HOST);

   (*(cogmres_v2_functions->CopyVector))(b,p[0]);

   /* compute initial residual */
   (*(cogmres_v2_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, p[0]);

   b_norm = sqrt((*(cogmres_v2_functions->InnerProd))(b,b));
   real_r_norm_old = b_norm;

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) ieee_check = b_norm/b_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
         hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
         hypre_printf("ERROR -- hypre_COGMRESv2Solve: INFs and/or NaNs detected in input.\n");
         hypre_printf("User probably placed non-numerics in supplied b.\n");
         hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   r_norm   = sqrt((*(cogmres_v2_functions->InnerProd))(p[0],p[0]));
   r_norm_0 = r_norm;

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.) ieee_check = r_norm/r_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
         hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
         hypre_printf("ERROR -- hypre_COGMRESv2Solve: INFs and/or NaNs detected in input.\n");
         hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
         hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }
 
   if ( logging>0 || print_level > 0)
   {
      norms[0] = r_norm;
      if ( print_level>1 && my_id == 0 )
      {
         hypre_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            hypre_printf("Rel_resid_norm actually contains the residual norm\n");
         hypre_printf("Initial L2 norm of residual: %e\n", r_norm);
      }
   }
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
      den_norm = r_norm;
   };

   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
      user specifies a_tol, or sets r_tol = 0.0, which means absolute
      tol only is checked  */

   epsilon = hypre_max(a_tol,r_tol*den_norm);

   /* so now our stop criteria is |r_i| <= epsilon */

   if ( print_level>1 && my_id == 0 )
   {
      if (b_norm > 0.0)
      {
         hypre_printf("=============================================\n\n");
         hypre_printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
         hypre_printf("-----    ------------    ---------- ------------\n");

      }
      else
      {
         hypre_printf("=============================================\n\n");
         hypre_printf("Iters     resid.norm     conv.rate\n");
         hypre_printf("-----    ------------    ----------\n");
      };
   }


   /* once the rel. change check has passed, we do not want to check it again */
   rel_change_passed = 0;

   while (iter < max_iter)
   {
      /* initialize first term of hessenberg system */
      if (r_norm == 0.0)
      {
         hypre_TFreeF(c,cogmres_v2_functions); 
         hypre_TFreeF(s,cogmres_v2_functions); 
         hypre_TFreeF(rs,cogmres_v2_functions);
         hypre_TFreeF(rv,cogmres_v2_functions);
         if (rel_change)  hypre_TFreeF(rs_2,cogmres_v2_functions);
         hypre_TFreeF(hh,cogmres_v2_functions); 
         hypre_TFreeF(uu,cogmres_v2_functions); 
         return hypre_error_flag;
      }

      /* see if we are already converged and 
         should print the final norm and exit */

      if (r_norm  <= epsilon && iter >= min_iter) 
      {
         if (!rel_change) /* shouldn't exit after no iterations if
                           * relative change is on*/
         {
            (*(cogmres_v2_functions->CopyVector))(b,r);
            (*(cogmres_v2_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
            r_norm = sqrt((*(cogmres_v2_functions->InnerProd))(r,r));
            if (r_norm  <= epsilon)
            {
               if ( print_level>1 && my_id == 0)
               {
                  hypre_printf("\n\n");
                  hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               break;
            }
            else if ( print_level>0 && my_id == 0)
               hypre_printf("false convergence 1\n");
         }
      }



      t = 1.0 / r_norm;
      (*(cogmres_v2_functions->ScaleVector))(t,p[0]);
      rs[0] = r_norm;
      rv[0] = 1.0;
      //memset(temprv,0,2*i*sizeof(HYPRE_Real));
      temprv[0] = rv[0];
      L[0] = 1.0;
      i = 0;
      /***RESTART CYCLE (right-preconditioning) ***/
      while (i < k_dim && iter < max_iter)
      {
         i++;
         iter++;
         itmp = (i-1)*(k_dim+1);

         (*(cogmres_v2_functions->ClearVector))(r);

         precond(precond_data, A, p[i-1], r);

         (*(cogmres_v2_functions->Matvec))(matvec_data, 1.0, A, r, 0.0, p[i]);

	 if (cgs==0) {
	   /* Classical GramSchmidt */
	   GramSchmidt_Classical(my_id, i, k_dim, unroll, p, hh, cogmres_v2_functions);
	 } else if (cgs==1) {
	   /* 1 Sync version, triangular solve, with delayed norm*/
	   GramSchmidt_1SyncTriSolveDelayedNorm(my_id, i, k_dim, unroll, p, hh, rv, temprv, L, cogmres_v2_functions);
	   if (i==1) {
	     iter--;
	     continue;
	   }
           /* we need to look back at the previous version of the house holder matrix, this change in the indexing needs to be undone later on. */
	   i--;
	   itmp = (i-1)*(k_dim+1);
	 } else if (cgs==2) {
	   /* 2 Sync version, triangular solve, no delayed norm*/
	   GramSchmidt_2SyncTriSolveNoDelayedNorm(my_id, i, k_dim, unroll, p, hh, rv, temprv, L, cogmres_v2_functions);
	 }

	 /* Same as cogmres from here down */
         for (j = 1; j < i; j++)
         {
            t = hh[itmp+j-1];
            hh[itmp+j-1] = s[j-1]*hh[itmp+j] + c[j-1]*t;
            hh[itmp+j] = -s[j-1]*t + c[j-1]*hh[itmp+j];
         }
         t= hh[itmp+i]*hh[itmp+i];
         t+= hh[itmp+i-1]*hh[itmp+i-1];
         gamma = sqrt(t);
         if (gamma == 0.0) gamma = epsmac;
         c[i-1] = hh[itmp+i-1]/gamma;
         s[i-1] = hh[itmp+i]/gamma;
         rs[i] = -hh[itmp+i]*rs[i-1];
         rs[i] /=  gamma;
         rs[i-1] = c[i-1]*rs[i-1];
         // determine residual norm 
         hh[itmp+i-1] = s[i-1]*hh[itmp+i] + c[i-1]*hh[itmp+i-1];
         r_norm = fabs(rs[i]);

         if ( print_level>0 )
         {
            norms[iter] = r_norm;
            if ( print_level>1 && my_id == 0 )
            {
               if (b_norm > 0.0)
                  hypre_printf("% 5d    %e    %f   %e\n", iter, 
                     norms[iter],norms[iter]/norms[iter-1],
                     norms[iter]/b_norm);
               else
                  hypre_printf("% 5d    %e    %f\n", iter, norms[iter],
                     norms[iter]/norms[iter-1]);
            }
         }
         /*convergence factor tolerance */
         if (cf_tol > 0.0)
         {
            cf_ave_0 = cf_ave_1;
            cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));

            weight = fabs(cf_ave_1 - cf_ave_0);
            weight = weight / hypre_max(cf_ave_1, cf_ave_0);

            weight = 1.0 - weight;
#if 0
           hypre_printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
              i, cf_ave_1, cf_ave_0, weight );
#endif
            if (weight * cf_ave_1 > cf_tol) 
            {
               break_value = 1;
               break;
            }
         }
         /* should we exit the restart cycle? (conv. check) */
         if (r_norm <= epsilon && iter >= min_iter)
         {
            if (rel_change && !rel_change_passed)
            {
               /* To decide whether to break here: to actually
                  determine the relative change requires the approx
                  solution (so a triangular solve) and a
                  precond. solve - so if we have to do this many
                  times, it will be expensive...(unlike cg where is
                  is relatively straightforward)
                  previously, the intent (there was a bug), was to
                  exit the restart cycle based on the residual norm
                  and check the relative change outside the cycle.
                  Here we will check the relative here as we don't
                  want to exit the restart cycle prematurely */
               for (k=0; k<i; k++) /* extra copy of rs so we don't need
                                   to change the later solve */
                  rs_2[k] = rs[k];

               /* solve tri. system*/
               rs_2[i-1] = rs_2[i-1]/hh[itmp+i-1];
               for (k = i-2; k >= 0; k--)
               {
                  t = 0.0;
                  for (j = k+1; j < i; j++)
                  {
                     t -= hh[j*(k_dim+1)+k]*rs_2[j];
                  }
                  t+= rs_2[k];
                  rs_2[k] = t/hh[k*(k_dim+1)+k];
               }
               (*(cogmres_v2_functions->CopyVector))(p[i-1],w);
               (*(cogmres_v2_functions->ScaleVector))(rs_2[i-1],w);
               for (j = i-2; j >=0; j--)
                  (*(cogmres_v2_functions->Axpy))(rs_2[j], p[j], w);

               (*(cogmres_v2_functions->ClearVector))(r);
               /* find correction (in r) */
               precond(precond_data, A, w, r);
               /* copy current solution (x) to w (don't want to over-write x)*/
               (*(cogmres_v2_functions->CopyVector))(x,w);

               /* add the correction */
               (*(cogmres_v2_functions->Axpy))(1.0,r,w);

               /* now w is the approx solution  - get the norm*/
               x_norm = sqrt( (*(cogmres_v2_functions->InnerProd))(w,w) );

               if ( !(x_norm <= guard_zero_residual ))
                  /* don't divide by zero */
               {  /* now get  x_i - x_i-1 */
                  if (num_rel_change_check)
                  {
                     /* have already checked once so we can avoid another precond.
                        solve */
                     (*(cogmres_v2_functions->CopyVector))(w, r);
                     (*(cogmres_v2_functions->Axpy))(-1.0, w_2, r);
                     /* now r contains x_i - x_i-1*/

                     /* save current soln w in w_2 for next time */
                     (*(cogmres_v2_functions->CopyVector))(w, w_2);
                  }
                  else
                  {
                     /* first time to check rel change*/
                     /* first save current soln w in w_2 for next time */
                     (*(cogmres_v2_functions->CopyVector))(w, w_2);

                     (*(cogmres_v2_functions->ClearVector))(w);
                     (*(cogmres_v2_functions->Axpy))(rs_2[i-1], p[i-1], w);
                     (*(cogmres_v2_functions->ClearVector))(r);
                     /* apply the preconditioner */
                     precond(precond_data, A, w, r);
                     /* now r contains x_i - x_i-1 */          
                  }
                  /* find the norm of x_i - x_i-1 */          
                  w_norm = sqrt( (*(cogmres_v2_functions->InnerProd))(r,r) );
                  relative_error = w_norm/x_norm;
                  if (relative_error <= r_tol)
                  {
                     rel_change_passed = 1;
                     break;
                  }
               }
               else
               {
                  rel_change_passed = 1;
                  break;
               }
               num_rel_change_check++;
            }
            else /* no relative change */
            {
               break;
            }
         }
	 if (cgs==1) {
           /* This code works on the delayed norm ... so we need to push the indices forward */
           i++;
	   itmp = (i-1)*(k_dim+1);
	 }
      } /*** end of restart cycle ***/

      /* now compute solution, first solve upper triangular system */
      if (break_value) break;
     
      rs[i-1] = rs[i-1]/hh[itmp+i-1];
      for (k = i-2; k >= 0; k--)
      {
         t = 0.0;
         for (j = k+1; j < i; j++)
         {
            t -= hh[j*(k_dim+1)+k]*rs[j];
         }
         t+= rs[k];
         rs[k] = t/hh[k*(k_dim+1)+k];
      }

      (*(cogmres_v2_functions->CopyVector))(p[i-1],w);
      (*(cogmres_v2_functions->ScaleVector))(rs[i-1],w);
      for (j = i-2; j >=0; j--)
         (*(cogmres_v2_functions->Axpy))(rs[j], p[j], w);

      (*(cogmres_v2_functions->ClearVector))(r);
      /* find correction (in r) */
      precond(precond_data, A, w, r);

      /* update current solution x (in x) */
      (*(cogmres_v2_functions->Axpy))(1.0,r,x);


      /* check for convergence by evaluating the actual residual */
      if (r_norm  <= epsilon && iter >= min_iter)
      {
         if (skip_real_r_check)
         {
            (cogmres_v2_data -> converged) = 1;
            break;
         }

         /* calculate actual residual norm*/
         (*(cogmres_v2_functions->CopyVector))(b,r);
         (*(cogmres_v2_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
         real_r_norm_new = r_norm = sqrt( (*(cogmres_v2_functions->InnerProd))(r,r) );

         if (r_norm <= epsilon)
         {
            if (rel_change && !rel_change_passed) /* calculate the relative change */
            {
               /* calculate the norm of the solution */
               x_norm = sqrt( (*(cogmres_v2_functions->InnerProd))(x,x) );

               if ( !(x_norm <= guard_zero_residual ))
               /* don't divide by zero */
               {
                  (*(cogmres_v2_functions->ClearVector))(w);
                  (*(cogmres_v2_functions->Axpy))(rs[i-1], p[i-1], w);
                  (*(cogmres_v2_functions->ClearVector))(r);
                  /* apply the preconditioner */
                  precond(precond_data, A, w, r);
                  /* find the norm of x_i - x_i-1 */          
                  w_norm = sqrt( (*(cogmres_v2_functions->InnerProd))(r,r) );
                  relative_error= w_norm/x_norm;
                  if ( relative_error < r_tol )
                  {
                     (cogmres_v2_data -> converged) = 1;
                     if ( print_level>1 && my_id == 0 )
                     {
                        hypre_printf("\n\n");
                        hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                     }
                     break;
                  }
               }
               else
               {
                  (cogmres_v2_data -> converged) = 1;
                  if ( print_level>1 && my_id == 0 )
                  {
                     hypre_printf("\n\n");
                     hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                  }
                  break;
               }
            }
            else /* don't need to check rel. change */
            {
               if ( print_level>1 && my_id == 0 )
               {
                  hypre_printf("\n\n");
                  hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               (cogmres_v2_data -> converged) = 1;
               break;
            }
         }
         else /* conv. has not occurred, according to true residual */
         {
            /* exit if the real residual norm has not decreased */
            if (real_r_norm_new >= real_r_norm_old)
            {
               if (print_level > 1 && my_id == 0)
               {
                  hypre_printf("\n\n");
                  hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               (cogmres_v2_data -> converged) = 1;
               break;
            }
            /* report discrepancy between real/COGMRESv2 residuals and restart */
            if ( print_level>0 && my_id == 0)
               hypre_printf("false convergence 2, L2 norm of residual: %e\n", r_norm);
            (*(cogmres_v2_functions->CopyVector))(r,p[0]);
            i = 0;
            real_r_norm_old = real_r_norm_new;
         }
      } /* end of convergence check */

      /* compute residual vector and continue loop */
      for (j=i ; j > 0; j--)
      {
         rs[j-1] = -s[j-1]*rs[j];
         rs[j] = c[j-1]*rs[j];
      }

      if (i) (*(cogmres_v2_functions->Axpy))(rs[i]-1.0,p[i],p[i]);
      for (j=i-1 ; j > 0; j--)
         (*(cogmres_v2_functions->Axpy))(rs[j],p[j],p[i]);

      if (i)
      {
         (*(cogmres_v2_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
         (*(cogmres_v2_functions->Axpy))(1.0,p[i],p[0]);
      }

   } /* END of iteration while loop */


   (cogmres_v2_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (cogmres_v2_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (cogmres_v2_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon && epsilon > 0) hypre_error(HYPRE_ERROR_CONV);

   hypre_TFreeF(c,cogmres_v2_functions); 
   hypre_TFreeF(s,cogmres_v2_functions); 
   hypre_TFreeF(rs,cogmres_v2_functions);
   hypre_TFreeF(rv,cogmres_v2_functions);
   if (rel_change)  hypre_TFreeF(rs_2,cogmres_v2_functions);

   /*for (i=0; i < k_dim+1; i++)
   {  
      hypre_TFreeF(hh[i],cogmres_v2_functions);
      hypre_TFreeF(uu[i],cogmres_v2_functions);
   }*/
   hypre_TFreeF(hh,cogmres_v2_functions); 
   hypre_TFreeF(uu,cogmres_v2_functions);

   hypre_TFreeF(temprv,cogmres_v2_functions);
   hypre_TFreeF(L,cogmres_v2_functions);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetKDim, hypre_COGMRESv2GetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetKDim( void   *cogmres_v2_vdata,
        HYPRE_Int   k_dim )
{
   hypre_COGMRESv2Data *cogmres_v2_data =(hypre_COGMRESv2Data *) cogmres_v2_vdata;
   (cogmres_v2_data -> k_dim) = k_dim;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetKDim( void   *cogmres_v2_vdata,
        HYPRE_Int * k_dim )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *k_dim = (cogmres_v2_data -> k_dim);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetUnroll, hypre_COGMRESv2GetUnroll
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetUnroll( void   *cogmres_v2_vdata,
        HYPRE_Int   unroll )
{
   hypre_COGMRESv2Data *cogmres_v2_data =(hypre_COGMRESv2Data *) cogmres_v2_vdata;
   (cogmres_v2_data -> unroll) = unroll;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetUnroll( void   *cogmres_v2_vdata,
        HYPRE_Int * unroll )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *unroll = (cogmres_v2_data -> unroll);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetCGS, hypre_COGMRESv2GetCGS
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetCGS( void   *cogmres_v2_vdata,
        HYPRE_Int   cgs )
{
   hypre_COGMRESv2Data *cogmres_v2_data =(hypre_COGMRESv2Data *) cogmres_v2_vdata;
   (cogmres_v2_data -> cgs) = cgs;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetCGS( void   *cogmres_v2_vdata,
        HYPRE_Int * cgs )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *cgs = (cogmres_v2_data -> cgs);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetTol, hypre_COGMRESv2GetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetTol( void   *cogmres_v2_vdata,
        HYPRE_Real  tol       )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   (cogmres_v2_data -> tol) = tol;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetTol( void   *cogmres_v2_vdata,
        HYPRE_Real  * tol      )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *tol = (cogmres_v2_data -> tol);
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetAbsoluteTol, hypre_COGMRESv2GetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetAbsoluteTol( void   *cogmres_v2_vdata,
        HYPRE_Real  a_tol       )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   (cogmres_v2_data -> a_tol) = a_tol;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetAbsoluteTol( void   *cogmres_v2_vdata,
        HYPRE_Real  * a_tol      )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *a_tol = (cogmres_v2_data -> a_tol);
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetConvergenceFactorTol, hypre_COGMRESv2GetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetConvergenceFactorTol( void   *cogmres_v2_vdata,
        HYPRE_Real  cf_tol       )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   (cogmres_v2_data -> cf_tol) = cf_tol;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetConvergenceFactorTol( void   *cogmres_v2_vdata,
        HYPRE_Real * cf_tol       )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *cf_tol = (cogmres_v2_data -> cf_tol);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetMinIter, hypre_COGMRESv2GetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetMinIter( void *cogmres_v2_vdata,
        HYPRE_Int   min_iter  )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   (cogmres_v2_data -> min_iter) = min_iter;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetMinIter( void *cogmres_v2_vdata,
        HYPRE_Int * min_iter  )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *min_iter = (cogmres_v2_data -> min_iter);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetMaxIter, hypre_COGMRESv2GetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetMaxIter( void *cogmres_v2_vdata,
        HYPRE_Int   max_iter  )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   (cogmres_v2_data -> max_iter) = max_iter;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetMaxIter( void *cogmres_v2_vdata,
        HYPRE_Int * max_iter  )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *max_iter = (cogmres_v2_data -> max_iter);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetRelChange, hypre_COGMRESv2GetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetRelChange( void *cogmres_v2_vdata,
        HYPRE_Int   rel_change  )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   (cogmres_v2_data -> rel_change) = rel_change;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetRelChange( void *cogmres_v2_vdata,
        HYPRE_Int * rel_change  )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *rel_change = (cogmres_v2_data -> rel_change);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetSkipRealResidualCheck, hypre_COGMRESv2GetSkipRealResidualCheck
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetSkipRealResidualCheck( void *cogmres_v2_vdata,
        HYPRE_Int skip_real_r_check )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   (cogmres_v2_data -> skip_real_r_check) = skip_real_r_check;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetSkipRealResidualCheck( void *cogmres_v2_vdata,
        HYPRE_Int *skip_real_r_check)
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *skip_real_r_check = (cogmres_v2_data -> skip_real_r_check);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetPrecond( void  *cogmres_v2_vdata,
        HYPRE_Int  (*precond)(void*,void*,void*,void*),
        HYPRE_Int  (*precond_setup)(void*,void*,void*,void*),
        void  *precond_data )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   hypre_COGMRESv2Functions *cogmres_v2_functions = cogmres_v2_data->functions;
   (cogmres_v2_functions -> precond)        = precond;
   (cogmres_v2_functions -> precond_setup)  = precond_setup;
   (cogmres_v2_data -> precond_data)   = precond_data;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2GetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2GetPrecond( void         *cogmres_v2_vdata,
        HYPRE_Solver *precond_data_ptr )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *precond_data_ptr = (HYPRE_Solver)(cogmres_v2_data -> precond_data);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetPrintLevel, hypre_COGMRESv2GetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetPrintLevel( void *cogmres_v2_vdata,
        HYPRE_Int   level)
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   (cogmres_v2_data -> print_level) = level;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetPrintLevel( void *cogmres_v2_vdata,
        HYPRE_Int * level)
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *level = (cogmres_v2_data -> print_level);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2SetLogging, hypre_COGMRESv2GetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2SetLogging( void *cogmres_v2_vdata,
        HYPRE_Int   level)
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   (cogmres_v2_data -> logging) = level;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESv2GetLogging( void *cogmres_v2_vdata,
        HYPRE_Int * level)
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *level = (cogmres_v2_data -> logging);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2GetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2GetNumIterations( void *cogmres_v2_vdata,
        HYPRE_Int  *num_iterations )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *num_iterations = (cogmres_v2_data -> num_iterations);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2GetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2GetConverged( void *cogmres_v2_vdata,
        HYPRE_Int  *converged )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *converged = (cogmres_v2_data -> converged);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESv2GetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESv2GetFinalRelativeResidualNorm( void   *cogmres_v2_vdata,
        HYPRE_Real *relative_residual_norm )
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   *relative_residual_norm = (cogmres_v2_data -> rel_residual_norm);
   return hypre_error_flag;
}


HYPRE_Int 
hypre_COGMRESv2SetModifyPC(void *cogmres_v2_vdata, 
      HYPRE_Int (*modify_pc)(void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm))
{
   hypre_COGMRESv2Data *cogmres_v2_data = (hypre_COGMRESv2Data *)cogmres_v2_vdata;
   hypre_COGMRESv2Functions *cogmres_v2_functions = cogmres_v2_data->functions;
   (cogmres_v2_functions -> modify_pc)        = modify_pc;
   return hypre_error_flag;
} 

