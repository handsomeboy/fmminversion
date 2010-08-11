#include "problem.h"

//Implementation of class Problem
Problem::Problem(void)
{	
}

Problem::Problem(int m,int n,string type)
{
	this->SetSizes(m,n,type);
}

Problem::~Problem(void)
{
} 

//Initialize everything
PetscErrorCode Problem::SetSizes(int _m, int _n, string type)
{
	PetscErrorCode ierr;
	m = _m;		n = _n;
	
	if (type == "Harmonic")
		p = 1;
	else if (type == "Biharmonic")
		p = 4;
	else
	{
		cout<<"Not a recognized model"<<endl;
		return PETSC_NULL;	
	}		

	//Resulting Matrix::A
	ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n+p,n+p,\
			this,&A);					CHKERRQ(ierr);

	//RHS and unknown
	ierr = VecCreate(PETSC_COMM_WORLD,&x);				CHKERRQ(ierr);
	ierr = VecSetSizes(x,PETSC_DECIDE,n+p);				CHKERRQ(ierr);
	ierr = VecSetFromOptions(x);					CHKERRQ(ierr);
	ierr = VecCreate(PETSC_COMM_WORLD,&b);				CHKERRQ(ierr);
	ierr = VecSetSizes(b,PETSC_DECIDE,n+p);				CHKERRQ(ierr);
	ierr = VecSetFromOptions(b);					CHKERRQ(ierr);

	//Create Shells
	ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,m,\
			this,&Q);					CHKERRQ(ierr);
	ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,\
			this,&R);					CHKERRQ(ierr);
	ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,p,\
			this,&HX);					CHKERRQ(ierr);
	
	//Regular internal matrices
	ierr = MatCreate(PETSC_COMM_WORLD,&H);				CHKERRQ(ierr);
	ierr = MatSetSizes(H,PETSC_DECIDE,PETSC_DECIDE,m,n);		CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD,&X);				CHKERRQ(ierr);
	ierr = MatSetSizes(H,PETSC_DECIDE,PETSC_DECIDE,m,p);		CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD,&y);				CHKERRQ(ierr);
	ierr = VecSetSizes(y,PETSC_DECIDE,n);				CHKERRQ(ierr);
	ierr = VecSetFromOptions(b);					CHKERRQ(ierr);

	//Set operators to the shells
	ierr = MatShellSetOperation(R,MATOP_MULT,\
		(void(*)(void))&RMult);					CHKERRQ(ierr);
	ierr = MatShellSetOperation(HX,MATOP_MULT,\
		(void(*)(void))&HXMult);				CHKERRQ(ierr);
	ierr = MatShellSetOperation(HX,MATOP_MULT_TRANSPOSE,\
			(void(*)(void))&HXMultTranspose);		CHKERRQ(ierr);

	//Set KSP operators 
	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);			CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);	CHKERRQ(ierr);
	ierr = KSPSetFromOptions(ksp);					CHKERRQ(ierr);

	//No Preconditioner yet

	return ierr;
} 

//Implementations of bbFMMWrapper
bbFMMWrapper::bbFMMWrapper(int _n)
{
	n = _n;
	kernel = new KLaplacian(n);

	param = new FMMParam(Nf,Ns,boxcntr,L,n);
    	tree = new FMMNode(boxcntr,param->L);


	//Set pointers to null
	U = NULL;
	V = NULL;
	K = NULL;
}

bbFMMWrapper::~bbFMMWrapper(void)
{
	FMMCleanup(param,kernel,tree,U,V,K);

	delete [] observpos;
	delete [] sourcepos;
}

void bbFMMWrapper::SetupRandom(int _Nf,int _Ns)
{
	Vector3 boxcntr(0.0,0.0,0.0);
	
	Nf = _Nf;	Ns = _Ns;

	for (int j=0; j<Ns; j++)
        	sourcepos[j] = boxcntr + 
            		Vector3(frand(-0.5,0.5)*L.x,frand(-0.5,0.5)*L.y,
                    		frand(-0.5,0.5)*L.z);
	for (int i=0; i<Nf; i++)
	        observpos[i] = sourcepos[i];
	//Complete the setup
	this->Setup();

	return;
}

void bbFMMWrapper::Setup(void)
{
	param = new FMMParam(Nf,Ns,boxcntr,L,n);
 	tree = new FMMNode(boxcntr,param->L);

	// Set up for bbFMM calculation
    	FMMSetup(param,kernel,&tree);
    
    	// Load the pre-computed SVD matrices
	LoadMatrices(&param,kernel,&U,&V,&K);

	delete kernel;
	return;
}

void bbFMMWrapper::Compute(double *qf, double *qs)
{
	bbFMM(param,kernel,&tree,U,V,K,observpos,sourcepos,qf,qs,field);
	return;
}

//Matrix Shell Operators
PetscErrorCode RMult(Mat A, Vec x, Vec y)
{
	PetscErrorCode ierr;
	
	void *ctx;	
	ierr = MatShellGetContext(A,&ctx);				CHKERRQ(ierr);
	Problem* problem = (Problem*)ctx;

	ierr = VecScale(x,problem->theta2);				CHKERRQ(ierr);
 	ierr = VecDuplicate(x,&y);					CHKERRQ(ierr);

	ctx = NULL;		
	problem = NULL;
	return ierr;
}
PetscErrorCode PsiMult(Mat A, Vec x, Vec y)
{
	PetscErrorCode ierr;

	void *ctx;	
	ierr = MatShellGetContext(A,&ctx);				CHKERRQ(ierr);
	Problem* problem = (Problem*)ctx;

	//Create temporary vec
	Vec temp;
	ierr = VecCreate(PETSC_COMM_WORLD,&temp);			CHKERRQ(ierr);
	ierr = VecDuplicate(x,&temp);					CHKERRQ(ierr);

	//HQH^T * x
	ierr = MatMultTranspose(problem->H,x,y);			CHKERRQ(ierr);
	ierr = MatMult(problem->Q,y,temp);				CHKERRQ(ierr);
	ierr = MatMult(problem->H,temp,y);				CHKERRQ(ierr);

	// + R * x part
	ierr = VecAXPY(y,problem->theta2,x);				CHKERRQ(ierr);
	
	ctx = NULL;
	problem = NULL;
	ierr = VecDestroy(temp);					CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode HXMult(Mat A, Vec x, Vec y)
{
	PetscErrorCode ierr;

	void *ctx;	
	ierr = MatShellGetContext(A,&ctx);				CHKERRQ(ierr);
	Problem* problem = (Problem*)ctx;

	//Create temporary vec
	Vec temp;
	ierr = VecCreate(PETSC_COMM_WORLD,&temp);			CHKERRQ(ierr);
	ierr = VecSetSizes(temp,PETSC_DECIDE,problem->m);		CHKERRQ(ierr);
	ierr = VecSetFromOptions(temp);					CHKERRQ(ierr);

	// X * x
	ierr = MatMult(problem->X,x,temp);				CHKERRQ(ierr);
	//H * (X * x)
	ierr = MatMult(problem->H,temp,y);				CHKERRQ(ierr);	

	ctx = NULL;
	problem = NULL;
	ierr = VecDestroy(temp);					CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode HXMultAdd(Mat A, Vec x, Vec y, Vec w)
{
	PetscErrorCode ierr;

	void *ctx;	
	ierr = MatShellGetContext(A,&ctx);				CHKERRQ(ierr);
	Problem* problem = (Problem*)ctx;

	//Create temporary vec
	Vec temp;
	ierr = VecCreate(PETSC_COMM_WORLD,&temp);			CHKERRQ(ierr);
	ierr = VecSetSizes(temp,PETSC_DECIDE,problem->n);		CHKERRQ(ierr);
	ierr = VecSetFromOptions(temp);					CHKERRQ(ierr);

	PetscScalar alpha = 1.0;
	ierr = MatMult(problem->HX,x,temp);				CHKERRQ(ierr);
	ierr = VecAXPY(temp,alpha,y);					CHKERRQ(ierr);	
	ctx = NULL;
	problem = NULL;
	ierr = VecDestroy(temp);					CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode HXMultTranspose(Mat A, Vec x, Vec y)
{
	PetscErrorCode ierr;

	void *ctx;	
	ierr = MatShellGetContext(A,&ctx);				CHKERRQ(ierr);
	Problem* problem = (Problem*)ctx;

	//Create temporary vec
	Vec temp;
	ierr = VecCreate(PETSC_COMM_WORLD,&temp);			CHKERRQ(ierr);
	ierr = VecSetSizes(temp,PETSC_DECIDE,problem->m);		CHKERRQ(ierr);
	ierr = VecSetFromOptions(temp);					CHKERRQ(ierr);

	// H^T * x
	ierr = MatMultTranspose(problem->H,x,temp);			CHKERRQ(ierr);
	//X^T * (H^T * x)
	ierr = MatMultTranspose(problem->X,temp,y);			CHKERRQ(ierr);	

	ctx = NULL;
	problem = NULL;
	ierr = VecDestroy(temp);					CHKERRQ(ierr);
	return ierr;
}


PetscErrorCode AMult(Mat A,Vec x,Vec y)
{
	PetscErrorCode ierr;
	
	void *ctx;	
	ierr = MatShellGetContext(A,&ctx);				CHKERRQ(ierr);
	Problem* problem = (Problem*)ctx;
	
	int *colPsi, *colHXT;
	double *valPsi, *valHXT;
	
	//Create Temporary vectors
	Vec temp,temp1,temp2;
	ierr = VecCreate(PETSC_COMM_WORLD,&temp);			CHKERRQ(ierr);
	ierr = VecSetSizes(temp,PETSC_DECIDE,problem->m);		CHKERRQ(ierr);
	ierr = VecSetFromOptions(temp);					CHKERRQ(ierr);

	ierr = VecDuplicate(temp,&temp1);				CHKERRQ(ierr);	

	ierr = VecCreate(PETSC_COMM_WORLD,&temp2);			CHKERRQ(ierr);
	ierr = VecSetSizes(temp2,PETSC_DECIDE,problem->m);		CHKERRQ(ierr);
	ierr = VecSetFromOptions(temp2);				CHKERRQ(ierr);

	colPsi = new int[problem->n];
	colHXT = new int[problem->p];
	valPsi = new double[problem->n];
	valHXT = new double[problem->p];
	
	//Indices to get information with
	for(int i = 0; i < problem->n; i++)		colPsi[i] = i;
	for(int i = 0; i < problem->p; i++)		colHXT[i] = problem->n + i;

	//Get values 
	ierr = VecGetValues(x,problem->n,colPsi,valPsi);		CHKERRQ(ierr);	
	ierr = VecGetValues(x,problem->n,colHXT,valHXT);		CHKERRQ(ierr);	
	
	//And Assemble data
	ierr = VecSetValues(temp1,problem->n,colPsi,valPsi,\
				INSERT_VALUES);				CHKERRQ(ierr);
	ierr = VecAssemblyBegin(temp1);					CHKERRQ(ierr);
	ierr = VecAssemblyEnd(temp1);					CHKERRQ(ierr);

	ierr = VecSetValues(temp2,problem->n,colPsi,valPsi,\
				INSERT_VALUES);				CHKERRQ(ierr);
	ierr = VecAssemblyBegin(temp2);					CHKERRQ(ierr);
	ierr = VecAssemblyEnd(temp2);					CHKERRQ(ierr);
	
	//Perform actual multiplication
	ierr = MatMult(problem->Psi,temp1,temp);			CHKERRQ(ierr);
	ierr = MatMultAdd(problem->HX,temp2,temp,temp);			CHKERRQ(ierr);
	ierr = MatMultTranspose(problem->HX,temp,temp2);		CHKERRQ(ierr);

	//Get values
	ierr = VecGetValues(temp,problem->n,colPsi,valPsi);		CHKERRQ(ierr);	
	ierr = VecGetValues(temp2,problem->p,colPsi,valHXT);		CHKERRQ(ierr);	
	//Assemble final answer
	ierr = VecSetValues(y,problem->n,colPsi,valPsi,INSERT_VALUES);	CHKERRQ(ierr);
	ierr = VecSetValues(y,problem->n,colHXT,valHXT,INSERT_VALUES);	CHKERRQ(ierr);
	ierr = VecAssemblyBegin(y);					CHKERRQ(ierr);
	ierr = VecAssemblyEnd(y);					CHKERRQ(ierr);

	delete [] valPsi;
	delete [] valHXT;
	delete [] colPsi;
	delete [] colHXT;

	ierr = VecDestroy(temp);					CHKERRQ(ierr);
	ierr = VecDestroy(temp1);					CHKERRQ(ierr);
	ierr = VecDestroy(temp2);					CHKERRQ(ierr);

	ctx = NULL;
	problem = NULL;
	
	return ierr;
}
PetscErrorCode QMult(Mat A, Vec x,Vec y)
{
	PetscErrorCode ierr;
	
	void *ctx;	
	ierr = MatShellGetContext(A,&ctx);				CHKERRQ(ierr);
	Problem* problem = (Problem*) ctx;

	double *xptr, *yptr;

	ierr = VecGetArray(x,&xptr);					CHKERRQ(ierr);
	ierr = VecGetArray(y,&yptr);					CHKERRQ(ierr);

#ifdef bbFMM
	problem->bbFMMWrapper->Compute(yptr,xptr);
#endif
	ierr = VecRestoreArray(x,&xptr);				CHKERRQ(ierr);
	ierr = VecRestoreArray(y,&yptr);				CHKERRQ(ierr);

	xptr = NULL;
	yptr = NULL;
	ctx = NULL;
	problem = NULL;
	return ierr;
}

PetscErrorCode AMultProper(Mat A, Vec x, Vec y)
{
	PetscErrorCode ierr;
	
	void *ctx;	
	ierr = MatShellGetContext(A,&ctx);				CHKERRQ(ierr);
	Problem* problem = (Problem*)ctx;

	double *xpt, *ypt;
	ierr = VecGetArray(x,&xpt);					CHKERRQ(ierr);
	ierr = VecGetArray(y,&ypt);					CHKERRQ(ierr);

	//Create Temporary vectors
	Vec v1, v2, v3;
	ierr = VecCreate(PETSC_COMM_WORLD,&v1);				CHKERRQ(ierr);
	ierr = VecSetSizes(v1,PETSC_DECIDE,problem->n);			CHKERRQ(ierr);
	ierr = VecSetFromOptions(v1);					CHKERRQ(ierr);
	ierr = VecDuplicate(v1,&v2);

	ierr = VecCreate(PETSC_COMM_WORLD,&v3);				CHKERRQ(ierr);
	ierr = VecSetSizes(v3,PETSC_DECIDE,problem->p);			CHKERRQ(ierr);
	ierr = VecSetFromOptions(v3);					CHKERRQ(ierr);
	
	double *v1pt, *v3pt;
	VecGetArray(v1,&v1pt);	VecGetArray(v3,&v3pt);

	for(int i = 0; i<problem->n; i++)	v1pt[i] = xpt[i];
	for(int i = 0; i<problem->p; i++)	v3pt[i] = xpt[problem->n + i];

	VecRestoreArray(v1,&v1pt);	VecRestoreArray(v3,&v3pt);
	
	//Perform actual Manipulations
	ierr = MatMult(problem->Psi,v1,v2);				CHKERRQ(ierr);
	ierr = MatMultAdd(problem->HX,v3,v2,v1);			CHKERRQ(ierr);
	ierr = MatMultTranspose(problem->HX,v1,v3);			CHKERRQ(ierr);

	//Restore y
	VecGetArray(v1,&v1pt);	VecGetArray(v3,&v3pt);
	for(int i = 0; i<problem->n; i++)	ypt[i] = v1pt[i];
	for(int i = 0; i<problem->p; i++)	ypt[problem->n + i] = v3pt[i];
	ierr = VecRestoreArray(y,&ypt);					CHKERRQ(ierr);


	//Clear and destroy stuff
	VecDestroy(v1);		VecDestroy(v2);		VecDestroy(v3);
	v1pt = NULL;	v3pt = NULL;
	xpt = NULL;	ypt = NULL;

	ctx = NULL;
	problem = NULL;
	return ierr;
}

//Compute extreme eigenvalues for computing Square root using Cheb polynomials
void Problem::ComputeExtremeEigenValues(double lambda_min,double lambda_max)
{	
	// IDO must be started at zero, used for internal control
    	int IDO = 0;
    	// Specifies that the problem is the 'standard' EV problem
    	// Ax = Lx
    	static char BMAT = 'I';
    	// Compute the smallest/largest eigenvalues 
    	static char WHICHSM[3]="BE";
    	// Number of eigenvalues to compute
    	int NEV = 2;
    	// Dimension of the eigenproblem
    	int N = this->m;
    	// The residual vector
    	double *RESID = new double[N];
    	// Number of matrix columns
    	int NCV = N/2;
    	// Leading dimension of matrix
    	int LDV = N;
    	// Contains the Lanczoz basis vectors upon exit
    	double* V= new double[N*NCV];
    	/// Parameter array
    	int IPARAM[11];
    	// 'Pointer' array to starting and ending locations in arrays
    	int IPNTR[11];
    	// Work vector
    	double *WORKD = new double[3*N];
    	int LWORKL = NCV*(NCV + 8);
    	double *WORKL = new double[LWORKL];
    	// Result output
    	int INFO = 0;
    	// Stopping criterion
    	double TOL = EIG_TOL;
    	int RVEC = 0;
    	char ALL = 'A';
    	int* SELECT = new int[NCV];
    	double D[NEV]; 
    	double SIGMA;

    	// shifts
    	IPARAM[0] = 1;
    	// max iterations
    	IPARAM[2] = 100;
    	// mode
    	IPARAM[6] = 1;
    	// compute the eigenvalue range with arpack
    	int nits=0;
    	do
   	{
        	dsaupd_(&IDO,&BMAT,&N,WHICHSM,&NEV,&TOL,RESID,&NCV,
                	V, &LDV, IPARAM, IPNTR, WORKD, WORKL, &LWORKL, &INFO);
        	nits++;
        	switch (IDO)
       		{
            		// compute the matrix product again
            		case -1:
#ifdef bbFMM
				bbFMM->Compute(&WORKD[IPNTR[0]-1])
#endif				
                		break;
            		case 1:
		
                		break;
            		default:
                		break;
        	}
        
    	}
	while (IDO != 99);
    	dseupd_(&RVEC,&ALL,SELECT,D,V,&LDV,&SIGMA,&BMAT,&N,WHICHSM,&NEV,
            	&TOL,RESID,&NCV,V,&LDV,IPARAM,IPNTR,WORKD,WORKL,&LWORKL,&INFO);
    

	lambda_min = D[0];
    	lambda_max = D[1];

	return;

}

	
