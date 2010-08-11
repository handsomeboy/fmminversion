#ifndef PROBLEM_H
#define PROBLEM_H

#include <iostream>
#include "petscksp.h"
#include "../bbfmm/bbfmm.h"

#define EIG_TOL 1.e-7

using std::cout;
using std::endl;
using std::string;

//Arpack wrappers
extern "C" 
{
	void dsaupd_(int*,char*,int*,char*,int*,double*,double*,int*,double*,\
		int*,int*,int*,double*,double*,int*,int*);	

	void dseupd_(int*,char*,int*,double*,double*,int*,double*,char*,int*,\
		char*,int*,double*,double*,int*,double*,int*,int*,int*,double*,\
			double*,int*,int*);
}



class Problem
{
	public:
	//Constructors, destructors and things in between
	Problem(void);
	Problem(int,int,string);
	~Problem(void);

	//Set sizes
	PetscErrorCode SetSizes(int,int,string);

	//Setup the linear system
	Mat A;
	Vec b,x;			

	//Internal matrices/vectors
	Mat Q,H,R,X;
	Mat HX,Psi;
	Vec y;
	
	//Constants
	double theta1,theta2;

	//Sizes
	int m, n, p;

	//Petsc Stuff
	PetscErrorCode ierr;
	KSP ksp;
	PC pc;
	
	//FMM stuff
#ifdef bbFMM
	bbFMMWrapper* bbFMM;
#endif

	//Compute Extreme Eigenvalues
	void ComputeExtremeEigenValues(double,double);

};

class bbFMMWrapper
{
	public:
	KLaplacian *kernel;
	
	//Locations of sources and target points
	Vector3 *sourcepos, *observpos;
	
	//Precomputed SVD Matrices
	varType **U;     // Truncated observation singular vectors
    	varType **V;     // Truncated source singular vectors
    	varType **K;     // Compressed kernel matrix
        
	//Initialize bbfMM parameters
	FMMParam *param;
	FMMNode *tree;
	
	//Dimensions of the enclosing box
	Vector3 L, boxcntr;

	int Nf, Ns, n;
	
	//Functions
	bbFMMWrapper(int);
	~bbFMMWrapper(void);

	//Random source and observation locations
	void SetupRandom(int,int);
	void Setup(void);

	//Compute Interactions
	void Compute(double*,double*);
};

//Define the Matrix shell operators
PetscErrorCode RMult(Mat,Vec,Vec);

PetscErrorCode AMult(Mat,Vec,Vec);
PetscErrorCode QMult(Mat,Vec,Vec);

PetscErrorCode HXMult(Mat,Vec,Vec);
PetscErrorCode HXMultAdd(Mat,Vec,Vec);
PetscErrorCode HXMultTranspose(Mat,Vec,Vec);

PetscErrorCode PsiMult(Mat,Vec,Vec);

#endif

