/** @file bbfmm.cpp
 *  @author Will Fong <willfong@gmail.com>
 *  @version 1.0
 * 
 *  Implements the functions of bbFMM.
 
 *  @mainpage
 *  "The Black-Box Fast Multipole Method (bbFMM)"
 *
 *  The kernel needs to be translation-invariant and homogeneous
 *
 *  Also n cannot be 1 (memory management issue - delete instead of delete [])
 *
 *  Future work: Non-homogeneous kernels, symmetric kernels, different 
 *  cutoffs for each dof, adaptive cutoff selection
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include "bbfmm.h"

// Constructor for parameters in bbFMM
FMMParam::FMMParam(int Nf0, int Ns0, Vector3 boxcntr0, Vector3 L0, int n0) : 
Nf(Nf0), Ns(Ns0), boxcntr(boxcntr0), n(n0), levels(0), ucutoff(NULL), 
vcutoff(NULL) {
    // Computes the smallest cube that encloses the simulation box
    varType maxlength = L0.x;
    if (L0.y > maxlength)
        maxlength = L0.y;
    if (L.z > maxlength)
        maxlength = L0.z;
    L = Vector3(maxlength,maxlength,maxlength);
    
    // Determine the number of levels in the FMM tree
    int l;
    if (Ns > Nf)
        l = (int)log(Ns*(maxlength*maxlength*maxlength)/
                     (L0.x*L0.y*L0.z))/log(8.0);
    else
        l = (int)log(Nf*(maxlength*maxlength*maxlength)/
                     (L0.x*L0.y*L0.z))/log(8.0);
    
    // Ensures at least 8 points per leaf node (for better efficiency)
    if (l > 0)
        levels = --l;
    else
        levels = l;   
    cout << "Number of levels in FMM tree: " << l << endl;
}

// Constructor for node in FMM tree
FMMNode::FMMNode(Vector3 center0, Vector3 dim0) : center(center0), dim(dim0),
    Nf(0), Ns(0), nneigh(0), ninter(0), isLeaf(true) {
    // Set all pointers to NULL
    for (int i=0; i<8; i++)
        child[i] = NULL;
    parent = NULL;
    
    for (int i=0; i<27; i++)
        neighbors[i] = NULL;
    
    for (int i=0; i<189; i++)
        interaction[i] = NULL;
    
    localcoeff = NULL;
    multcoeff = NULL;
    cmultcoeff = NULL;
    observlist = NULL;
    sourcelist = NULL;
}

/* 
 * Function: AccuracyTest
 * ----------------------------------------------------------------------------
 * Performs an accuracy test for the specified kernel and number of Chebyshev
 * nodes using a set of Ns sources randomly placed according to the uniform 
 * distribution in a box with the specified dimensions and center location 
 * - the observation points are taken to be the first 100 sources -
 * returns the 2-norm error for each observation dof as compared to the
 * reference solution obtained by direct calculation (for non-periodic systems)
 * or Ewald (periodic).
 *
 */
void AccuracyTest(int Ns, Vector3 boxcntr, Vector3 L, Kernel *kernel, int n) {
    // Initialize the locations of the sources
    Vector3 *sourcepos = new Vector3[Ns];
    for (int j=0; j<Ns; j++)
        sourcepos[j] = boxcntr + 
            Vector3(frand(-0.5,0.5)*L.x,frand(-0.5,0.5)*L.y,
                    frand(-0.5,0.5)*L.z);
    
    // Initialize the locations of the observation points
    int Nf = Ns;
    if (Nf > 100) Nf = 100;
    Vector3 *observpos = new Vector3[Nf];
    for (int i=0; i<Nf; i++)
        observpos[i] = sourcepos[i];
    
    // Initialize the field value coefficients and source strengths
    int odof = kernel->odof;
    int sdof = kernel->sdof;
    varType *qf = new varType[odof*Nf]; // Field value coefficients
    varType *qs = new varType[sdof*Ns]; // Source strengths
    
    int index = 0;  // Index for qf
    for (int i=0; i<Nf; i++) {
        for (int idof=0; idof<odof; idof++) {
            if (i%2 == 0)
                qf[index++] = 1.0;
            else
                qf[index++] = -1.0;
        }
    }
    
    index = 0;  // Reset index for qs
    for (int j=0; j<Ns; j++) {
        for (int jdof=0; jdof<sdof; jdof++) {
            if (j%2 == 0)
                qs[index++] = 1.0;
            else
                qs[index++] = -1.0;
        }
    }
    
    // Initialize bbFMM parameters and root node (cell encloses simulation box)
    FMMParam *param = new FMMParam(Nf,Ns,boxcntr,L,n);
    FMMNode *tree = new FMMNode(boxcntr,param->L);
    
    // Arrays for pre-computed SVD matrices
    varType **U = NULL;     // Truncated observation singular vectors
    varType **V = NULL;     // Truncated source singular vectors
    varType **K = NULL;     // Compressed kernel matrix
        
    // Allocate memory for the arrays of field values
    varType *field = new varType[odof*Nf];  // Field obtained by bbFMM
    varType *ref   = new varType[odof*Nf];  // Reference field values
    
    // Set up for bbFMM calculation
    FMMSetup(param,kernel,&tree);
    
    // Load the pre-computed SVD matrices
    LoadMatrices(&param,kernel,&U,&V,&K);
    
    // Compute field using bbFMM
    bbFMM(param,kernel,&tree,U,V,K,observpos,sourcepos,qf,qs,field);

    // Compute the reference field by direct calculation or Ewald
    ReferenceField(kernel,observpos,sourcepos,qf,qs,Nf,Ns,ref);
    
    // Compute the 2-norm error for each observation dof
    varType *error = new varType[odof];
    varType *numer = new varType[odof];
    varType *denom = new varType[odof];
    memset((void *)numer,0.0,odof*sizeof(varType));
    memset((void *)denom,0.0,odof*sizeof(varType));
    index = 0;      // Index for field values
    for (int i=0; i<Nf; i++) {
        for (int idof=0; idof<odof; idof++) {
            varType diff = field[index] - ref[index];
            numer[idof] += diff*diff;
            denom[idof] += ref[index]*ref[index];
            index++;
        }
    }
    
    // Output the error
    for (int idof=0; idof<odof; idof++) {
        error[idof] = sqrt(numer[idof]/denom[idof]);
        cout << "2-norm error in observation dof #" << idof+1 << 
        " is " << scientific << setprecision(5) << error[idof] << endl;
    }
    
    // Free up allocated memory
    FMMCleanup(param,kernel,tree,U,V,K);
    delete [] observpos;
    delete [] sourcepos;
    delete [] qf;
    delete [] qs;
    delete [] field;
    delete [] ref;
    
    // Set pointers to NULL
    observpos = NULL;
    sourcepos = NULL;
    qf = NULL;
    qs = NULL;
    field = NULL; 
    ref = NULL;
}

/* 
 * Function: ReferenceField
 * ----------------------------------------------------------------------------
 * Computes the reference field by direct calculation (for non-periodic systems)
 * or Ewald (periodic).
 *
 */
void ReferenceField(Kernel *kernel, Vector3 *observpos, Vector3 *sourcepos, 
                    varType *qf, varType *qs, int Nf, int Ns, varType *field) {
    // Parameters
    int odof = kernel->odof;    // Observation degrees of freedom
    int sdof = kernel->sdof;    // Source degrees of freedom
    
    // Parameters for BLAS matrix-vector multiplication
    char *trans   = "n";    // Indicator for matrix transpose
    varType alpha = 1.0;    // Coefficient for matrix-vector product
    varType beta  = 1.0;    // Add to previous value in storage array
    int incr      = 1;      // Stride for storing matrix-vector product
    
    // Array for storing field values
    varType *fval = new varType[odof*Nf];
    memset((void *)fval,0.0,odof*Nf*sizeof(varType));
    
    // Array for kernel values
    varType *kval = new varType[odof*sdof];
    
    // Loop over all observation points
    for (int i=0; i<Nf; i++) {
        // Loop over all sources
        for (int j=0; j<Ns; j++) {
            // Compute interaction directly
            kernel->evaluate_exclude(observpos[i],sourcepos[j],kval);
            #ifdef SINGLE
            sgemv_(trans,&odof,&sdof,&alpha,kval,&odof,&qs[sdof*j],
                   &incr,&beta,&fval[odof*i],&incr);
            #else
            dgemv_(trans,&odof,&sdof,&alpha,kval,&odof,&qs[sdof*j],
                   &incr,&beta,&fval[odof*i],&incr);
            #endif
        }
    }
    
    // Multiply by the field coefficients qf
    int fsize = odof*Nf;        // Length of field vector
    for (int i=0; i<fsize; i++)
        field[i] = qf[i]*fval[i];   
    
    // Free allocated memory
    // Free allocated memory
    delete [] fval;
    if (odof*sdof > 1)
        delete [] kval;
    else
        delete kval;
}


/* 
 * Function: SpiralGalaxy2D
 * ----------------------------------------------------------------------------
 * Simulates the formation of a 2-D spiral galaxy from a uniform elliptic 
 * distribution of masses - the particles interact according to Newton's law of 
 * gravitation in two dimensions.
 *
 */
void SpiralGalaxy2D(int N, int n) {
    // Parameters
    int nts         = 10;         // Number of time steps
    varType dt      = 1.0e-4;       // Forward Euler time step
    varType G       = 0.05;         // Gravitational constant
    varType vfac    = 50.0;         // Velocity scaling factor
    varType alpha   = 0.25;         // Parameter for initial ellipticity
    varType length  = 1.0;          // Length of simulation square
    varType rmax    = 0.25*length;  // Length of semi-major axis
    varType thmax   = 2*M_PI;       // Maximum angle (in radians)
    Vector3 boxcntr(0.5*length,0.5*length,0.0); // Box center
    Vector3 L(length,length,0.1*length);        // Box dimensions
            
    // Kernel
    KGravForce2D *kernel = new KGravForce2D(n);
    int odof = kernel->odof;    // Observation degrees of freedom
    int sdof = kernel->sdof;    // Source degrees of freedom
    
    /* Allocates memory to store the positions, velocities, masses,
     * and forces on the point masses */
    Vector3 *pos = new Vector3[N];      // Positions of masses
    Vector3 *vel = new Vector3[N];      // Velocities of masses
    varType *m   = new varType[N];      // Masses
    varType *F   = new varType[odof*N]; // Force vector - N 2-vectors (Fx,Fy)
    
    // Sets the initial position and velocity of the point masses
    for (int i=0; i<N; i++) {
        m[i] = 1.0;
        varType Ri = frand(0.0,rmax);
        varType Thetai = frand(0.0,thmax);
        pos[i] = boxcntr + Vector3(Ri*cos(Thetai),alpha*Ri*sin(Thetai),0.1);
        Vector3 diff = pos[i] - boxcntr;
        varType r  = sqrt(diff.x*diff.x+diff.y*diff.y);
        vel[i] = Vector3(-vfac*r*sin(Thetai),vfac*r*cos(Thetai),0.0);
    }
    
    // Initialize the field value coefficients and source strengths
    varType *qf = new varType[odof*N]; // Field value coefficients (i.e. G*mass)
    varType *qs = new varType[sdof*N]; // Source strengths (i.e. mass)
    
    int index = 0;  // Index for qf
    for (int i=0; i<N; i++) {
        for (int idof=0; idof<odof; idof++)
            qf[index++] = 1.0;
    }
    
    index = 0;  // Reset index for qs
    for (int j=0; j<N; j++) {
        for (int jdof=0; jdof<sdof; jdof++)
            qs[index++] = m[j];
    }
    
    // Initialize bbFMM parameters and root node (cell encloses simulation box)
    FMMParam *param = new FMMParam(N,N,boxcntr,L,n);
    FMMNode *tree = new FMMNode(boxcntr,param->L);
    
    // Arrays for pre-computed SVD matrices
    varType **U = NULL;     // Truncated observation singular vectors
    varType **V = NULL;     // Truncated source singular vectors
    varType **K = NULL;     // Compressed kernel matrix    
    
    // Set up for bbFMM calculation
    FMMSetup(param,kernel,&tree);
    
    // Load the pre-computed SVD matrices
    LoadMatrices(&param,kernel,&U,&V,&K);    
    
    // Write out the initial configuration
    ofstream outfile("ts0.dat");
    for (int i=0; i<N; i++)
        outfile << scientific << setprecision(6) 
            << pos[i].x << " " << pos[i].y << endl;

    // Integrate the system using first-order forward Euler
    for (int t=0; t<nts; t++) {
        // Output the time step
        if (t%10 == 9)
            cout << "Time step " << t+1 << " out of " << nts << endl; 
        
        // Set force vector to zero
        memset((void *)F,0.0,odof*N*sizeof(varType));
        
        // Compute the force vector using bbFMM
        bbFMM(param,kernel,&tree,U,V,K,pos,pos,qf,qs,F);
            
        // Update the velocities and positions
        for (int i=0; i<N; i++) {
            vel[i].x += dt*G*F[2*i];
            vel[i].y += dt*G*F[2*i+1];
            pos[i] += dt*vel[i];
        }
                
        // Outputs configuration every 250 ts
        if (t%250 == 249) {
            ofstream outfile;
            switch(t) {
                case 249:
                    outfile.open("ts250.dat");
                    break;
                case 499:
                    outfile.open("ts500.dat");
                    break;
                case 749:
                    outfile.open("ts750.dat");
                    break;
                case 999:
                    outfile.open("ts1000.dat");
                    break;
            }
            for (int i=0; i<N; i++)
                outfile << scientific << setprecision(6) 
                    << pos[i].x << " " << pos[i].y << endl;
        }
        
        // Clear all arrays in the FMM tree
        FMMClear(tree);
    }
    
    // Frees the allocated memory
    FMMCleanup(param,kernel,tree,U,V,K);
    delete [] pos;
    delete [] vel;
    delete [] m;
    delete [] F;
}

/*
 * Function: FMMSetup
 * ----------------------------------------------------------------------------
 * Prepare for the FMM calculation by building the FMM tree
 * and pre-computing the SVD (if necessary).
 *
 */
void FMMSetup(FMMParam *param, Kernel *kernel, FMMNode **tree) {
    // Number of levels in FMM tree
    int l = param->levels;
        
    // Builds the FMM tree
    BuildFMMTree(tree,l);
    
    // Check if the SVD of the kernel interaction matrix needs to be computed
    int odof = kernel->odof;
    bool computeSVD = false;
    ifstream Umatfile, Vmatfile;
    for (int idof=0; idof<odof; idof++) {
        Umatfile.open(kernel->Umat[idof].c_str());
        Vmatfile.open(kernel->Vmat[idof].c_str());
        if (Umatfile.fail() || Vmatfile.fail())
            computeSVD = true;
    }
    
    // Compute the SVD if necessary
    if (computeSVD)
        ComputeKernelSVD(param,kernel);
}

/*
 * Function: BuildFMMTree
 * ----------------------------------------------------------------------------
 * Recursively builds the FMM tree with l levels.
 *
 */
void BuildFMMTree(FMMNode **A, int l) {
    // Continue if not at the bottom of the tree
    if (l > 0) {
        // Compute the length and center coordinate of the children cells
        Vector3 dim     = (*A)->dim;
        Vector3 half    = 0.5*dim;
        Vector3 quarter = 0.25*dim;
        Vector3 pcenter = (*A)->center;       // Center of parent cell
        Vector3 left    = pcenter - quarter;  // Center of left subdivisions
        Vector3 right   = pcenter + quarter;  // Center of right subdivisions
        
        // Node A is no longer a leaf node
        (*A)->isLeaf = false;
        
        // Add children nodes to node A
        for (int i=0; i<8; i++) {
            // The center of the i-th child cell
            Vector3 ccenter;
            
            // Determine the x and y coordinates of center
            if (i < 4) {
                ccenter.x = left.x;
                
                if (i < 2)
                    ccenter.y = left.y;
                else
                    ccenter.y = right.y;
                
            } else {
                ccenter.x = right.x;
                
                if (i < 6)
                    ccenter.y = left.y;
                else
                    ccenter.y = right.y;
            }
            
            // Determine the z coordinate of center
            if (i%2 == 0)
                ccenter.z = left.z;
            else
                ccenter.z = right.z;
            
            // Create the child cell
            (*A)->child[i] = new FMMNode(ccenter,half);
            (*A)->child[i]->parent = *A;
        }
        
        // Recursively build octree if there is a subsequent level
        for (int i=0; i<8; i++)
            BuildFMMTree(&((*A)->child[i]),l-1);
    }
}

/*
 * Function: FMMCleanup
 * ---------------------------------------------------------------------------
 * Cleans up after FMM calculation.
 *
 */
void FMMCleanup(FMMParam *param, Kernel *kernel, FMMNode *tree, varType **U,
                varType **V, varType **K) {
    // Free cutoff arrays and paramater instance
    delete [] param->ucutoff;
    delete [] param->vcutoff;
    delete param;
    
    // Free memory allocated to FMM tree
    FreeNode(tree);
    
    // Free memory used for pre-computed SVD matrices
    int odof = kernel->odof;
    for (int idof=0; idof<odof; idof++) {
        delete [] U[idof];
        delete [] V[idof];
        delete [] K[idof];
    }
    if (odof > 1) {
        delete [] U;
        delete [] V;
        delete [] K;
    } else {
        delete U;
        delete V;
        delete K;
    }
    
    // Set pointers to NULL
    param = NULL;
    tree = NULL;
    U = NULL;
    V = NULL;
    K = NULL;
}

/*
 * Function: FreeNode
 * ------------------------------------------------------------------
 * Frees up memory associated with FMMNode.
 *
 */
void FreeNode(FMMNode *A) {
    // Free all child nodes first
    for (int i=0; i<8; i++) {
        if (A->child[i] != NULL) {
            FreeNode(A->child[i]);
        }
    }
    
    // Free the arrays for the local and multipole coefficients
    if (A->localcoeff != NULL) {
        delete [] A->localcoeff;
        A->localcoeff = NULL;
    }
    if (A->multcoeff != NULL) {
        delete [] A->multcoeff;
        A->multcoeff = NULL;
    }
    if (A->cmultcoeff != NULL) {
        delete [] A->cmultcoeff;
        A->cmultcoeff = NULL;
    }
    
    // Free the observation point and source index lists
    if (A->observlist != NULL) {
        delete [] A->observlist;
        A->observlist = NULL;
    }
    if (A->sourcelist != NULL) {
        delete [] A->sourcelist;
        A->sourcelist = NULL;
    }
    
    // Last free the node
    delete A;
    A = NULL;
}

/*
 * Function: ComputeKernelSVD
 * ----------------------------------------------------------------------------
 * Evaluates the kernel for cell-to-cell interactions corresponding to the 
 * 316 multipole-to-local transfer vectors and computes the SVD of this kernel 
 * matrix.
 *
 */
void ComputeKernelSVD(FMMParam *param, Kernel *kernel) {     
    // Parameters
    int n  = param->n;              // Number of Chebyshev nodes
    int n3 = n*n*n;                 
    varType pi   = M_PI;
    varType pi_n = pi/(varType)n;
    int odof = kernel->odof;        // Observation degrees of freedom
    int sdof = kernel->sdof;        // Source degrees of freedom
    int sdofn3 = sdof*n3;
    
    // Compute the n Chebyshev nodes of T_n(x) (in interval [-1,1])
    varType *cnodes = new varType[n]; //Positions of Chebyshev nodes
    for (int i=0; i<n; i++)
        cnodes[i] = cos(pi*((varType)i+0.5)/(varType)n);
    
    // Compute the weighting of the 3-D Chebyshev nodes for the kernel SVD
    varType *nodeweights = new varType[n3]; // Weights for 3-D Chebyshev nodes 
    int index = 0;
    for (int ix=0; ix<n; ix++) {
        varType wx = pi_n*sqrt(1.0-cnodes[ix]*cnodes[ix]);
        for (int iy=0; iy<n; iy++) {
            varType wxwy = wx*pi_n*sqrt(1.0-cnodes[iy]*cnodes[iy]);
            for (int iz=0; iz<n; iz++)
                nodeweights[index++] = sqrt(wxwy*pi_n*
                                            sqrt(1.0-cnodes[iz]*cnodes[iz]));
        }
    }
    
    /* Compute the locations of the observation 3-D Chebyshev nodes in the 
     * cell [-0.5,0.5] x [-0.5,0.5] x [-0.5,0.5] */
    Vector3 *observpos = new Vector3[n3]; // Positions of observation nodes
    index = 0;
    for (int ix=0; ix<n; ix++) {
        for (int iy=0; iy<n; iy++) {
            for (int iz=0; iz<n; iz++)
                observpos[index++] = 0.5*
                    Vector3(cnodes[ix],cnodes[iy],cnodes[iz]);
        }
    }
    
    /* Evaluate the kernel for all Chebyshev-Chebyshev interactions between the
     * observation cell and source cells corresponding to the 7^3 - 3^3 = 316
     * transfer vectors - this includes all unit cells located in the domain
     * [-3.5,3.5] x [-3.5,3.5] x [-3.5,3.5] except the observation cell itself
     * and the cells adjacent to it - the kernel matrix is a 2-D array whose
     * first dimension is the observation dof and the second is the kernel 
     * values - the kernel values correspond to a n^3 by 316*sdof*n^3 matrix
     * (observation nodes x source nodes) stored in vector form using the 
     * column-major format - this is the fat version of the kernel matrix */
    varType **Kfat = new varType*[odof];    // Kernel matrix (fat version)
    int c2cdof = n3*(316*sdofn3);           // Number of cell-to-cell dof
    for (int idof=0; idof<odof; idof++)
        Kfat[idof] = new varType[c2cdof];
    Vector3 *sourcepos = new Vector3[n3];   // Positions of source nodes
    varType **Ksub = new varType*[odof];    // Points to entries in matrix Kfat

    int kindex = 0;  // Index for matrix K
    for (int nx=-3; nx<4; nx++) {
        for (int ny=-3; ny<4; ny++) {
            for (int nz=-3; nz<4; nz++) {
                if (abs(nx) > 1 || abs(ny) > 1 || abs(nz) > 1) {
                    // Center of source cell
                    Vector3 scenter((varType)nx,(varType)ny,(varType)nz);
                    
                    /* Compute the locations of the source 3-D Chebyshev nodes
                     * in the unit cell whose center is located at scenter */
                    int sindex = 0;   // Index for source nodes
                    for (int jx=0; jx<n; jx++) {
                        for (int jy=0; jy<n; jy++) {
                            for (int jz=0; jz<n; jz++)
                                sourcepos[sindex++] = scenter + 0.5*
                                    Vector3(cnodes[jx],cnodes[jy],cnodes[jz]);
                        }
                    }
                    
                    // Set up the pointers to appropriate entries in K
                    for (int idof=0; idof<odof; idof++)
                        Ksub[idof] = &(Kfat[idof][kindex]);
                    kindex += sdof*n3*n3;
                    
                    // Evaluate the kernel interactions between the two cells
                    EvaluateKernelCell2Cell(kernel,observpos,sourcepos,n3,
                                            nodeweights,Ksub);
                }
            }
        }
    }
    
    /* Form the thin version of the kernel matrix - the kernel values 
     * correspond to a 316*n^3 by sdof*n^3 matrix, stored in vector form 
     * using the column-major format */
    varType **Kthin = new varType*[odof];    // Kernel matrix (thin version)
    for (int idof=0; idof<odof; idof++)
        Kthin[idof] = new varType[c2cdof];
    for (int idof=0; idof<odof; idof++) {
        int tindex = 0;     // Index for Kthin
        for (int jdof=0; jdof<sdof; jdof++) {
            for (int j=0; j<n3; j++) {
                for (int icell=0; icell<316; icell++) {
                    // findex = icell*sdof*n3*n3 + jdof*n3*n3 + j*n3;
                    int findex = ((icell*sdof + jdof)*n3 + j)*n3; // Kfat index
                    for (int i=0; i<n3; i++)
                        Kthin[idof][tindex++] = Kfat[idof][findex++];
                }
            }
        }
    }
    
    // SVD parameters
    char *save   = "s";
    char *nosave = "n";
    int nosavedim = 1;
    int info;
        
    // Allocate memory for computing the SVD of Kfat
    int Krows  = n3;
    int Kcols  = 316*sdofn3;
    int lwork  = 3*Kcols;
    varType **U      = new varType*[odof];  // Observation singular vectors
    varType **Sigmaf = new varType*[odof];  // Observation singular values    
    varType *workf   = new varType[lwork];  // Matrix for SVD work
    varType *A       = NULL;  // A null pointer for unnecessary singular vectors 
    for (int idof=0; idof<odof; idof++) {
        U[idof]      = new varType[Krows*Krows];
        Sigmaf[idof] = new varType[Krows];
    }
    
    /* Compute the SVD of Kfat to find the observation singular vectors
     * for each observation dof */
    for (int idof=0; idof<odof; idof++) {
        #ifdef SINGLE
        sgesvd_(save,nosave,&Krows,&Kcols,Kfat[idof],&Krows,Sigmaf[idof],
                U[idof],&Krows,A,&nosavedim,workf,&lwork,&info);
        #else
        dgesvd_(save,nosave,&Krows,&Kcols,Kfat[idof],&Krows,Sigmaf[idof],
                U[idof],&Krows,A,&nosavedim,workf,&lwork,&info);
        #endif
        
        // Write out the observation singular vectors to file
        ofstream Umatfile(kernel->Umat[idof].c_str());
        int uindex = 0; // Index for U[idof]
        for (int j=0; j<n3; j++) {
            for (int i=0; i<n3; i++)
                Umatfile << scientific << setprecision(13) 
                    << U[idof][uindex++] << endl;
        }
    }
    
    // Free memory for work matrix
    delete [] workf;
        
    // Allocate memory for computing the SVD of Kthin
    Krows  = 316*n3;
    Kcols  = sdofn3;
    lwork  = 3*Krows;
    varType **VT     = new varType*[odof];  // Source singular vectors
    varType **Sigmat = new varType*[odof];  // Source singular values    
    varType *workt   = new varType[lwork];  // Matrix for SVD work
    A = NULL;  // A null pointer for unnecessary singular vectors
    for (int idof=0; idof<odof; idof++) {
        VT[idof]     = new varType[Kcols*Kcols];
        Sigmat[idof] = new varType[Kcols];
    }
    
    /* Compute the SVD of Kthin to find the source singular vectors
     * for each observation dof */
    for (int idof=0; idof<odof; idof++) {
        #ifdef SINGLE
        sgesvd_(nosave,save,&Krows,&Kcols,Kthin[idof],&Krows,Sigmat[idof],
                A,&nosavedim,VT[idof],&Kcols,workt,&lwork,&info);
        #else
        dgesvd_(nosave,save,&Krows,&Kcols,Kthin[idof],&Krows,Sigmat[idof],
                A,&nosavedim,VT[idof],&Kcols,workt,&lwork,&info);
        #endif
        
        // Write out the source singular vectors to file
        ofstream Vmatfile(kernel->Vmat[idof].c_str());        
        for (int j=0; j<sdofn3; j++) {
            int vindex = j;  // Index for VT[idof]
            for (int jdof=0; jdof<sdof; jdof++) {
                for (int i=0; i<n3; i++) {
                    Vmatfile << scientific << setprecision(13) 
                        << VT[idof][vindex] << endl;
                    vindex += sdofn3;
                }
            }
        }
    }
    
    // Free memory for work matrix
    delete [] workt;
        
    // Free allocated memory
    delete [] cnodes;
    delete [] nodeweights;
    delete [] observpos;
    delete [] sourcepos;
    
    for (int idof=0; idof<odof; idof++) {
        delete [] Kfat[idof];
        delete [] Kthin[idof];
        delete [] U[idof];
        delete [] VT[idof];
        delete [] Sigmaf[idof];
        delete [] Sigmat[idof];
    }
    
    if (odof > 1) {
        delete [] Kfat;
        delete [] Kthin;
        delete [] Ksub;
        delete [] U;
        delete [] VT;
        delete [] Sigmaf;
        delete [] Sigmat;
    } else {
        delete Kfat;
        delete Kthin;
        delete Ksub;
        delete U;
        delete VT;
        delete Sigmaf;
        delete Sigmat;
    }
    
    // Set pointers to NULL
    cnodes = NULL;
    nodeweights = NULL;
    observpos = NULL;
    sourcepos = NULL;
    Kfat = NULL;
    Kthin = NULL;
    Ksub = NULL;
}

/*
 * Function: EvaluateKernelCell2Cell
 * ----------------------------------------------------------------------------
 * Evaluates the kernel for Chebyshev-Chebyshev interactions between a pair of 
 * cells and applies the appropriate SVD weighting.
 *
 */
void EvaluateKernelCell2Cell(Kernel *kernel, Vector3 *observpos, 
                             Vector3 *sourcepos, int n3, 
                             varType *nodeweights, varType **Ksub) {
    // Parameters
    int odof = kernel->odof;  // Observation degrees of freedom
    int sdof = kernel->sdof;  // Source degrees of freedom
    int stride = n3*n3;       // Stride for subsequent source dof
    
    varType *Kij = kernel->value;   // Array for kernel evaluation
    
    for (int j=0; j<n3; j++) {      // Loop over all source nodes
        int start = j*n3;           // Index in Ksub[odof] for j-th source node
        varType sweight = nodeweights[j];       // Weight for source node
        
        for (int i=0; i<n3; i++) {  // Loop over all observation nodes
            int kindex = start + i; // Index in Ksub[odof] for i-th observ. node
            varType oweight = nodeweights[i];   // Weight for observ. node
            varType weight = sweight*oweight;   // Weight for ij interaction
            
            // Computes the field at i-th observation point due to j-th source
            kernel->evaluate(observpos[i],sourcepos[j],Kij);
            
            // Insert the kernel values into the matrix K
            int count = 0;          // Index for Kij
            for (int jdof=0; jdof<sdof; jdof++) {
                for (int idof=0; idof<odof; idof++)
                    Ksub[idof][kindex] = weight*Kij[count++];
                kindex += stride;
            }
        }
    }
}

/*
 * Function: LoadMatrices
 * ---------------------------------------------------------------------------
 * Read in the matrix of observation singular vectors U and the matrix of 
 * source singular vectors V and compute the compressed kernel matrix K.
 *
 */
void LoadMatrices(FMMParam **param, Kernel *kernel, varType ***U, varType ***V, 
                  varType ***K) {    
    // Parameters
    int n = (*param)->n;        // Number of Chebyshev nodes in each direction
    int n3 = n*n*n;
    int odof = kernel->odof;    // Observation degrees of freedom
    int sdof = kernel->sdof;    // Source degrees of freedom
    int sdofn3 = sdof*n3;
    varType pi   = M_PI;
    varType pi_n = pi/(varType)n;
    
    // Compute the n Chebyshev nodes of T_n(x) (in interval [-1,1])
    varType *cnodes = new varType[n]; //Positions of Chebyshev nodes
    for (int i=0; i<n; i++)
        cnodes[i] = cos(pi*((varType)i+0.5)/(varType)n);
    
    // Compute the weighting of the 3-D Chebyshev nodes for the kernel SVD
    varType *nodeweights = new varType[n3];//Weights for 3-D Chebyshev nodes 
    int index = 0;
    for (int ix=0; ix<n; ix++) {
        varType wx = pi_n*sqrt(1.0-cnodes[ix]*cnodes[ix]);
        for (int iy=0; iy<n; iy++) {
            varType wxwy = wx*pi_n*sqrt(1.0-cnodes[iy]*cnodes[iy]);
            for (int iz=0; iz<n; iz++)
                nodeweights[index++] = sqrt(wxwy*pi_n*
                                            sqrt(1.0-cnodes[iz]*cnodes[iz]));
        }
    }
    
    // Determine number of singular vectors to keep (same for all dof)
    int ucutoff = n3/2;         // Observation singular vectors
    int vcutoff = sdofn3/2;     // Source singular vectors
    int uvdof = ucutoff*vcutoff;
    
    // Store the cutoffs
    (*param)->ucutoff = new int[odof];
    (*param)->vcutoff = new int[odof];
    for (int idof=0; idof<odof; idof++) {
        (*param)->ucutoff[idof] = ucutoff;
        (*param)->vcutoff[idof] = vcutoff;
    }
    
    // Size of pre-computed SVD matrices
    int Ksize = 316*uvdof;          // Compressed kernel matrix
    int Usize = ucutoff*n3;         // Truncated observ. singular vectors
    int Vsize = vcutoff*sdofn3;     // Truncated source singular vectors
    
    // Allocate memory for matrices
    (*U) = new varType*[odof];
    (*V) = new varType*[odof];
    (*K) = new varType*[odof];
    for (int idof=0; idof<odof; idof++) {
        (*U)[idof] = new varType[Usize];
        (*V)[idof] = new varType[Vsize];
        (*K)[idof] = new varType[Ksize];
    }
    
    // Read in observation singular vectors U for each observation dof
    for (int idof=0; idof<odof; idof++) {
        ifstream Umatfile(kernel->Umat[idof].c_str());
        for (int i=0; i<Usize; i++)
            Umatfile >> (*U)[idof][i];
    }
    
    // Read in source singular vectors V for each observation dof
    for (int idof=0; idof<odof; idof++) {
        ifstream Vmatfile(kernel->Vmat[idof].c_str());
        for (int i=0; i<Vsize; i++)
            Vmatfile >> (*V)[idof][i];
    }
    
    // Determine if the compressed kernel matrix needs to be computed
    bool computeK = false;
    ifstream Kmatfile;
    for (int idof=0; idof<odof; idof++) {
        Kmatfile.open(kernel->Kmat[idof].c_str());
        if (Kmatfile.fail())
            computeK = true;
    }
    
    // Compute the compressed kernel matrix if necessary - otherwise read it
    if (computeK) {
        /* Compute the locations of the observation 3-D Chebyshev nodes in the 
         * cell [-0.5,0.5] x [-0.5,0.5] x [-0.5,0.5] */
        Vector3 *observpos = new Vector3[n3]; // Positions of observation nodes
        index = 0;
        for (int ix=0; ix<n; ix++) {
            for (int iy=0; iy<n; iy++) {
                for (int iz=0; iz<n; iz++)
                    observpos[index++] = 0.5*
                        Vector3(cnodes[ix],cnodes[iy],cnodes[iz]);
            }
        }
        
        /* Evaluate the kernel for all Chebyshev-Chebyshev interactions between 
         * the observation cell and source cells corresponding to the 7^3 - 3^3 
         * = 316 transfer vectors - this includes all unit cells located in the 
         * domain [-3.5,3.5] x [-3.5,3.5] x [-3.5,3.5] except the observation 
         * cell itself and the cells adjacent to it - then post-multiply the
         * kernel matrix for each cell-to-cell interaction with the truncated 
         * source singular vectors and pre-multiply by the transpose of the 
         * observation singular vectors - store the resulting ucutoff-by-vcutoff
         * matrix in the array K (column-major format) and output to file */
        varType **Kcell = new varType*[odof];   // Cell-to-cell kernel matrix
        int c2cdof = n3*sdofn3;                 // Number of cell-to-cell dof
        for (int idof=0; idof<odof; idof++)
            Kcell[idof] = new varType[c2cdof];
        Vector3 *sourcepos = new Vector3[n3];   // Positions of source nodes
        varType **Ksub = new varType*[odof];    // Points to entries in Kcell
        varType *KV = new varType[n3*vcutoff];  // Post-multiplied kernel matrix
        
        int kindex = 0;  // Index for compressed kernel matrix
        for (int nx=-3; nx<4; nx++) {
            for (int ny=-3; ny<4; ny++) {
                for (int nz=-3; nz<4; nz++) {
                    if (abs(nx) > 1 || abs(ny) > 1 || abs(nz) > 1) {
                        // Center of source cell
                        Vector3 scenter((varType)nx,(varType)ny,(varType)nz);
                        
                        /* Compute the locations of the source 3-D Chebyshev nodes
                         * in the unit cell whose center is located at scenter */
                        int sindex = 0;   // Index for source nodes
                        for (int jx=0; jx<n; jx++) {
                            for (int jy=0; jy<n; jy++) {
                                for (int jz=0; jz<n; jz++)
                                    sourcepos[sindex++] = scenter + 0.5*
                                    Vector3(cnodes[jx],cnodes[jy],cnodes[jz]);
                            }
                        }
                        
                        // Set up the pointers to appropriate entries in K
                        for (int idof=0; idof<odof; idof++)
                            Ksub[idof] = &(Kcell[idof][0]);
                        
                        // Evaluate the kernel interactions between the two cells
                        EvaluateKernelCell2Cell(kernel,observpos,sourcepos,n3,
                                                nodeweights,Ksub);
                        
                        // Compute the compressed kernel matrix U^T Kcell V
                        for (int idof=0; idof<odof; idof++) {
                            // Compute Kcell V (post-multiply by source vectors)
                            char *transa = "n";
                            char *transb = "n";
                            varType alpha = 1.0;
                            varType beta = 0.0;
                            
                            #ifdef SINGLE
                            sgemm_(transa,transb,&n3,&vcutoff,&sdofn3,&alpha,
                                   Kcell[idof],&n3,(*V)[idof],&sdofn3,&beta,
                                   KV,&n3); 
                            #else
                            dgemm_(transa,transb,&n3,&vcutoff,&sdofn3,&alpha,
                                   Kcell[idof],&n3,(*V)[idof],&sdofn3,&beta,
                                   KV,&n3);
                            #endif
                            
                            // Then compute U^T KV (pre-multiply by obs vectors)
                            transa = "t";
                            transb = "n";
                            
                            #ifdef SINGLE
                            sgemm_(transa,transb,&ucutoff,&vcutoff,&n3,&alpha,
                                   (*U)[idof],&n3,KV,&n3,&beta,
                                   &((*K)[idof][kindex]),&ucutoff);
                            #else
                            dgemm_(transa,transb,&ucutoff,&vcutoff,&n3,&alpha,
                                   (*U)[idof],&n3,KV,&n3,&beta,
                                   &((*K)[idof][kindex]),&ucutoff);
                            #endif
                        }
                        
                        // Increment index of compressed kernel matrix
                        kindex += uvdof;
                    }
                }
            }
        }
        
        // Write out the compressed kernel matrix to file
        for (int idof=0; idof<odof; idof++) {
            ofstream Kmatfile(kernel->Kmat[idof].c_str());        
            for (int i=0; i<Ksize; i++)
                Kmatfile << scientific << setprecision(13) 
                    << (*K)[idof][i] << endl;
        }
        
        // Free allocated memory
        delete [] observpos;
        
        for (int idof=0; idof<odof; idof++)
            delete [] Kcell[idof];
        if (odof > 1) {
            delete [] Kcell;
            delete [] Ksub;
        } else {
            delete Kcell;
            delete Ksub;
        }
        
        delete [] sourcepos;
        delete [] KV;
    } else {
        for (int idof=0; idof<odof; idof++) {
            ifstream Kmatfile(kernel->Kmat[idof].c_str());
            for (int i=0; i<Ksize; i++)
                Kmatfile >> (*K)[idof][i];
        }
    }
    
    // Compute the inverse weighting of the 3-D Chebyshev nodes
    for (int i=0; i<n3; i++)
        nodeweights[i] = 1.0/nodeweights[i];
    
    // Scale the observation singular vectors U for each observation dof
    for (int idof=0; idof<odof; idof++) {
        int uindex = 0;     // Index for matrix U[idof]
        for (int j=0; j<ucutoff; j++) {
            for (int i=0; i<n3; i++)
                (*U)[idof][uindex++] *= nodeweights[i];
        }
    }
    
    // Scale the source singular vectors V for each observation dof
    for (int idof=0; idof<odof; idof++) {
        int vindex = 0;     // Index for matrix V[idof]
        for (int j=0; j<vcutoff; j++) {
            for (int jdof=0; jdof<sdof; jdof++) {
                for (int i=0; i<n3; i++)
                    (*V)[idof][vindex++] *= nodeweights[i];
            }
        }
    }
    
    // Free allocated memory
    delete [] cnodes;
    delete [] nodeweights;
}

/*
 * Function: bbFMM
 * ----------------------------------------------------------------------------
 * Given the source and observation point locations and the strength of the 
 * sources, the field is computed for the specified kernel using bbFMM.
 *
 */
void bbFMM(FMMParam *param, Kernel *kernel, FMMNode **tree, varType **U,
           varType **V, varType **K, Vector3 *observpos, Vector3 *sourcepos, 
           varType *qf, varType *qs, varType *field) {
    // Clear the arrays in the octree
    FMMClear(*tree);
    
    // Distribute the sources and observation points and set up interaction list
    FMMDistribute(param,tree,observpos,sourcepos);
    
    // Compute the field using bbFMM
    FMMCompute(param,kernel,tree,U,V,K,observpos,sourcepos,qf,qs,field);
}

/*
 * Function: FMMClear
 * ----------------------------------------------------------------------------
 * Clears all of the arrays in the octree in preparation for the FMM
 * calculation (allows for the use of the same FMM tree).
 *
 */
void FMMClear(FMMNode *tree) {
    ClearNode(tree);
}

/*
 * Function: ClearNode
 * ------------------------------------------------------------------
 * Frees up the arrays associated with FMMNode
 *
 */
void ClearNode(FMMNode *A) {
    // Set parameters to zero (re-initialize)
    A->Nf     = 0;
    A->Ns     = 0;
    A->nneigh = 0;
    A->ninter = 0;
    
    // Free the arrays for the local and multipole coefficients
    if (A->localcoeff != NULL) {
        delete [] A->localcoeff;
        A->localcoeff = NULL;
    }
    if (A->multcoeff != NULL) {
        delete [] A->multcoeff;
        A->multcoeff = NULL;
    }
    if (A->cmultcoeff != NULL) {
        delete [] A->cmultcoeff;
        A->cmultcoeff = NULL;
    }
    
    // Free the observation point and source index lists
    if (A->observlist != NULL) {
        delete [] A->observlist;
        A->observlist = NULL;
    }
    if (A->sourcelist != NULL) {
        delete [] A->sourcelist;
        A->sourcelist = NULL;
    }
    
    // Then clear child nodes
    for (int i=0; i<8; i++) {
        if (A->child[i] != NULL) {
            ClearNode(A->child[i]);
        }
    } 
}

/*
 * Function: FMMDistribute
 * ----------------------------------------------------------------------------
 * Distribute the observation points and sources to the appropriate node 
 * in the FMM tree and builds the interaction list.
 *
 */
void FMMDistribute(FMMParam *param, FMMNode **tree, Vector3 *observpos, 
                   Vector3 *sourcepos) {
    // Parameters
    int Nf = param->Nf;     // Number of observation points
    int Ns = param->Ns;     // Number of sources
    
    // Create arrays containing indices for observation points and sources
    int *observlist = new int[Nf];
    int *sourcelist = new int[Ns];
    
    /* Initialize the point distribution for the root node
     * i.e. all observation points and sources are in the simulation box */
    for (int i=0; i<Nf; i++)
        observlist[i] = i;
    for (int j=0; j<Ns; j++)
        sourcelist[j] = j;
    (*tree)->Nf = Nf;
    (*tree)->Ns = Ns;
    
    // Determine which observation points and sources are in each subcell
    DistributeObservPoints(tree,observpos,observlist);
    DistributeSources(tree,sourcepos,sourcelist);
    
    // Construct the interaction list for all nodes in the FMM tree
    (*tree)->neighbors[0] = *tree;      // Only neighbor of root node is iself
    (*tree)->nneigh = 1;                // Hence only one neighbor
    (*tree)->shiftneigh[0] = Vector3(0.0,0.0,0.0);  // No periodic shift needed
    if (!(*tree)->isLeaf)
        BuildInteractionList(tree);     // Call only if tree has children nodes
    
    // Free allocated memory for arrays
    delete [] observlist;
    delete [] sourcelist;
}

/*
 * Function: DistributeObservPoints
 * ----------------------------------------------------------------------------
 * If the specified FMMNode is a leaf node, then store the indices of the
 * observation points located in the corresponding leaf cell - otherwise
 * determine which observation points are in each child cell of the cell 
 * corresponding to the specified FMMNode.
 *
 */
void DistributeObservPoints(FMMNode **A, Vector3 *observpos, int *observlist) {
    // Parameters
    int Nf = (*A)->Nf;  // Number of observation points
    
    // If node corresponds to a leaf cell, store the observation point indices
    if ((*A)->isLeaf) {
        (*A)->observlist = new int[Nf];
        int *F = (*A)->observlist;
        for (int i=0; i<Nf; i++) 
            F[i] = observlist[i];
    }
    
    // Otherwise determine which observation points are in each child cell
    else {        
        // Distribute the observation points (if necessary)  
        if (Nf > 0) {
            // Create index lists for each child cell
            int *cobservlist[8];
            for (int icell=0; icell<8; icell++)
                cobservlist[icell] = new int[Nf];
            
            // Obtain the center of the cell
            Vector3 center = (*A)->center;
            
            // Determine which child cell each observation point belongs to
            for (int i=0; i<Nf; i++) {
                int k = observlist[i];          // Observation point index
                Vector3 point = observpos[k];   // Location of observation point
                int j;                          // Index of child cell
                
                // Determine which cell the point is in
                if (point.x < center.x) {
                    if (point.y < center.y) {
                        if (point.z < center.z)
                            j = 0;
                        else
                            j = 1;
                    } else {
                        if (point.z < center.z)
                            j = 2;
                        else
                            j = 3;
                    }
                } else {
                    if (point.y < center.y) {
                        if (point.z < center.z)
                            j = 4;
                        else
                            j = 5;
                    } else {
                        if (point.z < center.z)
                            j = 6;
                        else
                            j = 7;
                    }
                }
                
                // Add the observation point to the list for child cell j
                int m = (*A)->child[j]->Nf;  // Current number of obs pts
                cobservlist[j][m] = k;        // Store index in list for cell j
                (*A)->child[j]->Nf++;        // Increment number of obs pts
            }
            
            // Recursively distribute the points (if necessary)
            for (int icell=0; icell<8; icell++) {
                if ((*A)->child[icell]->Nf > 0)
                    DistributeObservPoints(&((*A)->child[icell]),observpos,
                                           cobservlist[icell]);
            }
            
            // Free allocated memory for arrays
            for (int icell=0; icell<8; icell++) {
                if (Nf > 1)
                    delete [] cobservlist[icell];
                else
                    delete cobservlist[icell];
            }
        }
    } 
}

/*
 * Function: DistributeSources
 * ----------------------------------------------------------------------------
 * If the specified FMMNode is a leaf node, then store the indices of the
 * sources located in the corresponding leaf cell - otherwise determine which 
 * sources are in each child cell of the cell corresponding to the specified 
 * FMMNode.
 *
 */
void DistributeSources(FMMNode **A, Vector3 *sourcepos, int *sourcelist) {
    // Parameters
    int Ns = (*A)->Ns;  // Number of sources
    
    // If node corresponds to a leaf cell, store the source indices
    if ((*A)->isLeaf) {
        (*A)->sourcelist = new int[Ns];
        int *S = (*A)->sourcelist;
        for (int j=0; j<Ns; j++) 
            S[j] = sourcelist[j];
    }
    
    // Otherwise determine which sources are in each child cell
    else {        
        // Distribute the sources (if necessary)  
        if (Ns > 0) {
            // Create index lists for each child cell
            int *csourcelist[8];
            for (int icell=0; icell<8; icell++)
                csourcelist[icell] = new int[Ns];
            
            // Obtain the center of the cell
            Vector3 center = (*A)->center;
            
            // Determine which child cell each source belongs to
            for (int j=0; j<Ns; j++) {
                int k = sourcelist[j];          // Source index
                Vector3 point = sourcepos[k];   // Location of source
                int i;                          // Index of child cell
                
                // Determine which cell the point is in
                if (point.x < center.x) {
                    if (point.y < center.y) {
                        if (point.z < center.z)
                            i = 0;
                        else
                            i = 1;
                    } else {
                        if (point.z < center.z)
                            i = 2;
                        else
                            i = 3;
                    }
                } else {
                    if (point.y < center.y) {
                        if (point.z < center.z)
                            i = 4;
                        else
                            i = 5;
                    } else {
                        if (point.z < center.z)
                            i = 6;
                        else
                            i = 7;
                    }
                }
                
                // Add the source to the list for child cell i
                int m = (*A)->child[i]->Ns;  // Current number of sources
                csourcelist[i][m] = k;        // Store index in list for cell j
                (*A)->child[i]->Ns++;        // Increment number of sources
            }
            
            // Recursively distribute the points (if necessary)
            for (int icell=0; icell<8; icell++) {
                if ((*A)->child[icell]->Ns > 0)
                    DistributeSources(&((*A)->child[icell]),sourcepos,
                                      csourcelist[icell]);
            }
            
            // Free allocated memory for arrays
            for (int icell=0; icell<8; icell++) {
                if (Ns > 1)
                    delete [] csourcelist[icell];
                else
                    delete csourcelist[icell];
            }
        }
    } 
}

/*
 * Function: BuildInteractionList
 * ----------------------------------------------------------------------------
 * Builds the interaction list for the child nodes of the specified FMMNode.
 * (only works for square cells)
 *
 */
void BuildInteractionList(FMMNode **A) {
    // Number of neighbor cells
    int nneigh = (*A)->nneigh;
    
    /* Sets the cutoff between near and far to be the length of an edge of the
     * cell (this is equivalent to a one cell buffer) - assumes square cell */
    Vector3 dim = (*A)->dim;        // Cell dimensions
    varType cutoff2 = dim.x*dim.x;  // Squared cutoff
    
    /* Only construct interaction lists for child nodes containing
     * observation point */
    FMMNode *Achild[8];
    int nchild = 0;      // Number of child nodes needing interaction lists
    for (int icell=0; icell<8; icell++) {
        if ((*A)->child[icell]->Nf > 0)
            Achild[nchild++] = (*A)->child[icell];
    }
    
    /* 
     * Finds all neighbors that are too close for the far field 
     * approximation and stores them in the neighbors array - 
     * also finds the neighboring cells that are sufficiently 
     * far away and stores them in interaction array
     */
    for (int i=0; i<nneigh; i++) {
        FMMNode *B = (*A)->neighbors[i];        // Neighbor of A
        Vector3 shift = (*A)->shiftneigh[i];    // Periodic shift of B
        for (int icell=0; icell<8; icell++) {   // Loop over children of B
            if (B->child[icell]->Ns > 0) {      // Ignore empty cells
                // Center of child of B
                Vector3 center1 = B->child[icell]->center + shift;
                
                // Loop over all children of A needing interaction lists 
                for (int jcell=0; jcell<nchild; jcell++) { 
                    FMMNode *C = Achild[jcell];  // Child of A
                    int nneigh = C->nneigh;      // No. of neighboring cells
                    int ninter = C->ninter;      // No. of interacting cells
                    Vector3 center2 = C->center; // Center of child of A
                    
                    // Computes the squared distance between nodes B and C
                    Vector3 diff = center1 - center2;
                    varType dist2 = diff.length2();
                    
                    // If within the cutoff, B is a neighbor of C
                    if (dist2 < cutoff2) {
                        C->neighbors[nneigh] = B->child[icell];
                        C->shiftneigh[nneigh] = shift;
                        C->nneigh++;
                    } 
                    
                    // Otherwise B is an interacting node of C
                    else {
                        C->interaction[ninter] = B->child[icell];
                        C->shiftinter[ninter] = shift;
                        C->ninter++;
                    }
                }
            }
        }
    }
    
    // Recursively build the interaction lists
    for (int icell=0; icell<8; icell++) {
        // Only call routine for non-empty non-leaf cells
        if ((*A)->child[icell]->Nf > 0 && !(*A)->child[icell]->isLeaf)
            BuildInteractionList(&((*A)->child[icell]));
    }
}

/*
 * Function: FMMCompute
 * ----------------------------------------------------------------------------
 * Computes the field using bbFMM.
 *
 */
void FMMCompute(FMMParam *param, Kernel *kernel, FMMNode **tree, varType **U,
                varType **V, varType **K, Vector3 *observpos,
                Vector3 *sourcepos, varType *qf, varType *qs, varType *field) {
    // Parameters
    int n = param->n;           // Number of Chebyshev nodes in each direction
    int odof = kernel->odof;    // Observation degrees of freedom
    int sdof = kernel->sdof;    // Source degrees of freedom
    int sdofn3 = sdof*n*n*n;
    int *vcutoff = param->vcutoff;  // Number of source singular vectors
    
    // Lookup table for the index of interaction cell (for use in M2L operation)
    int Ktable[343]; 
    
    // Weights for mapping children Chebyshev nodes to parent nodes (1-D)
    varType *c2cweights = new varType[2*n*n];
    
    // Value of Chebyshev polynomials T_0,...,T_{n-1} at Chebyshev nodes of T_n
    varType *Tk = new varType[n*n];
    
    /* Set up the lookup table, compute the Chebyshev mapping weights, and
     * evaluate the Chebyshev polynomials at the Chebyshev nodes */
    PrecomputeArrays(n,Ktable,Tk,c2cweights);
    
    // Begin timing
    timeType t0 = Timer();
    
    // Upward pass (M2M operation)
    UpwardPass(param,kernel,tree,sourcepos,qs,Tk,c2cweights);
    
    timeType t1 = Timer();
    
    // Computes all of the cell interactions (M2L operation)
    for (int idof=0; idof<odof; idof++) {
        // Begin by compressing the multipole coefficients
        CompressMultCoeff(tree,V[idof],sdofn3,vcutoff[idof]);
    
        // Compute the local coefficients for observation dof idof
        InteractionPass(param,kernel,tree,U[idof],K[idof],Ktable,idof);
    }
    
    timeType t2 = Timer();
    
    // Downward pass (L2L operation)
    DownwardPass(param,kernel,tree,observpos,qf,Tk,c2cweights,field);
    
    timeType t3 = Timer();
    
    // Direct interactions
    DirectInteractions(param,kernel,tree,observpos,sourcepos,qf,qs,field);
    
    timeType t4 = Timer();
    
    // Output the computation time for each pass
    /*
    cout << "Upward:      " << (t1-t0)/(t4-t0) << " secs" << endl;
    cout << "Interaction: " << (t2-t1)/(t4-t0) << " secs" << endl;
    cout << "Downward:    " << (t3-t2)/(t4-t0) << " secs" << endl;
    cout << "Direct:      " << (t4-t3)/(t4-t0) << " secs" << endl;
    //*/
    
    // Free allocated memory for arrays
    delete [] c2cweights;
    delete [] Tk;
}

/*
 * Function: PrecomputeArrays
 * ----------------------------------------------------------------------------
 * Set up the lookup table for transfer vectors, evaluate the Chebyshev 
 * polynomials T_0,...,T_{n-1} at the Chebyshev nodes of T_n, and compute the 
 * weights for mapping children Chebyshev nodes to parent nodes.
 *
 */
void PrecomputeArrays(int n, int *Ktable, varType *Tk, varType *c2cweights) {
    // Parameters
    int n3 = n*n*n;          
    int Nc = 2*n3;            // Number of child Chebyshev nodes
    varType pi = M_PI;
        
    // Initialize lookup table
    for (int i=0; i<343; i++)
        Ktable[i] = -1;
    
    // Create lookup table
    int ncell = 0;      // Counter for each of the 7^3 = 343 cells
    int ninteract = 0;  // Counter for interacting cells (excludes neighbors)
    for (int ix=-3; ix<4; ix++) {
        for (int iy=-3; iy<4; iy++) {
            for (int iz=-3; iz<4; iz++) {
                if (abs(ix) > 1 || abs(iy) > 1 || abs(iz) > 1)
                    Ktable[ncell] = ninteract++;
                ncell++;
            }
        }
    }	
    
    // Compute the n Chebyshev nodes of T_n(x)
    varType *cnodes = new varType[n];   // Positions of Chebyshev nodes
    for (int i=0; i<n; i++)
        cnodes[i] = cos(pi*((varType)i+0.5)/(varType)n);
    
    // Evaluate the Chebyshev polynomials of degree 0 to n-1 at the nodes
    varType *Tkvec = new varType[n];    // Values of polynomials at a node
    int index = 0;                      // Index for array Tk
    for (int i=0; i<n; i++) {
        EvaluateTk(cnodes[i],n,Tkvec);  // Evaluate polynomials at cnodes[i]
        for (int j=0; j<n; j++)
            Tk[index++] = Tkvec[j];
    }
    
    /* Map Chebyshev nodes from two children cells to the parent domain
     * (same x- and y-coordinates; in z, child intervals are [-1,0] and [0,1]
     * and parent interval is [-1,1]) */
    Vector3 *cnodepos = new Vector3[Nc];    // Array of node locations
    index = 0;              // Index for array of children node locations
    for (int i=0; i<2; i++) {
        // Select the shift vector for the specific child cell
        Vector3 shift = Vector3(-1.0,-1.0,-1.0);
        if (i == 1)
            shift.z = 1.0;
        
        // Compute the location of the children nodes in the parent domain
        for (int ix=0; ix<n; ix++) {
            for (int iy=0; iy<n; iy++) {
                for (int iz=0; iz<n; iz++)
                    cnodepos[index++] = 0.5*(Vector3(cnodes[ix],cnodes[iy],
                                                     cnodes[iz]) + shift);
            }
        }
    }
    
    /* Evaluate the interpolating functions Sn for the parent nodes at the 
     * children nodes */
    Vector3 *Sn = new Vector3[n*Nc];
    EvaluateInterpFn(cnodepos,Nc,n,Tk,Sn);
    
    // Extract out the Chebyshev mapping weights
    index = 0;      // Index for c2cweights
    for (int i=0; i<n; i++) {
        int k = i*Nc;
        for (int iz=0; iz<n; iz++)
            c2cweights[index++] = Sn[k++].z;
    }
    for (int i=0; i<n; i++) {
        int k = i*Nc + n3;
        for (int iz=0; iz<n; iz++)
            c2cweights[index++] = Sn[k++].z;
    }
        
    // Free allocated memory for arrays
    delete [] cnodes;
    delete [] Tkvec;
    delete [] cnodepos;
    delete [] Sn;
}

/*
 * Function: EvaluateTk
 * ----------------------------------------------------------------------------
 * Evaluates T_k(x), the first-kind Chebyshev polynomial of degree k,  
 * for k between 0 and n-1 inclusive.
 *
 */
void EvaluateTk(varType x, int n, varType *Tkvec) {
    Tkvec[0] = 1;
    Tkvec[1] = x;
    
    // Use the recurrence relation of Chebyshev polynomials
    for (int k=2; k<n; k++)
        Tkvec[k] = 2.0*x*Tkvec[k-1] - Tkvec[k-2];
}

/*
 * Function: EvaluateInterpFn
 * ----------------------------------------------------------------------------
 * Evaluates the interpolating function S_n(x_m,x_i) for all Chebyshev node-
 * point pairs using Clenshaw's recurrence relation.
 *
 */
void EvaluateInterpFn(Vector3 *pos, int N, int n, varType *Tk, Vector3 *Sn) {
    // Parameters
    varType prefac = 2.0/(varType)n;    // Pre-factor
    
    // Allocate memory for arrays
    varType *Tkvec = new varType[n];    // Vector of values from Tk
    varType *d = new varType[n+2];      // A (n+2)-vector for storage
    
    int tindex = 0;     // Index for array Tk
    int sindex = 0;     // Index for array Sn
    for (int m=0; m<n; m++) {
        // Extract T_k for the Chebyshev node x_m
        for (int j=0; j<n; j++) 
            Tkvec[j] = Tk[tindex++];
        
        // Compute S_n for each direction independently using Clenshaw
        for (int i=0; i<N; i++) {
            varType x = pos[i].x;
            d[n] = d[n+1] = 0.0;
            for (int j=n-1; j>0; j--)
                d[j] = 2.0*x*d[j+1] - d[j+2] + Tkvec[j];
            Sn[sindex].x = prefac*(x*d[1] - d[2] + 0.5*Tkvec[0]);
            
            x = pos[i].y;
            d[n] = d[n+1] = 0.0;
            for (int j=n-1; j>0; j--)
                d[j] = 2.0*x*d[j+1] - d[j+2] + Tkvec[j];
            Sn[sindex].y = prefac*(x*d[1] - d[2] + 0.5*Tkvec[0]);
            
            x = pos[i].z;
            d[n] = d[n+1] = 0.0;
            for (int j=n-1; j>0; j--)
                d[j] = 2.0*x*d[j+1] - d[j+2] + Tkvec[j];
            Sn[sindex++].z = prefac*(x*d[1] - d[2] + 0.5*Tkvec[0]);
        }
    }
    
    // Free allocated memory for arrays
    delete [] Tkvec;
    delete [] d;
}

/*
 * Function: UpwardPass
 * ----------------------------------------------------------------------------
 * Gathers the coefficients from the children cells and determines the multipole 
 * coefficients for the Chebyshev nodes of the parent cell.
 * (upward pass of bbFMM - M2M operation)
 */
void UpwardPass(FMMParam *param, Kernel *kernel, FMMNode **A, 
                Vector3 *sourcepos, varType *qs, varType *Tk, 
                varType *c2cweights) {
    // Parameters
    int n    = param->n;        // Number of Chebyshev nodes
    int n3   = n*n*n;
    int sdof = kernel->sdof;    // Source degrees of freedom
    int sdofn3 = sdof*n3;
        
    /* If node A is a leaf node compute the multipole coefficients for A from
     * the sources located in the corresponding cell */
    if ((*A)->isLeaf) {
        // Parameters and initialization
        int Ns = (*A)->Ns;                  // Number of sources in cell
        int *sourcelist  = (*A)->sourcelist;// Indices of sources in cell
        Vector3 center   = (*A)->center;    // Center of cell
        Vector3 dim      = (*A)->dim;       // Cell dimensions 
        Vector3 halfdim  = 0.5*dim;         // Child cell dimensions
        Vector3 ihalfdim = 1.0/halfdim;     // Inverse dimensions
        
        // Source locations transformed to [-1,1] x [-1,1] x [-1,1] domain
        Vector3 *transpos = new Vector3[Ns];
        
        // Interpolating functions
        Vector3 *Sn = new Vector3[n*Ns];
        
        // Map all of the sources to the box ([-1 1])^3
        for (int j=0; j<Ns; j++) {
            int k = sourcelist[j];  // Global index for j-th source
            transpos[j].x = ihalfdim.x*(sourcepos[k].x - center.x);
            transpos[j].y = ihalfdim.y*(sourcepos[k].y - center.y);
            transpos[j].z = ihalfdim.z*(sourcepos[k].z - center.z);
        }
        
        /* Evaluate the interpolating functions Sn for the Chebyshev nodes 
         * at the source locations */
        EvaluateInterpFn(transpos,Ns,n,Tk,Sn);
        
        // Compute the multipole coefficients
        (*A)->multcoeff = new varType[sdofn3];  // Array for multipole coeff
        varType *M = (*A)->multcoeff;           // Pointer to this array
        int index = 0;                          // Index for M
        for (int jdof=0; jdof<sdof; jdof++) {
            varType *qsdof = &qs[jdof];         // First source strength
            for (int ix=0; ix<n; ix++) {
                Vector3 *Snx = &Sn[ix*Ns];
                for (int iy=0; iy<n; iy++) {
                    Vector3 *Sny = &Sn[iy*Ns];
                    for (int iz=0; iz<n; iz++) {
                        Vector3 *Snz = &Sn[iz*Ns];
                        varType sum = 0.0;
                        for (int j=0; j<Ns; j++)
                            sum += qsdof[sdof*sourcelist[j]]*
                            Snx[j].x*Sny[j].y*Snz[j].z;
                        M[index++] = sum;
                    }
                }
            }
        }
        
        // Free allocated memory for arrays
        delete [] transpos;
        delete [] Sn;
    
    }        
    
    /* Otherwise use the multipole coefficients from all children cells to
     * compute the multipole coefficients for the parent cell */
    else {        
        // Initialization
        int zindex[8], yindex[4], xindex[2]; // Indicate non-zero child cells
        varType *Mzin[8],*Myin[4],*Mxin[2];  // Pointers to array entries
        int xcount = 0;
        int ycount = 0;
        int zcount = 0;
        for (int icell=0; icell<2; icell++)
            xindex[icell] = -1;
        for (int icell=0; icell<4; icell++)
            yindex[icell] = -1;
        for (int icell=0; icell<8; icell++)
            zindex[icell] = -1;
        
        // Allocate memory for arrays used in mapping the multipole coefficients
        varType *My = new varType[2*sdofn3];
        varType *Mz = new varType[4*sdofn3];
        memset((void *)My,0.0,2*sdofn3*sizeof(varType));
        memset((void *)Mz,0.0,4*sdofn3*sizeof(varType));
        
        // Allocate memory for the array of multipole coefficients and set to 0
        (*A)->multcoeff = new varType[sdofn3];
        varType *M = (*A)->multcoeff;   // Pointer to array
        memset((void *)M,0.0,sdofn3*sizeof(varType));
        
        // Determine which children cells contain sources
        for (int icell=0; icell<8; icell++) {
            if ((*A)->child[icell]->Ns > 0) { 
                zindex[zcount++] = icell;
                if (ycount == 0 || yindex[ycount-1] != icell/2)
                    yindex[ycount++] = icell/2;
                
                if (xcount == 0 || xindex[xcount-1] != icell/4)
                    xindex[xcount++] = icell/4;
                
                // Recursively compute the multipole coefficients
                UpwardPass(param,kernel,&((*A)->child[icell]),sourcepos,qs,Tk,
                           c2cweights);
            }
        }
        
        // Initialize pointers
        for (int icell=0; icell<8; icell++)
            Mzin[icell] = (*A)->child[icell]->multcoeff;
        for (int icell=0; icell<4; icell++)
            Myin[icell] = &Mz[icell*sdofn3];
        for (int icell=0; icell<2; icell++)
            Mxin[icell] = &My[icell*sdofn3];
        
        // Gather the child multipole coefficients along the z-component
        ChebyshevMapUp(Mzin,zindex,zcount,c2cweights,n,sdof,Myin);
        
        // Gather the child multipole coefficients along the y-component
        ChebyshevMapUp(Myin,yindex,ycount,c2cweights,n,sdof,Mxin);
        
        // Gather the child multipole coefficients along the x-component
        ChebyshevMapUp(Mxin,xindex,xcount,c2cweights,n,sdof,&M);
        
        // Free allocated memory for arrays
        delete [] My;
        delete [] Mz;
    }
}

/* 
 * Function: ChebyshevMapUp
 * ----------------------------------------------------------------------------
 * Maps the multipole coefficients at child Chebyshev nodes to nodes in a 
 * parent cell along one dimension.
 *
 */
void ChebyshevMapUp(varType **in, int *cellindex, int ncells, 
                    varType *c2cweights, int n, int sdof, varType **out) {
    // Parameters
    int n2 = n*n;
    int n3 = n2*n;
    int incr = 1;   // Stride for BLAS dot product routine
    
    // For each child cell with nonzero coefficients, map to parent domain
    for (int icell=0; icell<ncells; icell++) {
        int j = cellindex[icell];   // Cell index
        int k = (int)(j/2);         // Destination index
        
        varType *y = in[j];     // Start of child multipole coefficients
        varType *x = out[k];    // Start of parent multipole coefficients
        
        // Select which set of Chebyshev mapping weights to use
        int start;     // Starting index for c2cweights
        if (j%2 == 0)
            start = 0;
        else
            start = n2;
        
        int xindex = 0; // Index for array of parent multipole coefficients
        for (int idof=0; idof<sdof; idof++) {
            int cindex = start;   // Index for c2cweights
            for (int l=0; l<n; l++) {
                int yindex = idof*n3;  // Index for array of child coefficients
                for (int m=0; m<n2; m++) {
                    #ifdef SINGLE
                    x[xindex++] += sdot_(&n,&y[yindex],&incr,
                                         &c2cweights[cindex],&incr);
                    #else
                    x[xindex++] += ddot_(&n,&y[yindex],&incr,
                                         &c2cweights[cindex],&incr);
                    #endif
                    yindex += n;
                }
                cindex += n;
            }
        }
    }
}

/*
 * Function: CompressMultCoeff
 * ----------------------------------------------------------------------------
 * Compresses the multipole coefficients of the specified FMMNode in
 * preparation for the interaction pass (M2L operation).
 *
 */
void CompressMultCoeff(FMMNode **A, varType *V, int Vrows, int Vcols) {
    // Parameters for BLAS matrix-vector multiplication
    char *trans   = "t";    // Indicator for matrix transpose
    varType alpha = 1.0;    // Coefficient for matrix-vector product
    varType beta  = 0.0;    // No addition to previous value in storage array
    int incr      = 1;      // Stride for storing matrix-vector product
    
    // Free memory for compressed multipole coefficients (if necessary)
    if ((*A)->cmultcoeff != NULL)
        delete [] (*A)->cmultcoeff;
    
    // Allocated memory for coefficients
    (*A)->cmultcoeff = new varType[Vcols];
    
    varType *M = (*A)->multcoeff;   // Pointer to multipole coefficients
    varType *C = (*A)->cmultcoeff;  // Pointer to compressed coefficients
    
    // Compress the multipole coefficients
    #ifdef SINGLE
    sgemv_(trans,&Vrows,&Vcols,&alpha,V,&Vrows,M,&incr,&beta,C,&incr);
    #else
    dgemv_(trans,&Vrows,&Vcols,&alpha,V,&Vrows,M,&incr,&beta,C,&incr);
    #endif
        
    // Continue compression recursively
    if (!(*A)->isLeaf) {
        for (int icell=0; icell<8; icell++) {
            // Only need to compress if the multipole coefficients are non-zero
            if ((*A)->child[icell]->Ns > 0)
                CompressMultCoeff(&((*A)->child[icell]),V,Vrows,Vcols);
        }
    }
}

/*
 * Function: InteractionPass
 * ----------------------------------------------------------------------------
 * At each level of the FMM tree the interaction between well-separated cells 
 * of observation and source Chebyshev nodes is computed. 
 * (interaction pass of bbFMM - M2L operation)
 * (only works for square cells due to homogeneity implementation)
 *
 */
void InteractionPass(FMMParam *param, Kernel *kernel, FMMNode **A, varType *U,
                     varType *K, int *Ktable, int idof) {
    // Parameters
    int n         = param->n;               // Number of Chebyshev nodes
    int n3        = n*n*n;
    int odof      = kernel->odof;           // Observation degrees of freedom
    int ucutoff   = param->ucutoff[idof];   // Number of obs singular vectors
    int vcutoff   = param->vcutoff[idof];   // Number of source singular vecs
    int uvdof     = ucutoff*vcutoff;
    int odofn3    = odof*n3;      
    Vector3 dim   = (*A)->dim;              // Cell dimensions
    Vector3 idim  = 1.0/dim;                // Inverse cell dimensions
    
    // Parameters for BLAS matrix-vector multiplication
    char *trans   = "n";    // Indicator for matrix transpose
    varType alpha = 1.0;    // Coefficient for matrix-vector product
    varType beta  = 0.0;    // No addition to previous value in storage array
    int incr      = 1;      // Stride for storing matrix-vector product
    
    // Compute the scaling factor for homogeneous kernels
    varType homogen = kernel->homogen[idof];// Order of homogeneity
    varType scale = pow(dim.x,homogen);     // Scaling factor for SVD
                
    // Allocate memory for storing compressed local coefficients and set to zero
    varType *CL = new varType[ucutoff];
    memset((void *)CL,0.0,ucutoff*sizeof(varType));
    
    // Allocate memory for local coefficients (if necessary)
    if ((*A)->localcoeff == NULL)
        (*A)->localcoeff = new varType[odofn3];
    
    // Initialize pointer to the local coefficients of observation dof idof
    varType *L = &((*A)->localcoeff[idof*n3]);
    
    // Obtain the center of the observation cell
    Vector3 ocenter = (*A)->center;
    
    // Obtain the number of interaction cells
    int ninter = (*A)->ninter;
    
    // Compute the local coefficients due to all members of the interaction list
    for (int i=0; i<ninter; i++) {
        FMMNode *B = (*A)->interaction[i];  // Node in interaction list
        
        // Initialize pointer to the compressed multipole coefficients of B
        varType *CM = B->cmultcoeff;
        
        // Obtain the center of the source cell
        Vector3 scenter = B->center + (*A)->shiftinter[i];
        
        // Determine the corresponding index in the lookup table
        int k1 = (int)(idim.x*(scenter.x-ocenter.x)) + 3;
        int k2 = (int)(idim.y*(scenter.y-ocenter.y)) + 3;
        int k3 = (int)(idim.z*(scenter.z-ocenter.z)) + 3;
        int ninteract = Ktable[49*k1+7*k2+k3];
        int index = ninteract*uvdof;  // Index for compressed kernel matrix
        
        // Compute the contribution compressed local coefficients and add
        #ifdef SINGLE
        sgemv_(trans,&ucutoff,&vcutoff,&alpha,&K[index],&ucutoff,CM,&incr,
               &alpha,CL,&incr);
        #else
        dgemv_(trans,&ucutoff,&vcutoff,&alpha,&K[index],&ucutoff,CM,&incr,
               &alpha,CL,&incr);
        #endif
    }
    
    // Compute the local coefficients by uncompressing with U
    #ifdef SINGLE
    sgemv_(trans,&n3,&ucutoff,&scale,U,&n3,CL,&incr,&beta,L,&incr);
    #else
    dgemv_(trans,&n3,&ucutoff,&scale,U,&n3,CL,&incr,&beta,L,&incr);
    #endif
    
    // Recursively compute the kernel interactions for all children cells
    if (!(*A)->isLeaf) {
        for (int icell=0; icell<8; icell++) {
            if ((*A)->child[icell]->Nf > 0) 
                InteractionPass(param,kernel,&((*A)->child[icell]),U,K,
                                Ktable,idof);
        }
    }
    
    // Free allocated memory for arrays
    delete [] CL;
}

/*
 * Function: DownwardPass
 * ----------------------------------------------------------------------------
 * Distributes the local coefficients from the parent cell to the children cells 
 * using Chebyshev interpolation. (downward pass of bbFMM - L2L operation)
 *
 */
void DownwardPass(FMMParam *param, Kernel *kernel, FMMNode **A, 
                  Vector3 *observpos, varType *qf, varType *Tk,
                  varType *c2cweights, varType *field) {
    // Parameters
    int n    = param->n;        // Number of Chebyshev nodes
    int n3   = n*n*n;
    int odof = kernel->odof;    // Observation degrees of freedom
    int odofn3 = odof*n3; 
    
    /* If node A is a leaf node compute the field value at the observation 
     * points located in the corresponding cell by interpolating the local
     * coefficients */
    if ((*A)->isLeaf) {
        // Parameters and initialization
        int Nf           = (*A)->Nf;        // Number of obs points in cell
        int *observlist  = (*A)->observlist;// Indices of obs points in cell
        Vector3 center   = (*A)->center;    // Center of cell
        Vector3 dim      = (*A)->dim;       // Cell dimensions 
        Vector3 halfdim  = 0.5*dim;         // Child cell dimensions
        Vector3 ihalfdim = 1.0/halfdim;     // Inverse of child cell dimensions
        
        // Obs point locations transformed to [-1,1] x [-1,1] x [-1,1] domain
        Vector3 *transpos = new Vector3[Nf];
        
        // Interpolating functions
        Vector3 *Sn = new Vector3[n*Nf];
        
        // Map all of the observation points to the box ([-1 1])^3
        for (int i=0; i<Nf; i++) {
            int k = observlist[i];  // Global index for i-th obs point
            transpos[i].x = ihalfdim.x*(observpos[k].x - center.x);
            transpos[i].y = ihalfdim.y*(observpos[k].y - center.y);
            transpos[i].z = ihalfdim.z*(observpos[k].z - center.z);
        }
        
        /* Evaluate the interpolating functions Sn for the Chebyshev nodes at 
         * the source locations */
        EvaluateInterpFn(transpos,Nf,n,Tk,Sn);
        
        // Compute the field values at the observation points
        varType *L = (*A)->localcoeff;          // Array of local coefficients
        for (int idof=0; idof<odof; idof++) {
            varType *qfdof = &qf[idof];         // First field coeff
            varType *fielddof = &field[idof];   // First field value
            for (int i=0; i<Nf; i++) {
                Vector3 *Sni = &Sn[i];
                varType sum = 0.0;
                int lindex = idof*n3;           // Index for local coefficients
                int xindex = 0;
                for (int ix=0; ix<n; ix++) {
                    int yindex = 0;
                    varType Snix = Sni[xindex].x;
                    for (int iy=0; iy<n; iy++) {
                        int zindex = 0;
                        varType Snixy = Snix*Sni[yindex].y;
                        for (int iz=0; iz<n; iz++) {
                            sum += L[lindex++]*Snixy*Sni[zindex].z;
                            zindex += Nf;
                        }
                        yindex += Nf;
                    }
                    xindex += Nf;
                }
                
                // Store the far-field contribution
                int k = odof*observlist[i];
                fielddof[k] = qfdof[k]*sum;
            }
        }
        
        // Free allocated memory for arrays
        delete [] transpos;
        delete [] Sn;
    }
    
    /* Otherwise use the local coefficients from the parent cells to 
     * interpolate the local coefficients for the children cells */
    else {
        // Initialization
        int zindex[8], yindex[4], xindex[2];    // Indicate non-zero child cells
        varType *Lzout[8],*Lyout[4],*Lxout[2];  // Pointers to array entries
        int xcount = 0;
        int ycount = 0;
        int zcount = 0;
        for (int icell=0; icell<2; icell++)
            xindex[icell] = -1;
        for (int icell=0; icell<4; icell++)
            yindex[icell] = -1;
        for (int icell=0; icell<8; icell++)
            zindex[icell] = -1;
        
        // Allocate memory for arrays used in mapping the local coefficients
        varType *Lx = new varType[2*odofn3];
        varType *Ly = new varType[4*odofn3];
        memset((void *)Lx,0.0,2*odofn3*sizeof(varType));
        memset((void *)Ly,0.0,4*odofn3*sizeof(varType));
        
        // Determine which children cells contain observation points
        for (int icell=0; icell<8; icell++) {
            if ((*A)->child[icell]->Nf > 0) { 
                zindex[zcount++] = icell;
                if (ycount == 0 || yindex[ycount-1] != icell/2)
                    yindex[ycount++] = icell/2;
                
                if (xcount == 0 || xindex[xcount-1] != icell/4)
                    xindex[xcount++] = icell/4;
            }
        }
        
        // Initialize pointers
        for (int icell=0; icell<8; icell++)
            Lzout[icell] = (*A)->child[icell]->localcoeff;
        for (int icell=0; icell<4; icell++)
            Lyout[icell] = &Ly[icell*odofn3];
        for (int icell=0; icell<2; icell++)
            Lxout[icell] = &Lx[icell*odofn3];
        
        // Array of local coefficients
        varType *L = (*A)->localcoeff;
        
        // Interpolate the parent field along the x-component
        ChebyshevMapDown(&L,xindex,xcount,c2cweights,n,odof,Lxout);
        
        // Interpolate the parent field along the y-component
        ChebyshevMapDown(Lxout,yindex,ycount,c2cweights,n,odof,Lyout);
        
        // Interpolate the parent field along the z-component
        ChebyshevMapDown(Lyout,zindex,zcount,c2cweights,n,odof,Lzout);
        
        // Recursively compute the local coefficients
        for (int icell=0; icell<8; icell++) {
            if ((*A)->child[icell]->Nf > 0)
                DownwardPass(param,kernel,&((*A)->child[icell]),observpos,qf,Tk,
                             c2cweights,field);
        }
        
        // Free allocated memory for arrays
        delete [] Lx;
        delete [] Ly;
    } 
}

/* 
 * Function: ChebyshevMapDown
 * ----------------------------------------------------------------------------
 * Maps the local coefficients at parent Chebyshev nodes to nodes in a child 
 * cell along one dimension.
 *
 */
void ChebyshevMapDown(varType **in, int *cellindex, int ncells, 
                      varType *c2cweights, int n, int odof, varType **out) {
    // Parameters
    int n2 = n*n;
    int n3 = n2*n;
    
    // For each child cell with nonzero coefficients, map to parent domain
    for (int icell=0; icell<ncells; icell++) {
        int j = cellindex[icell];   // Destination index
        int k = (int)(j/2);         // Cell index
        
        varType *y = in[k];         // Start of parent local coefficients
        varType *x = out[j];        // Start of child local coefficients
        
        // Select which set of Chebyshev mapping weights to use
        int start;      // Starting index of c2cweights
        if (j%2 == 0)
            start = 0;
        else
            start = n2;
        
        int xindex = 0; // Index for array of child local coefficients 
        for (int idof=0; idof<odof; idof++) {
            int yindex = idof*n3;       // Array index for parent coefficients
            for (int l=0; l<n2; l++) {
                int cindex = start;     // Index for c2cweights
                for (int m=0; m<n; m++) {
                    #ifdef SINGLE
                    x[xindex++] += sdot_(&n,&y[yindex],&n2,
                                         &c2weights[cindex],&n);
                    #else
                    x[xindex++] += ddot_(&n,&y[yindex],&n2,
                                         &c2cweights[cindex],&n);
                    #endif
                    cindex++;
                }
                yindex++;
            }
        }
    }
}

/* 
 * Function: DirectInteractions
 * ----------------------------------------------------------------------------
 * Computes the interactions between observation points in a leaf cell and 
 * sources in neighboring cells directly.
 *
 */
void DirectInteractions(FMMParam *param, Kernel *kernel, FMMNode **A, 
                        Vector3 *observpos, Vector3 *sourcepos, varType *qf,
                        varType *qs, varType *field) {
    /* Compute the interactions between observation points in a leaf cell and
     * sources in neighboring cells directly */
    if ((*A)->isLeaf) {
        // Parameters
        int odof = kernel->odof;    // Observation degrees of freedom
        int sdof = kernel->sdof;    // Source degrees of freedom
        int nneigh = (*A)->nneigh;  // Number of neighboring cells
        int Nf = (*A)->Nf;          // Number of observation points in cell
        
        // Parameters for BLAS matrix-vector multiplication
        char *trans   = "n";    // Indicator for matrix transpose
        varType alpha = 1.0;    // Coefficient for matrix-vector product
        varType beta  = 1.0;    // Add to previous value in storage array
        int incr      = 1;      // Stride for storing matrix-vector product
        
        // Array of indices of observation points
        int *observlist = (*A)->observlist;
        
        // Array for storing field values
        varType *fval = new varType[odof*Nf];
        memset((void *)fval,0.0,odof*Nf*sizeof(varType));
        
        // Array for kernel values
        varType *kval = new varType[odof*sdof];
        
        // Loop over all neighboring cells
        for (int m=0; m<nneigh; m++) {
            FMMNode *B = (*A)->neighbors[m];    // The m-th neighbor cell of A
            int *sourcelist = B->sourcelist;    // Array of indices of sources
            int Ns = B->Ns;                     // Number of sources
            Vector3 shift = (*A)->shiftneigh[m];// Periodic shift
            
            // Loop over all observation points
            for (int i=0; i<Nf; i++) {
                int k = observlist[i];  // Global observation point index
                
                // Loop over all sources
                for (int j=0; j<Ns; j++) {
                    int l = sourcelist[j];  // Global source index
                    
                    // Compute interaction directly
                    kernel->evaluate_exclude(observpos[k],sourcepos[l],kval);
                    #ifdef SINGLE
                    sgemv_(trans,&odof,&sdof,&alpha,kval,&odof,&qs[sdof*l],
                           &incr,&beta,&fval[odof*i],&incr);
                    #else
                    dgemv_(trans,&odof,&sdof,&alpha,kval,&odof,&qs[sdof*l],
                           &incr,&beta,&fval[odof*i],&incr);
                    #endif
                }
            }
        }
        
        // Add the contribution from direct interactions to far-field
        int findex = 0;        // Index for fval
        for (int i=0; i<Nf; i++) {
            int k = observlist[i];
            varType *fieldk = &field[odof*k];
            varType *qfk = &qf[odof*k];
            for (int idof=0; idof<odof; idof++) {
                fieldk[idof] += qfk[idof]*fval[findex++];
            }
        }
        
        // Free allocated memory
        delete [] fval;
        if (odof*sdof > 1)
            delete [] kval;
        else
            delete kval;
    }
    
    // Otherwise proceed down the FMM tree
    else {
        for (int icell=0; icell<8; icell++) {
            if ((*A)->child[icell]->Nf > 0)
                DirectInteractions(param,kernel,&((*A)->child[icell]),observpos,
                                   sourcepos,qf,qs,field);
        }
    }
}