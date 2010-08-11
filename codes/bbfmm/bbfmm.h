/** @file bbfmm.h
 *  @author Will Fong <willfong@gmail.com>
 *  @version 1.0
 * 
 *  Header file for bbfmm.cpp which contains functions used
 *  in the implementation of the black-box fast multiple method (bbFMM).
 */

#ifndef _BBFMM_H
#define _BBFMM_H

#include "common.h"
#include "kernels.h"

/** @brief Stores all of the parameters needed for bbFMM */
class FMMParam {
public:
    int Nf;             /**< Number of observation points */
    int Ns;             /**< Number of sources */
    Vector3 boxcntr;    /**< Center of simulation box */ 
    Vector3 L;          /**< Dimensions of simulation box */
    int n;              /**< Number of Chebyshev nodes in one direction */
    int levels;         /**< Number of levels in FMM tree */
    int *ucutoff;       /**< Number of observation singular vectors to keep */
    int *vcutoff;       /**< Number of source singular vectors to keep */
    
    /** @brief Constructor
     *  @param Nf0 Number of observation points
     *  @param Ns0 Number of sources
     *  @param boxcntr0 Center of simulation box
     *  @param L0 Dimensions of simulation box
     *  @param n0 Number of Chebyshev nodes in one direction
     */
    FMMParam(int Nf0, int Ns0, Vector3 boxcntr0, Vector3 L0, int n0);
    
    /** @brief Destructor */
    ~FMMParam() {};
};

/** @brief A node in the FMM tree - stores cell data associated with the node */
class FMMNode {
public:    
    FMMNode *child[8];          /**< Child cells */
    FMMNode *parent;            /**< Parent cell */
    FMMNode *neighbors[27];     /**< Cells in neighbor list */ 
    FMMNode *interaction[189];  /**< Cells in interaction list */
    Vector3 shiftneigh[27];     /**< PBC shift for neighbor cells */
    Vector3 shiftinter[189];    /**< PBC shift for interacton cells */
    varType *localcoeff;        /**< Local coefficients of cell */
    varType *multcoeff;         /**< Multipole coefficients of cell */
    varType *cmultcoeff;        /**< Compressed multipole coefficients */
    int *observlist;            /**< Indices of observation points in cell */
    int *sourcelist;            /**< Indices of sources in cell */
    
    Vector3 center;             /**< Location of center of cell */
    Vector3 dim;                /**< Dimensions of cell */
    int Nf;                     /**< Number of observation points in cell*/
    int Ns;                     /**< Number of sources in cell*/
    int nneigh;                 /**< Number of neighbor cells */
    int ninter;                 /**< Number of interaction cells */
    bool isLeaf;                /**< Indicates whether node is a leaf node */
    
    /** @brief Constructor
     *  @param center0 Center of cell corresponding to node
     *  @param dim0 Cell dimensions
     */
    FMMNode(Vector3 center0, Vector3 dim0);
    
    /** @brief Destructor */
    ~FMMNode() {};
};

/** @brief Performs an accuracy test for bbFMM.
 *  @param Ns Number of sources
 *  @param boxcntr Center of simulation box
 *  @param L Dimensions of simulation box
 *  @param kernel Kernel of interest
 *  @param n Number of Chebyshev nodes
 *
 *  Performs an accuracy test for the specified kernel and number of Chebyshev
 *  nodes using a set of Ns sources randomly placed according to the uniform 
 *  distribution in a box with the specified dimensions and center location 
 *  - the observation points are taken to be the first 100 sources -
 *  returns the 2-norm error for each observation dof as compared to the
 *  reference solution obtained by direct calculation (for non-periodic systems)
 *  or Ewald (periodic).
 */
void AccuracyTest(int Ns, Vector3 boxcntr, Vector3 L, Kernel *kernel, int n);

/** @brief Computes the reference field by direct calculation (for non-periodic 
 *  systems) or Ewald (periodic).
 *  @param kernel Kernel of interest
 *  @param observpos Array of observation point locations
 *  @param sourcepos Array of source locations
 *  @param qf Array of coefficients for pre-multiplying field values
 *  @param qs Array of source strengths
 *  @param Nf Number of observation points
 *  @param Ns Number of sources
 *  @param[in,out] field Array of field values
 */
void ReferenceField(Kernel *kernel, Vector3 *observpos, Vector3 *sourcepos, 
                    varType *qf, varType *qs, int Nf, int Ns, varType *field);

/** @brief 2-D spiral galaxy simulation
 *  @param N Number of masses
 *  @param n Number of Chebyshev nodes
 *
 *  Simulates the formation of a 2-D spiral galaxy from a uniform elliptic 
 *  distribution of masses - the particles interact according to Newton's law of 
 *  gravitation in two dimensions.
 *
 */
void SpiralGalaxy2D(int N, int n);

/** @brief Prepare for the FMM calculation by building the FMM tree
 *  and pre-computing the SVD if necessary.
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *  @param[in,out] tree bbFMM tree
 */
void FMMSetup(FMMParam *param, Kernel *kernel, FMMNode **tree);

/** @brief Recursively builds the FMM tree with l levels.
 *  @param A A node in the FMM tree
 *  @param l Number of levels in FMM tree
 */
void BuildFMMTree(FMMNode **A, int l);

/** @brief Cleans up after FMM calculation.
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *  @param tree bbFMM tree
 *  @param U Truncated matrix of observation singular vectors
 *  @param V Truncated matrix of source singular vectors
 *  @param K Compressed kernel matrix 
 */
void FMMCleanup(FMMParam *param, Kernel *kernel, FMMNode *tree, varType **U,
                varType **V, varType **K);

/** @brief Frees up memory associated with FMMNode.
 *  @param A A node in the FMM tree
 */
void FreeNode(FMMNode *A);

/** @brief Computes the SVD of the kernel interaction matrix.
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *
 *  Evaluates the kernel for cell-to-cell interactions corresponding to the 
 *  316 multipole-to-local transfer vectors and computes the SVD of this kernel 
 *  matrix.
 */
void ComputeKernelSVD(FMMParam *param, Kernel *kernel);

/** @brief Evaluates the kernel for Chebyshev-Chebyshev interactions between 
 *  a pair of cells and applies the appropriate SVD weighting.
 *  @param kernel Kernel of interest
 *  @param observpos Array of observation node locations
 *  @param sourcepos Array of source node locations
 *  @param n3 Total number of 3-D Chebyshev nodes
 *  @param nodeweights Weights for 3-D Chebyshev nodes 
 *  @param[in,out] Ksub Array of pointers for updating the kernel matrix K 
 */
void EvaluateKernelCell2Cell(Kernel *kernel, Vector3 *observpos, 
                             Vector3 *sourcepos, int n3, 
                             varType *nodeweights, varType **Ksub);

/** @brief Read in the matrix of observation singular vectors U and the matrix 
 *  of source singular vectors V and compute the compressed kernel matrix K.
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *  @param[in,out] U Truncated matrix of observation singular vectors
 *  @param[in,out] V Truncated matrix of source singular vectors
 *  @param[in,out] K Compressed kernel matrix 
 */
void LoadMatrices(FMMParam **param, Kernel *kernel, varType ***U, varType ***V, 
                  varType ***K); 

/** @brief Given the source and observation point locations and the strength of
 *  the sources, the field is computed for the specified kernel using bbFMM.
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *  @param tree bbFMM tree
 *  @param U Truncated matrix of observation singular vectors
 *  @param V Truncated matrix of source singular vectors
 *  @param K Compressed kernel matrix
 *  @param observpos Array of observation point locations
 *  @param sourcepos Array of source locations
 *  @param qf Array of coefficients for pre-multiplying field values
 *  @param qs Array of source strengths
 *  @param[in,out] field Array of field values
 */
void bbFMM(FMMParam *param, Kernel *kernel, FMMNode **tree, varType **U,
           varType **V, varType **K, Vector3 *observpos, Vector3 *sourcepos, 
           varType *qf, varType *qs, varType *field);

/** @brief Clears all of the arrays in the octree in preparation for the FMM
 *  calculation (allows for the use of the same FMM tree).
 *  @param tree bbFMM tree
 */
void FMMClear(FMMNode *tree);

/** @brief Frees up the arrays associated with FMMNode
 *  @param A A node in the FMM tree
 */
void ClearNode(FMMNode *A);

/** @brief Distribute the observation points and sources to the appropriate 
 *  node in the FMM tree and builds the interaction list.
 *  @param param bbFMM parameters
 *  @param tree bbFMM tree
 *  @param observpos Array of observation point locations
 *  @param sourcepos Array of source locations
 */
void FMMDistribute(FMMParam *param, FMMNode **tree, Vector3 *observpos, 
                   Vector3 *sourcepos);

/** @brief Distributes the observation points to the appropriate leaf cell
 *  @param A A node in the FMM tree
 *  @param observpos Array of observation point locations
 *  @param observlist Array of observation point indices
 *
 *  If the specified FMMNode is a leaf node, then store the indices of the
 *  observation points located in the corresponding leaf cell - otherwise
 *  determine which observation points are in each child cell of the cell 
 *  corresponding to the specified FMMNode.
 */
void DistributeObservPoints(FMMNode **A, Vector3 *observpos, int *observlist);

/** @brief Distributes the sources to the appropriate leaf cell
 *  @param A A node in the FMM tree
 *  @param sourcepos Array of source locations
 *  @param sourcelist Array of source indices
 *
 *  If the specified FMMNode is a leaf node, then store the indices of the
 *  sources located in the corresponding leaf cell - otherwise determine which 
 *  sources are in each child cell of the cell corresponding to the specified 
 *  FMMNode.
 */
void DistributeSources(FMMNode **A, Vector3 *sourcepos, int *sourcelist);

/** @brief Builds the interaction list for the child nodes of the specified 
 *  FMMNode. (only works for square cells)
 *  @param A A node in the FMM tree
 */
void BuildInteractionList(FMMNode **A);

/** @brief Computes the field using bbFMM.
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *  @param tree bbFMM tree
 *  @param U Truncated matrix of observation singular vectors
 *  @param V Truncated matrix of source singular vectors
 *  @param K Compressed kernel matrix
 *  @param observpos Array of observation point locations
 *  @param sourcepos Array of source locations
 *  @param qf Array of coefficients for pre-multiplying field values
 *  @param qs Array of source strengths
 *  @param[in,out] field Array of field values
 */
void FMMCompute(FMMParam *param, Kernel *kernel, FMMNode **tree, varType **U,
                varType **V, varType **K, Vector3 *observpos,
                Vector3 *sourcepos, varType *qf, varType *qs, varType *field);

/** @brief Precompute some arrays needed for the bbFMM routine.
 *  @param n Number of Chebyshev nodes
 *  @param[in,out] Ktable Lookup table for transfer vectors
 *  @param[in,out] Tk Values of Chebyshev polynomials at Chebyshev nodes
 *  @param[in,out] c2cweights Weights for mapping Chebyshev nodes
 *
 *  Set up the lookup table for transfer vectors, evaluate the Chebyshev 
 *  polynomials T_0,...,T_{n-1} at the Chebyshev nodes of T_n, and compute 
 *  the weights for mapping children Chebyshev nodes to parent nodes.
 *
 */
void PrecomputeArrays(int n, int *Ktable, varType *Tk, varType *c2cweights);

/** @brief Evaluates T_k(x), the first-kind Chebyshev polynomial of degree k,  
 *  for k between 0 and n-1 inclusive.
 *  @param x x-value at which to evaluate T_k(x)
 *  @param n Evaluates T_k(x) up to order n-1
 *  @param[in,out] Tkvec Array of storing the evaluation results
 */
void EvaluateTk(varType x, int n, varType *Tkvec);

/** @brief Evaluates the interpolating function S_n(x_m,x_i) for all Chebyshev 
 *  node-point pairs using Clenshaw's recurrence relation.
 *  @param pos Location of evaluation points
 *  @param N Number of evaluation points
 *  @param n Number of Chebyshev nodes
 *  @param Tk Values of Chebyshev polynomials at Chebyshev nodes
 *  @param[in,out] Sn Values of interpolating functions at evaluation points
 */
void EvaluateInterpFn(Vector3 *pos, int N, int n, varType *Tk, Vector3 *Sn);

/** @brief Upward pass of bbFMM - M2M operation
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *  @param A A node in the FMM tree
 *  @param sourcepos Array of source locations
 *  @param qs Array of source strengths
 *  @param Tk Values of Chebyshev polynomials at Chebyshev nodes
 *  @param c2cweights Weights for mapping Chebyshev nodes
 *
 *  Gathers the coefficients from the children cells and determines the 
 *  multipole coefficients for the Chebyshev nodes of the parent cell.
 */
void UpwardPass(FMMParam *param, Kernel *kernel, FMMNode **A, 
                Vector3 *sourcepos,varType *qs, varType *Tk,
                varType *c2cweights);

/** @brief Maps the multipole coefficients at child Chebyshev nodes to nodes 
 *  in a parent cell along one dimension.
 *  @param in Multipole coefficients at child Chebyshev nodes
 *  @param cellindex Indices corresponding to cells with nonzero coefficients
 *  @param ncells Number of cells with nonzero coefficients
 *  @param c2cweights Weights for mapping Chebyshev nodes
 *  @param n Number of Chebyshev nodes
 *  @param sdof Number of source degrees of freedom
 *  @param[in,out] out Multipole coefficients at parent Chebyshev nodes
 */
void ChebyshevMapUp(varType **in, int *cellindex, int ncells, 
                    varType *c2cweights, int n, int sdof, varType **out);

/** @brief Compresses the multipole coefficients of the specified FMMNode in
 *  preparation for the interaction pass (M2L operation).
 *  @param A A node in the FMM tree
 *  @param V Truncated matrix of source singular vectors
 *  @param Vrows Number of rows in matrix V
 *  @param Vcols Number of columns in matrix V
 */
void CompressMultCoeff(FMMNode **A, varType *V, int Vrows, int Vcols);

/** @brief Interaction pass of bbFMM - M2L operation
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *  @param A A node in the FMM tree
 *  @param U Truncated matrix of observation singular vectors
 *  @param K Compressed kernel matrix
 *  @param Ktable Lookup table for transfer vectors
 *  @param idof Index of observation dof
 *
 *  At each level of the FMM tree the interaction between well-separated cells 
 *  of observation and source Chebyshev nodes is computed.
 *  (only works for square cells due to homogeneity implementation)
 *
 */
void InteractionPass(FMMParam *param, Kernel *kernel, FMMNode **A, varType *U,
                     varType *K, int *Ktable, int idof);

/** @brief Downward pass of bbFMM - L2L operation
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *  @param A A node in the FMM tree
 *  @param observpos Array of observation point locations
 *  @param qf Array of coefficients for pre-multiplying field values
 *  @param Tk Values of Chebyshev polynomials at Chebyshev nodes
 *  @param c2cweights Weights for mapping Chebyshev nodes
 *  @param[in,out] field Array of field values
 *
 *  Distributes the local coefficients from the parent cell to the children 
 *  cells using Chebyshev interpolation.
 */
void DownwardPass(FMMParam *param, Kernel *kernel, FMMNode **A, 
                  Vector3 *observpos, varType *qf, varType *Tk,
                  varType *c2cweights, varType *field);

/** @brief Maps the local coefficients at parent Chebyshev nodes to nodes in a 
 *  child cell along one dimension.
 *  @param in Local coefficients at parent Chebyshev nodes
 *  @param cellindex Indices corresponding to cells with nonzero coefficients
 *  @param ncells Number of cells with nonzero coefficients
 *  @param c2cweights Weights for mapping Chebyshev nodes
 *  @param n Number of Chebyshev nodes
 *  @param odof Number of observation degrees of freedom
 *  @param[in,out] out Local coefficients at child Chebyshev nodes
 */
void ChebyshevMapDown(varType **in, int *cellindex, int ncells, 
                      varType *c2cweights, int n, int odof, varType **out);

/** @brief Computes the interactions between observation points in a leaf cell 
 *  and sources in neighboring cells directly.
 *  @param param bbFMM parameters
 *  @param kernel Kernel of interest
 *  @param A A node in the FMM tree
 *  @param observpos Array of observation point locations
 *  @param sourcepos Array of source locations
 *  @param qf Array of coefficients for pre-multiplying field values
 *  @param qs Array of source strengths
 *  @param[in,out] field Array of field values
 */
void DirectInteractions(FMMParam *param, Kernel *kernel, FMMNode **A, 
                        Vector3 *observpos, Vector3 *sourcepos, varType *qf,
                        varType *qs, varType *field);
#endif
