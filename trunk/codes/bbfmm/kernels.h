/** @file kernels.h
 *  @author Will Fong <willfong@gmail.com>
 *  @version 1.0
 * 
 *  Header file for kernels.cpp - contains the kernel base class, a derived
 *  class for each kernel of interest, and functions for evaluating the kernels.
 */

#ifndef _KERNELS_H
#define _KERNELS_H

#include <string>
#include "common.h"

using namespace std;

/** @brief The base class for all kernels */
class Kernel {
public:
    int odof;           /**< Number of observation degrees of freedom */
    int sdof;           /**< Number of source degrees of freedom */
    varType *homogen;   /**< Array of kernel homogeneity orders */
    varType *value;     /**< Array for storing the result from evaluation */
    string *Kmat;       /**< Array of file names for kernel matrices */
    string *Umat;       /**< Array of file names for observ. singular vectors */
    string *Vmat;       /**< Array of file names for source singular vectors */
    
    /** @brief Function wrapper for evaluating the kernel
     *  @param observpos Location of observation node
     *  @param sourcepos Location of source node
     *  @param[in,out] value Returns the results of the evaluation in a 
     *  odof-by-sdof matrix, stored as a vector in column-major format
     */
    virtual void evaluate(Vector3 observpos, Vector3 sourcepos, 
                          varType *value) = 0;
    
    /** @brief Function wrapper for evaluating the kernel with exclusions
     *  @param observpos Location of observation node
     *  @param sourcepos Location of source node
     *  @param[in,out] value Returns the results of the evaluation in a 
     *  odof-by-sdof matrix, stored as a vector in column-major format
     */
    virtual void evaluate_exclude(Vector3 observpos, Vector3 sourcepos, 
                                  varType *value) = 0;
    
    /** @brief Virtual destructor - needed for handling virtual functions */
    virtual ~Kernel() {};
};

/** @brief Derived class for Laplacian kernel
 *  @see Kernel
 */
class KLaplacian : public Kernel {
public:
    /** @brief Constructor
     *  @param n Number of Chebyshev nodes in each direction
     */
    KLaplacian(int n);
    
    /** @brief Destructor */
    ~KLaplacian();
    
    /** @brief Routine for evaluating the Laplacian kernel
     *  @param observpos Location of observation node
     *  @param sourcepos Location of source node
     *  @param[in,out] value Returns the results of the evaluation in a 
     *  odof-by-sdof matrix, stored as a vector in column-major format
     */
    void evaluate(Vector3 observpos, Vector3 sourcepos, varType *value);
    
    /** @brief Routine for evaluating the Laplacian kernel with exclusions
     *  @param observpos Location of observation node
     *  @param sourcepos Location of source node
     *  @param[in,out] value Returns the results of the evaluation in a 
     *  odof-by-sdof matrix, stored as a vector in column-major format
     */
    void evaluate_exclude(Vector3 observpos, Vector3 sourcepos, varType *value);
};

/** @brief Derived class for Laplacian force kernel
 *  @see Kernel
 */
class KLaplacianForce : public Kernel {
public:
    /** @brief Constructor
     *  @param n Number of Chebyshev nodes in each direction
     */
    KLaplacianForce(const int n);
    
    /** @brief Destructor */
    ~KLaplacianForce();
    
    /** @brief Routine for evaluating the Laplacian force kernel
     *  @param observpos Location of observation node
     *  @param sourcepos Location of source node
     *  @param[in,out] value Returns the results of the evaluation in a 
     *  odof-by-sdof matrix, stored as a vector in column-major format
     */
    void evaluate(Vector3 observpos, Vector3 sourcepos, varType *value);
    
    /** @brief Routine for evaluating the Laplacian force kernel with exclusions
     *  @param observpos Location of observation node
     *  @param sourcepos Location of source node
     *  @param[in,out] value Returns the results of the evaluation in a 
     *  odof-by-sdof matrix, stored as a vector in column-major format
     */
    void evaluate_exclude(Vector3 observpos, Vector3 sourcepos, varType *value);
};

/** @brief Derived class for 2-D gravitational force kernel
 *  @see Kernel
 */
class KGravForce2D : public Kernel {
public:
    /** @brief Constructor
     *  @param n Number of Chebyshev nodes in each direction
     */
    KGravForce2D(const int n);
    
    /** @brief Destructor */
    ~KGravForce2D();
    
    /** @brief Routine for evaluating the 2-D gravitational force kernel
     *  @param observpos Location of observation node
     *  @param sourcepos Location of source node
     *  @param[in,out] value Returns the results of the evaluation in a 
     *  odof-by-sdof matrix, stored as a vector in column-major format
     */
    void evaluate(Vector3 observpos, Vector3 sourcepos, varType *value);
    
    /** @brief Routine for evaluating the 2-D gravitational force kernel with 
     *  exclusions
     *  @param observpos Location of observation node
     *  @param sourcepos Location of source node
     *  @param[in,out] value Returns the results of the evaluation in a 
     *  odof-by-sdof matrix, stored as a vector in column-major format
     */
    void evaluate_exclude(Vector3 observpos, Vector3 sourcepos, varType *value);
};

#endif


