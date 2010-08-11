/** @file common.h
 *  @author Will Fong <willfong@gmail.com>
 *  @version 1.0
 * 
 *  Header file containing declarations for data structures
 *  and LAPACK/BLAS routines that are common to different parts of the code.
 */

#ifndef _COMMON_H
#define _COMMON_H
//Modified by Arvind.
#include <stdio.h>
#include <string.h>
//End Modification
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

/** @brief Type for all variables - float is SINGLE is defined, 
 *  otherwise double */
#ifdef SINGLE
typedef float varType;
#else
typedef double varType;
#endif

/** @brief Type for time measurements - default is double */
typedef double timeType;

/** @brief A three-dimensional vector with coordinates x,y,z */
class Vector3 {
public:
    varType x;  /**< x-coordinate */
    varType y;  /**< y-coordinate */
    varType z;  /**< z-coordinate */
    
    /** @brief Default constructor - returns the zero vector */
    Vector3() : x(0.0), y(0.0), z(0.0) {};
    
    /** @brief Constructor given x, y, and z coordinates
     *  @param x0 x-coordinate
     *  @param y0 y-coordinate
     *  @param z0 z-coordinate
     */
    Vector3(varType x0, varType y0, varType z0) : x(x0), y(y0), z(z0) {};
    
    /** @brief Copy constructor (given Vector3)
     *  @param vec 3-dimensional vector
     */
    Vector3(const Vector3 &vec) : x(vec.x), y(vec.y), z(vec.z) {};
    
    /** @brief Destructor */
    ~Vector3() {};
    
    /*************
     * OPERATORS *
     *************/
    /** @brief Assignment operator */
	Vector3& operator=(const Vector3 &vec) {
		x = vec.x;
		y = vec.y;
		z = vec.z;
		return *this;
	}
	
	/** @brief Add-to operator */
	void operator+=(const Vector3 &vec) {
		x += vec.x;
		y += vec.y;
		z += vec.z;
	}
	
	/** @brief Subtract-from operator */
	void operator-=(const Vector3 &vec) {
		x -= vec.x;
		y -= vec.y;
		z -= vec.z;
	}
	
	/** @brief Scalar-multiply operator */
	void operator*=(const varType &fac) {
		x *= fac;
		y *= fac;
		z *= fac;
	}
    
    /** @brief Scalar-divide operator */
	void operator/=(const varType &fac) {
		x /= fac;
		y /= fac;
		z /= fac;
	}
			
	/** @brief Addition of two vectors */
	friend Vector3 operator+(const Vector3 &vec1, const Vector3 &vec2) {
		return Vector3(vec1.x+vec2.x,vec1.y+vec2.y,vec1.z+vec2.z);
	}
	
	/** @brief Subtraction of two vectors */
	friend Vector3 operator-(const Vector3 &vec1, const Vector3 &vec2) {
		return Vector3(vec1.x-vec2.x,vec1.y-vec2.y,vec1.z-vec2.z);
	}

	/** @brief Scalar product */
	friend Vector3 operator*(const varType &fac, const Vector3 &vec) {
		return Vector3(fac*vec.x,fac*vec.y,fac*vec.z);
	}
    
    /** @brief Scalar division */
	friend Vector3 operator/(const varType &fac, const Vector3 &vec) {
		return Vector3(fac/vec.x,fac/vec.y,fac/vec.z);
	}
	
	/** @brief Norm operator */
	inline varType length() const {
		return sqrt(x*x+y*y+z*z);
	}
	
    /** @brief Squared norm operator */
	inline varType length2() const {
		return (x*x+y*y+z*z);
	}
};

/** @brief Uniform random number generator
 *  @param[in] xmin Lower bound
 *  @param[in] xmax Upper bound
 *  @return Random number
 */
inline varType frand(varType xmin, varType xmax) {
    return (xmin+(xmax-xmin)*rand()/(varType)RAND_MAX);
}

/** @brief Returns the time in seconds
 *  @return Time (seconds)
 */
inline timeType Timer() {
    struct timeval timeval_time;
    gettimeofday(&timeval_time,NULL);
    return ((timeType)timeval_time.tv_sec+(timeType)timeval_time.tv_usec*1.e-6);
}

#ifdef SINGLE
/** @brief Declaration for LAPACK's single-precision SVD routine */
extern "C" void sgesvd_(char *jobu, char *jobvt, int *m, int *n, varType *A,
                        int *lda, varType *S, varType *U, int *ldu, varType *VT,
                        int *ldvt, varType *work, int *lwork, int *info);

/** @brief Declaration for BLAS single-precision matrix-matrix multiply */
extern "C" void sgemm_(char *transa, char *transb, int *m, int *n, int *k,
                       varType *alpha, varType *A, int *lda, varType *B,
                       int *ldb, varType *beta, varType *C, int *ldc);

/** @brief Declaration for BLAS single-precision matrix-vector multiply */
extern "C" void sgemv_(char *trans, int *m, int *n, varType *alpha, varType *A,
                       int *lda, varType *x, int *incx, varType *beta, 
                       varType *y, int *incy);

/** @brief Declaration for BLAS double-precision dot product */
extern "C" varType sdot_(int *n, varType *dx, int *incx, varType *dy, int *incy);

#else
/** @brief Declaration for LAPACK double-precision SVD routine */
extern "C" void dgesvd_(char *jobu, char *jobvt, int *m, int *n, varType *A,
                        int *lda, varType *S, varType *U, int *ldu, varType *VT,
                        int *ldvt, varType *work, int *lwork, int *info);

/** @brief Declaration for BLAS double-precision matrix-matrix multiply */
extern "C" void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
                       varType *alpha, varType *A, int *lda, varType *B,
                       int *ldb, varType *beta, varType *C, int *ldc);

/** @brief Declaration for BLAS double-precision matrix-vector multiply */
extern "C" void dgemv_(char *trans, int *m, int *n, varType *alpha, varType *A,
                       int *lda, varType *x, int *incx, varType *beta, 
                       varType *y, int *incy);

/** @brief Declaration for BLAS double-precision dot product */
extern "C" varType ddot_(int *n, varType *dx, int *incx, varType *dy, int *incy);

#endif

#endif


