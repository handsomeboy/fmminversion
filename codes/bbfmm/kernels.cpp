/** @file kernels.cpp
 *  @author Will Fong <willfong@gmail.com>
 *  @version 1.0
 * 
 *  Implements constructors for each kernel of interest and functions for 
 *  evaluating the kernels.
 */

#include <string>
#include <sstream>
#include "kernels.h"

using namespace std;

// Constructor for Laplacian kernel
KLaplacian::KLaplacian(const int n) {
    odof = 1;                   // Observation degrees of freedom
    sdof = 1;                   // Source degrees of freedom
    homogen = new varType[odof];
    value = new varType[odof];
    Kmat = new string[odof];
    Umat = new string[odof];
    Vmat = new string[odof];
    
    // Form the file names for storing the kernel matrices and singular vectors
    stringstream nstr;
    nstr << n;
    string head = "./lib/laplacian";
    string tail = "n" + nstr.str() + ".out";
    for (int idof=0; idof<odof; idof++) {
        homogen[idof] = -1.0;
        
        stringstream index;
        index << idof;
        Kmat[idof] = head + "K" + index.str() + tail;
        Umat[idof] = head + "U" + index.str() + tail;
        Vmat[idof] = head + "V" + index.str() + tail;
    }
}

// Destructor for Laplacian kernel
KLaplacian::~KLaplacian() {
    // Free allocated memory
    delete homogen;
    delete value;
    delete [] Kmat;
    delete [] Umat;
    delete [] Vmat;
    
    // Set pointers to NULL
    homogen = NULL;
    value = NULL;
    Kmat = NULL;
    Umat = NULL;
    Vmat = NULL;
}

// Evaluation routine for Laplacian kernel
void KLaplacian::evaluate(Vector3 observpos, Vector3 sourcepos,
                          varType *value) {
    // Compute 1/r
    Vector3 diff = sourcepos - observpos;
    varType rinv = 1.0/diff.length();
    
    // Output result (1/r)
    value[0] = rinv;    
}

// Evaluation routine for Laplacian kernel with exclusions
void KLaplacian::evaluate_exclude(Vector3 observpos, Vector3 sourcepos,
                                  varType *value) {
    // Compute r^2
    Vector3 diff = sourcepos - observpos;
    varType r2   = diff.length2();
    
    // For excluded interactions return zero
    if (r2 == 0.0)
        value[0] = 0.0;
    
    // Evaluate kernel as usual for non-excluded interactions
    else {
        varType rinv = 1.0/sqrt(r2);
    
        // Output result (1/r)
        value[0] = rinv;
    }
}

// Constructor for Laplacian force kernel
KLaplacianForce::KLaplacianForce(const int n) {
    odof = 3;                   // Observation degrees of freedom
    sdof = 1;                   // Source degrees of freedom
    homogen = new varType[odof];
    value = new varType[odof];
    Kmat = new string[odof];
    Umat = new string[odof];
    Vmat = new string[odof];
    
    // Form the file names for storing the kernel matrices and singular vectors
    stringstream nstr;
    nstr << n;
    string head = "./lib/laplacianforce";
    string tail = "n" + nstr.str() + ".out";
    for (int idof=0; idof<odof; idof++) {
        homogen[idof] = -2.0;
        
        stringstream index;
        index << idof;
        Kmat[idof] = head + "K" + index.str() + tail;
        Umat[idof] = head + "U" + index.str() + tail;
        Vmat[idof] = head + "V" + index.str() + tail;
    }
}

// Destructor for Laplacian force kernel
KLaplacianForce::~KLaplacianForce() {
    // Free allocated memory
    delete [] homogen;
    delete [] value;
    delete [] Kmat;
    delete [] Umat;
    delete [] Vmat;
    
    // Set pointers to NULL
    homogen = NULL;
    value = NULL;
    Kmat = NULL;
    Umat = NULL;
    Vmat = NULL;
}

// Evaluation routine for Laplacian force kernel
void KLaplacianForce::evaluate(Vector3 observpos, Vector3 sourcepos, 
                               varType *value) {
    // Compute 1/r
    Vector3 diff = sourcepos - observpos;
    varType rinv2  = 1.0/diff.length2();
    varType rinv   = sqrt(rinv2);
    varType rinv3  = rinv2*rinv;
    
    // Output result (rvec/r^3)
    value[0] = diff.x*rinv3;
    value[1] = diff.y*rinv3;
    value[2] = diff.z*rinv3;
}

// Evaluation routine for Laplacian force kernel with exclusions
void KLaplacianForce::evaluate_exclude(Vector3 observpos, Vector3 sourcepos, 
                                       varType *value) {
    // Compute r^2
    Vector3 diff = sourcepos - observpos;
    varType r2   = diff.length2();
    
    // For excluded interactions return the zero vector
    if (r2 == 0.0) {
        value[0] = 0.0;
        value[1] = 0.0;
        value[2] = 0.0;
    }
    
    // Evaluate kernel as usual for non-excluded interactions
    else {    
        varType rinv2  = 1.0/r2;
        varType rinv   = sqrt(rinv2);
        varType rinv3  = rinv2*rinv;
    
        // Output result (rvec/r^3)
        value[0] = diff.x*rinv3;
        value[1] = diff.y*rinv3;
        value[2] = diff.z*rinv3;
    }
}


// Constructor for 2-D gravitational force kernel
KGravForce2D::KGravForce2D(const int n) {
    odof = 2;                   // Observation degrees of freedom
    sdof = 1;                   // Source degrees of freedom
    homogen = new varType[odof];
    value = new varType[odof];
    Kmat = new string[odof];
    Umat = new string[odof];
    Vmat = new string[odof];
    
    // Form the file names for storing the kernel matrices and singular vectors
    stringstream nstr;
    nstr << n;
    string head = "./lib/gravforce2d";
    string tail = "n" + nstr.str() + ".out";
    for (int idof=0; idof<odof; idof++) {
        homogen[idof] = -1.0;
        
        stringstream index;
        index << idof;
        Kmat[idof] = head + "K" + index.str() + tail;
        Umat[idof] = head + "U" + index.str() + tail;
        Vmat[idof] = head + "V" + index.str() + tail;
    }
}

// Destructor for 2-D gravitational force kernel
KGravForce2D::~KGravForce2D() {
    // Free allocated memory
    delete [] homogen;
    delete [] value;
    delete [] Kmat;
    delete [] Umat;
    delete [] Vmat;
    
    // Set pointers to NULL
    homogen = NULL;
    value = NULL;
    Kmat = NULL;
    Umat = NULL;
    Vmat = NULL;
}

// Evaluation routine for 2-D gravitational force kernel
void KGravForce2D::evaluate(Vector3 observpos, Vector3 sourcepos, 
                            varType *value) {
    /* Must use version with exclusions for all evaluations - otherwise
     * interaction between points (x,y,z0) and (x,y,z1) are allowed although
     * this is undefined */
    evaluate_exclude(observpos,sourcepos,value);
}

// Evaluation routine for 2-D gravitational force kernel with exclusions
void KGravForce2D::evaluate_exclude(Vector3 observpos, Vector3 sourcepos, 
                                       varType *value) {
    // Compute x^2+y^2
    Vector3 diff = sourcepos - observpos;
    varType r2   = diff.x*diff.x+diff.y*diff.y;
    
    // For excluded interactions return the zero vector
    if (r2 == 0.0) {
        value[0] = 0.0;
        value[1] = 0.0;
    }
    
    // Evaluate kernel as usual for non-excluded interactions
    else {    
        varType rinv2  = 1.0/r2;
        
        // Output result (-(x,y)/r^2)
        value[0] = diff.x*rinv2;
        value[1] = diff.y*rinv2;
    }
}