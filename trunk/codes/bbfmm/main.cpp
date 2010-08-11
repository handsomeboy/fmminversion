/** @file main.cpp
 *  @author Will Fong <willfong@gmail.com>
 *  @version 1.0
 * 
 *  Driver program for bbFMM.
 */
#include <iostream>
#include "bbfmm.h"

using namespace std;

/** @brief Driver routine */
int main (int argc, char * const argv[]) {
    /*************************************
     *  Simulation of 2-D Spiral Galaxy  *
     *************************************/
    /*
    // Simulation parameters
    int N = 2500;   // Number of masses
    int n = 5;      // Number of Chebyshev nodes
    SpiralGalaxy2D(N,n);
    //*/
    
    /*****************************
     *  Example - Accuracy Test  *
     *****************************/
    //*
    // Simulation parameters
    int Ns = 10000;                 // Number of sources
    Vector3 boxcntr(0.0,0.0,0.0);   // Center of simulation box
    Vector3 L(1.0,1.0,1.0);         // Dimensions of simulation box
    int n = 4;                      // Number of Chebyshev nodes
    
    // Specify kernel
    KLaplacianForce *laplacianforce = new KLaplacianForce(n);
   std::cout<<"Running Accuracy test"<<std::endl; 
    // Run the accuracy test for bbFMM
    AccuracyTest(Ns,boxcntr,L,laplacianforce,n);
    
    // Free allocated memory
    delete laplacianforce;
    //*/
    
    return 0;
}
