static char help []  = "Testing the sections of Linear inversing.\n";

#include "problem.h"

int main(int argc,char **argv)
{
	PetscErrorCode ierr;
	PetscInitialize(&argc,&argv,(char *)0,help);

	string s("Harmonic");

	Problem problem(100,10,s);

	ierr = PetscFinalize();							CHKERRQ(ierr);
	return 0;
}


