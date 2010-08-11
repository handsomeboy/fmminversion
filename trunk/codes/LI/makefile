PETSC_DIR 	= 	/home/arvind/install/petsc-3.1-p4
PETSC_ARCH      =       linux-gnu-cxx-debug
LIB             =       -larpack


include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

test: test.o  problem.o chkopts
	-${CLINKER} -o test test.o problem.o ${PETSC_KSP_LIB} ${LIB}
	${RM} test.o

