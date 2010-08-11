#
# File: Makefile
# Description: Makefile for bbFMM
# ------------------------------------------------------------------
# Use "make" to compile the code.
#
# Black-Box Fast Multipole Method (bbFMM)
# Will Fong <willfong@gmail.com>
#

CC = g++
LD = g++
LDPATH  = -L/usr/lib64
LDFLAGS = -llapack -lblas -lm
FLAGS   = -Wall -O3 -g
PROF    = 

######################################
#     PRECISION (DEFINE ONLY ONE)    #
######################################
# SINGLE PRECISION
#PRECISION = -DSINGLE

# DOUBLE PRECISION (DEFAULT)
PRECISION = -DDOUBLE

PROG = bbfmm
OBJS = main.o bbfmm.o kernels.o

all:		$(PROG)

main.o:		main.cpp bbfmm.h
			$(CC) $(FLAGS) $(KERNEL) $(PRECISION) -c main.cpp $(PROF)

%.o:		%.cpp %.h
			$(CC) $(FLAGS) $(KERNEL) $(PRECISION) -c $< $(PROF)

$(PROG):	$(OBJS)
			$(LD) $(OBJS) -o $(PROG) $(LDPATH) $(LDFLAGS) $(PROF)

clean:
			/bin/rm -rf *.o *~ \#*\# $(PROG)

clobber:	
			/bin/rm -rf lib/*K*out lib/*U*out lib/*V*out