CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = 
EXAMPLESF       = ex1.c ex1f.F90 ex2a.c ex2b.c ex2b_MMS.c
MANSEC          = TS
DIRS            = toy-problem
CLEANFILES      = SA-data/*
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

all: ex1 ex1f ex2a ex2b ex2b_MMS
.PHONY: all
