p: relaxation_technique.c
	mpicc -Wall -o relaxation relaxation_technique.c -lm

s: relaxation_technique_sequential.c
	gcc -o relaxation_seq relaxation_technique_sequential.c -lm -lpthread