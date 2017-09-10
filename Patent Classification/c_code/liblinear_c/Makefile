CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fno-stack-protector
LIBS = blas/blas.a
SHVER = 3
OS = $(shell uname)
#LIBS = -lblas

all:  sthread mthread

sthread: tron.o linear.o q1.c q2.c q3.c blas/blas.a
	$(CXX) $(CFLAGS) -o q1 q1.c tron.o linear.o $(LIBS)
	$(CXX) $(CFLAGS) -o q2 q2.c tron.o linear.o $(LIBS)
	$(CXX) $(CFLAGS) -o q3 q3.c tron.o linear.o $(LIBS)

mthread: tron.o linear.o q4.c q5.c blas/blas.a
	$(CXX) $(CFLAGS) -o q4 q4.c -lpthread -lrt tron.o linear.o $(LIBS)
	$(CXX) $(CFLAGS) -o q5 q5.c -lpthread -lrt tron.o linear.o $(LIBS)

clean:
	rm -f q[0-9]
