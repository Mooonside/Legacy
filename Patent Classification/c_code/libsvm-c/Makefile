CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC 
#-fno-stack-protector
SHVER = 2
OS = $(shell uname)

all: mthread

mthread:q6.c q7.c svm.o
	$(CXX) $(CFLAGS) q6.c svm.o -o q6 -lpthread -lrt -lm
	$(CXX) $(CFLAGS) q7.c svm.o -o q7 -lpthread -lrt -lm

clean:
	rm q[0-7]
