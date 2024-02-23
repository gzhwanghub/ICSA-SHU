MPICXX=mpicxx
CFLAGS=-std=c++0x  
INC = -I. -I../include

all:admm
admm:train.cpp asyc_admm.o tron.o conf_util.o prob.o logistic.o svm.o
	$(MPICXX) -o admm train.cpp asyc_admm.o tron.o conf_util.o prob.o logistic.o svm.o $(INC) $(CFLAGS)
	
conf_util.o:conf_util.cpp
	$(MPICXX) -c conf_util.cpp $(INC) $(CFLAGS)
prob.o:prob.cpp
	$(MPICXX) -c prob.cpp $(INC) $(CFLAGS)
asyc_admm.o:asyc_admm.cpp 
	$(MPICXX) -c asyc_admm.cpp $(INC) $(CFLAGS)
logistic.o:logistic.cpp 
	$(MPICXX) -c logistic.cpp $(INC) $(CFLAGS)

tron.o:tron.cpp
	$(MPICXX) -c tron.cpp $(INC) $(CFLAGS)
	
svm.o:svm.cpp
	$(MPICXX) -c svm.cpp $(INC) $(CFLAGS)
clean:
	-rm -f *.o admm


