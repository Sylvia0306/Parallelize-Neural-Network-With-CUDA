NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -arch=sm_20 -pg # Added -pg here for profiling
else
NVCC_FLAGS  = -O3 -pg # Added -pg here for profiling
endif
LD_FLAGS    = -lcudart -pg # Added -pg here for profiling
EXE	        = nn
OBJ	        = main.o

default: $(EXE)

main.o: main.cu app.cu back.cu init.cu propagate.cu random.cu weight.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
	rm -rf BPN.txt
	rm -rf gmon.out
	rm -rf report.txt