EXEC   = jing

OBJS   = main.o grid_fft.o grid_pk.o rng.o

INCL   = grid_fft.h grid_pk.h rng.h

LIBS     = -lgsl -lgslcblas -lmpi -lfftw3_mpi -lfftw3 -lm -stdlib=libstdc++

CC       = mpicxx
CXX      = mpicxx
CFLAGS   = -fopenmp -stdlib=libstdc++
CPPFLAGS = -stdlib=libstdc++


$(EXEC): $(OBJS) 
	 $(CXX) $(OBJS) $(LIBS) -o $(EXEC)   
         

$(OBJS): $(INCL) 

.PHONY : clean

clean:
	 rm -f $(OBJS) $(EXEC)

