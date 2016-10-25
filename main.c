#include <stdio.h>
#include <math.h>
#include "rng.h"
#include "grid_fft.h"
#include "grid_pk.h"

struct Particle
{
  double *m;
  double *x;
  double *y;
  double *z;
};


double *grid_make_gaussian_kernel(double A, double r_cells, FFTW_Grid_Info grid_info)
{
  double *kernel = allocate_real_fftw_grid(grid_info);
  int i,j,k,ijk;
  int nx_start = grid_info.nx_local_start;
  int nx_local = grid_info.nx_local;
  int nx = grid_info.nx;
  int ny = grid_info.ny;
  int nz = grid_info.nz;
  double r;
  double x;
  double y;
  double z;

  /*populate gaussian kernel*/
  for(i=0;i<nx_local;++i)
    for(j=0;j<ny;++j)
      for(k=0;k<nz;++k)
      {
  //grid index
  ijk = grid_ijk(i,j,k,grid_info);

  if(i>nx/2)
  {
    //radius from corner in cells
    //x = nx - (i + nx_start);
    x = nx - i;

  }else{
    x = i;
  }

  if(j>ny/2)
  {
    //radius from corner in cells
    y = ny - j;
  }else{
    y = j;
  }

  if(k>nz/2)
  {
    //radius from corner in cells
    z = nz - k;
  }else{
    z = k;
  }

  //radius
  r = sqrt(x*x + y*y + z*z);

  //3-d gaussian
  kernel[ijk] = A*exp( -0.5*pow( r/r_cells, 2) );

  }

  /*return the answer*/
  return kernel;
}

void create_normal_distribution(struct Particle *p, int N, double sigma)
{
  double dx;
  double dy;
  double dz;
  p->m = (double *) calloc(N,sizeof(double));
  p->x = (double *) calloc(N,sizeof(double));
  p->y = (double *) calloc(N,sizeof(double));
  p->z = (double *) calloc(N,sizeof(double));

  for(int i=0;i<N;i++)
  {
    p->m[i] = 1.0;
    dx = rng_gaussian(0,sigma);
    if(dx<0)
      dx += 1.0;
    dy = rng_gaussian(0,sigma);
    if(dy<0)
      dy += 1.0;
    dz = rng_gaussian(0,sigma);
    if(dz<0)
      dz += 1.0;
    p->x[i] = dx;
    p->y[i] = dy;
    p->z[i] = dz;
  }
}
void create_uniform_distribution(struct Particle *p, int N, double a, int id)
{
  double dx;
  double dy;
  double dz;
  p->m = (double *) calloc(N,sizeof(double));
  p->x = (double *) calloc(N,sizeof(double));
  p->y = (double *) calloc(N,sizeof(double));
  p->z = (double *) calloc(N,sizeof(double));
  set_rng_uniform_seed(1337+id);

  for(int i=0;i<N;i++)
  {
    p->m[i] = 1.0;
    dx = rng_uniform(-1.0*a,a);
    if(dx<0)
      dx += 1.0;
    dy = rng_uniform(-1.0*a,a);
    if(dy<0)
      dy += 1.0;
    dz = rng_uniform(-1.0*a,a);
    if(dz<0)
      dz += 1.0;
    p->x[i] = dx;
    p->y[i] = dy;
    p->z[i] = dz;
  }
}
int main(int argc, char **argv)
{
  int N = 100000; //number of objects

  struct Particle p;

  FFTW_Grid_Info grid_info;
  char fname[200];

  double *u;
  double *dfk;
  double A     = 1.0;
  double sigma = 0.1;
  int ns = 10;
  int nx = 64;
  int ny = 64;
  int nz = 64;

  if(argc!=1)
  {
    nx = atoi(argv[1]);
    ny = nx;
    nz = nx;
  }

  grid_info.nx = nx;
  grid_info.ny = ny;
  grid_info.nz = nz;
  grid_info.ndim = 3;
  grid_info.BoxSize = 1.0;


  //kN = \pi/H
  //H = 1./nx

  int myid;
  int numprocs;

  if(argc>=3)
    ns = atoi(argv[2]);
  if(argc>=4)
    N = atoi(argv[3]);

  printf("nx %d ns %d N %d\n",nx,ns,N);

  MPI_Init(&argc,&argv);
  MPI_Comm world = MPI_COMM_WORLD;
  MPI_Comm_rank(world,&myid);
  MPI_Comm_size(world,&numprocs);

  initialize_mpi_local_sizes(&grid_info, world);
  for(int id=0;id<ns;id++)
  {
    set_rng_uniform_seed(1337+id);
    set_rng_gaussian_seed(1337+id);

    //create_normal_distribution(&p,N,sigma);
    create_uniform_distribution(&p,N,A,id);


    //printf("p[0] %e %e %e\n",p.x[0],p.y[0],p.z[0]);

    //grid particles
    u = grid_ngp(p.x,p.y,p.z,p.m,N,grid_info);
    //u =  grid_make_gaussian_kernel(A, sigma*grid_info.nx, grid_info);

    //output the ngp grid
    sprintf(fname,"ngp.%d.dat",id);
    output_fft_grid(fname, u, grid_info, 0, nx, 0, ny, 0, nz, myid, numprocs, world);
    printf("output...\n");

    //compute <|df(k)|^2>
    dfk = grid_dfk(N,u,grid_info,world);

    //output the grid
    sprintf(fname,"dfk.%d.dat",id);
    output_fft_grid(fname, dfk, grid_info, 0, nx, 0, ny, 0, nz, myid, numprocs, world);

    

    if(id==0)
    {
      FILE *fp;
      sprintf(fname,"particles.%d.dat",id);
      fp = fopen(fname,"w");
      for(int i=0;i<N;i++)
        fprintf(fp,"%e\t%e\t%e\n",p.x[i],p.y[i],p.z[i]);
      fclose(fp);
    }

    free(p.x);
    free(p.y);
    free(p.z);
    free(p.m);

    free(u);
    free(dfk);
  }
  return 0;
}
