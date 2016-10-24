#include <math.h>
#include "grid_pk.h"

double *grid_ngp(double *x, double *y, double *z, double *m, int N, FFTW_Grid_Info grid_info)
{
  //create a new grid
  //with an NGP assignment
  //of the particles

  //grid sizes
  int nx = grid_info.nx;
  int ny = grid_info.ny;
  int nz = grid_info.nz;

  //grid indices
  double dx, dy, dz;
  int ix, iy, iz;
  int ijk;

  //create the grid
  double *u = allocate_real_fftw_grid(grid_info);

  //loop over the particles and assign them
  //to the grid using NGP
  for(int i=0;i<N;i++)
  {
    dx = nx*x[i];
    if(fmod(dx,1.) >= 0.5)
    {
      ix = floor(dx)+1;
    }else{
      ix = floor(dx);
    }
    if(ix>=nx)
      ix-=nx;
    if(ix<0)
      ix+=nx;

    dy = ny*y[i];
    if(fmod(dy,1.) >= 0.5)
    {
      iy = floor(dy)+1;
    }else{
      iy = floor(dy);
    }
    if(iy>=ny)
      iy-=ny;
    if(iy<0)
      iy+=ny;

    dz = nz*z[i];
    if(fmod(dz,1.) >= 0.5)
    {
      iz = floor(dz)+1;
    }else{
      iz = floor(dz);
    }
    if(iz>=nz)
      iz-=nz;
    if(iz<0)
      iz+=nz;

    //get index on the grid
    ijk = grid_ijk(ix,iy,iz,grid_info);

    //add the particle to the grid
    u[ijk] += m[i];
  }

  //return the grid
  return u;
}

double w_p(int p, double kx, double ky, double kz, FFTW_Grid_Info grid_info)
{
  int nx = grid_info.nx;
  double kN = M_PI*((double) nx)/grid_info.BoxSize;
  double k1 = 0.5*M_PI*kx/kN;
  double k2 = 0.5*M_PI*ky/kN;
  double k3 = 0.5*M_PI*kz/kN;
  return pow( sin(k1)*sin(k2)*sin(k3)/(k1*k2*k3), p);
}

double *grid_dfk(double *u, FFTW_Grid_Info grid_info, MPI_Comm world)
{
  int nx       = grid_info.nx;
  int ny       = grid_info.ny;
  int nz       = grid_info.nz;

  //normalization
  double scale = 1./( ((double) grid_info.nx)*((double) grid_info.ny)*((double) grid_info.nz) );

  fftw_complex *uk;
  fftw_plan plan;

  int ijk, ijkc;

  //real power spectrum
  double *dfk = allocate_real_fftw_grid(grid_info);


  //allocate work and transform
  uk    = allocate_complex_fftw_grid(grid_info);

  //create the fftw plans
  plan  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, uk, uk, world, FFTW_FORWARD,  FFTW_ESTIMATE);

  //get complex version of A
  grid_copy_real_to_complex_in_place(u, uk, grid_info);

  //perform the forward transform on the components of u
  fftw_execute(plan);

  double real_tot=0, img_tot=0;

  //find delta^f_k;
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      for(int k=0;k<nz;k++)
      {
        ijk  = grid_ijk(i,j,k,grid_info);
        ijkc = grid_complex_ijk(i,j,k,grid_info);
        real_tot += uk[ijkc][0];
        img_tot  += uk[ijkc][1];

        dfk[ijk] = uk[ijkc][0]*uk[ijkc][0] + uk[ijkc][1]*uk[ijkc][1];
        dfk[ijk] *= (scale*scale); // one for each factor of u
        //dfk[ijk] = uk[ijkc][0];
        //dfk[ijk] *= (scale); // one for each factor of u

      }
  printf("real_tot %e img_tot %e\n",real_tot,img_tot);

  //remove DC
  //dfk[0] -= 1.0;

  //free memory
  fftw_free(uk);
  fftw_destroy_plan(plan);

  //return the result
  return dfk;
}

/*! \fn double *grid_copy_real_to_complex_in_place(double *source, fftw_complex *copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid in place into the real elements of a complex grid. */
void grid_copy_real_to_complex_in_place(double *source, fftw_complex *copy, FFTW_Grid_Info grid_info)
{
        int i, j, k;

        int nx_local = grid_info.nx_local;
        int nx       = grid_info.nx;
        int ny       = grid_info.ny;
        int nz       = grid_info.nz;

        int ijk;
        int ijkc;

        //Copy source into copy.
        for(i=0;i<nx_local;++i)
                for(j=0;j<ny;++j)
                        for(k=0;k<nz;++k)
                        {
                                //real grid index
                                ijk = grid_ijk(i,j,k,grid_info);

                                //complex grid index
                                ijkc = grid_complex_ijk(i,j,k,grid_info);

                                //copy
                                copy[ijkc][0] = source[ijk];
                                copy[ijkc][1] = 0;
                        }
}
