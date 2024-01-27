/******************************************************************************
                     P R O P A G A T I N G   S I G N A L S
 ******************************************************************************/

// Parallelization with CUDA that did not work
// __global__ void PropagateLayerKernel( NET* Net, LAYER *Upper, LAYER *Lower, REAL *upper_d, REAL *lower_d ) {

//   INT  i,j;
//   REAL Sum;

//   i = threadIdx.x + blockDim.x * blockIdx.x + 1 ;
//   j = threadIdx.y + blockDim.y * blockIdx.y ;
  
//   if ( i <= Upper->Units ) {
//     Sum = 0 ;
//     if ( j <= Lower->Units ) {
//       Sum += upper_d[ j * blockDim.x + i ] * lower_d[i] ;
//     } // if

//     Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum)) ;
//   } // if

// } // PropagateLayerKernel()

// void PropagateNet( NET* Net )
// {
//   INT l;
//   cudaError_t cuda_ret ; // get error message
//   dim3 dim_grid, dim_block ;
//   REAL *upper_d, *lower_d ; // device

//   for ( l = 0 ; l < NUM_LAYERS - 1 ; l++ ) {

//     // allocate device variables
//     // printf( "Allocating device variables..." ) ; fflush(stdout) ;
//     cuda_ret = cudaMalloc((void**)&upper_d, Net->Layer[l]->Units * sizeof(REAL) ) ;
//     if ( cuda_ret != cudaSuccess ) assert( "Unable to allocate device memory" ) ;

//     cuda_ret = cudaMalloc((void**)&lower_d, Net->Layer[l+1]->Units * sizeof(REAL) ) ;
//     if ( cuda_ret != cudaSuccess ) assert( "Unable to allocate device memory" ) ;

//     cudaDeviceSynchronize() ;
//     // copy data to variables
//     // printf( "Copying data from host to device..." ) ; fflush(stdout) ;
//     cuda_ret = cudaMemcpy( upper_d, Net->Layer[l]->Weight, Net->Layer[l]->Units * sizeof(REAL), cudaMemcpyHostToDevice ) ;
//     if( cuda_ret != cudaSuccess ) assert( "Unable to copy memory to the device" ) ;

//     cuda_ret = cudaMemcpy( lower_d, Net->Layer[l+1]->Output, Net->Layer[l+1]->Units * sizeof(REAL), cudaMemcpyHostToDevice ) ;
//     if( cuda_ret != cudaSuccess ) assert( "Unable to copy memory to the device" ) ;

//     cudaDeviceSynchronize() ;

//     // Launching the kernel...
//     // printf("Launching kernel...") ; fflush(stdout) ;
//     dim_block.x = BLOCK_SIZE ; dim_block.y = dim_block.z = 1 ;
//     dim_grid.x = Net->Layer[l]->Units ; dim_grid.y = dim_grid.z = 1 ;

//     PropagateLayerKernel<<<dim_grid, dim_block >>>( Net, Net->Layer[l], Net->Layer[l+1], upper_d, lower_d ) ;

//     cuda_ret = cudaDeviceSynchronize() ;
//     if( cuda_ret != cudaSuccess ) assert( "Unable to launch/execute kernel" ) ;

//     // Free memory
//     cudaFree( upper_d ) ; cudaFree( lower_d ) ;
//   } // for
// }

// Parallelization with OpenMP
void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
{
  INT  i,j;
  REAL Sum;

  #pragma omp parallel for
  for (i=1; i<=Upper->Units; i++) {
    Sum = 0;

    #pragma omp parallel
    for (j=0; j<=Lower->Units; j++) {
      Sum += Upper->Weight[i][j] * Lower->Output[j];
    }
    Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum));
  }
}

// Original C code
// void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
// {
//   INT  i,j;
//   REAL Sum;

//   for (i=1; i<=Upper->Units; i++) {
//     Sum = 0;

//     for (j=0; j<=Lower->Units; j++) {
//       Sum += Upper->Weight[i][j] * Lower->Output[j];
//     }
//     Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum));
//   }
// }

void PropagateNet( NET* Net ) {
  INT l;
  for ( l=0 ; l < NUM_LAYERS - 1 ; l++ ) {
    PropagateLayer(Net, Net->Layer[l], Net->Layer[l+1]);
  }
}