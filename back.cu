/******************************************************************************
                  B A C K P R O P A G A T I N G   E R R O R S
 ******************************************************************************/

// Parallelization with CUDA
__device__ double atomicAddDouble(double* address, double val) {\

  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__global__ void ComputeOutputErrorKernel(INT Units, REAL *Output, REAL *Error, REAL NetGain, REAL *Target_d, REAL *TotalError_d) {
  INT i = threadIdx.x + blockIdx.x * blockDim.x;
  REAL Out, Err;
  if (i <= Units) {
    Out = Output[i+1];
    Err = Target_d[i] - Out;
    Error[i+1] = NetGain * Out * (1 - Out) * Err;
    atomicAddDouble(TotalError_d, 0.5 * Err * Err);
  }
}

void ComputeOutputError(NET* Net, REAL* Target) {
  
  // Initialize host variables

  REAL *Output_d;
  REAL *Error_d;
  REAL *TotalError_d;
  REAL *Target_d;
  cudaError_t cuda_ret;

  // Allocate device variables

  cuda_ret = cudaMalloc(&Output_d, (Net->OutputLayer->Units + 1) * sizeof(REAL));
  if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - ComputeOutputError");
  cuda_ret = cudaMalloc(&Target_d, (Net->OutputLayer->Units) * sizeof(REAL));
  if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - ComputeOutputError");
  cuda_ret = cudaMalloc(&Error_d, (Net->OutputLayer->Units + 1) * sizeof(REAL));
  if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - ComputeOutputError");
  cuda_ret = cudaMalloc(&TotalError_d, sizeof(REAL));
  if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - ComputeOutputError");

  // Copy host variables to device

  cuda_ret = cudaMemcpy(Output_d, Net->OutputLayer->Output, (Net->OutputLayer->Units + 1) * sizeof(REAL), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - ComputeOutputError");
  cuda_ret = cudaMemcpy(Target_d, Target, Net->OutputLayer->Units * sizeof(REAL), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - ComputeOutputError");
  cuda_ret = cudaMemset(TotalError_d, 0, sizeof(REAL));
  if (cuda_ret != cudaSuccess) printf("Unable to set device memory - ComputeOutputError");

  // Launch kernel

  dim3 DimGrid((Net->OutputLayer->Units + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  ComputeOutputErrorKernel<<<DimGrid, DimBlock>>>(Net->OutputLayer->Units, Output_d, Error_d, Net->Gain, Target_d, TotalError_d);

  // Copy device variables to host

  cuda_ret = cudaMemcpy(Net->OutputLayer->Error, Error_d, (Net->OutputLayer->Units + 1) * sizeof(REAL), cudaMemcpyDeviceToHost);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to host - ComputeOutputError");
  cuda_ret = cudaMemcpy(&(Net->Error), TotalError_d, sizeof(REAL), cudaMemcpyDeviceToHost);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to host - ComputeOutputError");

  // Free memory

  cudaFree(Output_d);
  cudaFree(Target_d);
  cudaFree(Error_d);
  cudaFree(TotalError_d);
}

// Original C code
// void ComputeOutputError(NET* Net, REAL* Target)
// {
//   INT  i;
//   REAL Out, Err;
   
//   Net->Error = 0;
//   for (i=1; i<=Net->OutputLayer->Units; i++) {
//     Out = Net->OutputLayer->Output[i];
//     Err = Target[i-1]-Out;
//     Net->OutputLayer->Error[i] = Net->Gain * Out * (1-Out) * Err;
//     Net->Error += 0.5 * sqr(Err);

//   }
// }

// Parallelization with CUDA that did not work
// __global__ void BackpropagateLayerKernel(INT lowerUnits, INT upperUnits, REAL NetGain, REAL *Output_d, REAL *Weight_d, REAL *Error_d, REAL *TotalError_d) {
//   INT i = threadIdx.x + blockIdx.x * blockDim.x + 1;
//   INT j = threadIdx.y + blockIdx.y * blockDim.y + 1;

//   if (i <= lowerUnits && j <= upperUnits) {
//     REAL Out = Output_d[i];
//     REAL Err = 0;
//     __syncthreads();

//     Err += Weight_d[j * (lowerUnits) + i] * Error_d[j];

//     __syncthreads();
//     TotalError_d[i] = NetGain * Out * (1 - Out) * Err;
//   }
// }

// void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower) {

//   // Initialize host variables
  
//   REAL *Output_d;
//   REAL *Weight_d;
//   REAL *Error_d;
//   REAL *TotalError_d;
//   cudaError_t cuda_ret;  

//   // Allocate device variables

//   cuda_ret = cudaMalloc(&Output_d, (Lower->Units) * sizeof(REAL));
//   if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - BackpropagateLayer");
//   cuda_ret = cudaMalloc(&Weight_d, (Upper->Units * Lower->Units) * sizeof(REAL));
//   if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - BackpropagateLayer");
//   cuda_ret = cudaMalloc(&Error_d, (Upper->Units) * sizeof(REAL));
//   if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - BackpropagateLayer");
//   cuda_ret = cudaMalloc(&TotalError_d, (Lower->Units) * sizeof(REAL));
//   if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - BackpropagateLayer");

//   // Copy host variables to device

//   cuda_ret = cudaMemcpy(Output_d, Lower->Output, (Lower->Units) * sizeof(REAL), cudaMemcpyHostToDevice);
//   if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - BackpropagateLayer");
//   cuda_ret = cudaMemcpy(Weight_d, Upper->Weight, (Lower->Units * Upper->Units) * sizeof(REAL), cudaMemcpyHostToDevice);
//   if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - BackpropagateLayer");
//   cuda_ret = cudaMemset(TotalError_d, 0, (Lower->Units) * sizeof(REAL));
//   if (cuda_ret != cudaSuccess) printf("Unable to set device memory - BackpropagateLayer");

//   // Launch kernel

//   dim3 DimGrid((Lower->Units + BLOCK_SIZE - 1) / BLOCK_SIZE, (Upper->Units + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
//   dim3 DimBlock(BLOCK_SIZE, 1, 1);
//   BackpropagateLayerKernel<<<DimGrid, DimBlock>>>(Lower->Units, Upper->Units, Net->Gain, Output_d, Weight_d, Error_d, TotalError_d);
  
//   // Copy device variables to host

//   cuda_ret = cudaMemcpy(&(Lower->Error), TotalError_d, (Lower->Units) * sizeof(REAL), cudaMemcpyDeviceToHost);
//   if (cuda_ret != cudaSuccess) printf("Unable to copy memory to host - BackpropagateLayer");

//   // Free memory

//   cudaFree(Output_d);
//   cudaFree(Weight_d);
//   cudaFree(Error_d);
//   cudaFree(TotalError_d);
// }


// Parallelization with OpenMP
void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)
{
  INT  i,j;
  REAL Out, Err;

  #pragma omp parallel for
  for (i=1; i<=Lower->Units; i++) {
    Out = Lower->Output[i];
    Err = 0;

    #pragma omp parallel
    for (j=1; j<=Upper->Units; j++) {
      Err += Upper->Weight[j][i] * Upper->Error[j];
    }
    Lower->Error[i] = Net->Gain * Out * (1-Out) * Err;
  }
}

// Original C code
// void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)
// {
//   INT  i,j;
//   REAL Out, Err;

//   for (i=1; i<=Lower->Units; i++) {
//     Out = Lower->Output[i];
//     Err = 0;

//     for (j=1; j<=Upper->Units; j++) {
//       Err += Upper->Weight[j][i] * Upper->Error[j];
//     }
//     Lower->Error[i] = Net->Gain * Out * (1-Out) * Err;
//   }
// }

void BackpropagateNet(NET* Net)
{
  INT l;
   
  for (l=NUM_LAYERS-1; l>1; l--) {
    BackpropagateLayer(Net, Net->Layer[l], Net->Layer[l-1]);
  }
}

// Parallelization with CUDA that did not work
// __global__ void adjustWeightsKernel(REAL* d_Output, REAL* d_Error, REAL* d_Weight, REAL* d_dWeight, REAL eta, REAL alpha, int lowerUnits, int upperUnits) {
//   int i = blockIdx.y * blockDim.y + threadIdx.y;
//   int j = blockIdx.x * blockDim.x + threadIdx.x;

//   if (i < upperUnits && j <= lowerUnits) {
//     REAL Out = d_Output[j];
//     REAL Err = d_Error[i];
//     REAL dWeight = d_dWeight[i * (lowerUnits + 1) + j];
    
//     d_Weight[i * (lowerUnits + 1) + j] += eta * Err * Out + alpha * dWeight;
//     d_dWeight[i * (lowerUnits + 1) + j] = eta * Err * Out;
//   }
// }

// void AdjustWeights(NET* Net) {
//   for (int l = 1; l < NUM_LAYERS; l++) {

//     // Initialize host variables

//     REAL *d_Output;
//     REAL *d_Error;
//     REAL *d_Weight;
//     REAL *d_dWeight;
//     cudaError_t cuda_ret;

//     // Allocate device variables

//     int weightSize = (Net->Layer[l]->Units + 1) * (Net->Layer[l-1]->Units + 1) * sizeof(REAL);
    
//     cuda_ret = cudaMalloc(&d_Output, (Net->Layer[l-1]->Units + 1) * sizeof(REAL));
//     if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - AdjustWeights");
//     cuda_ret = cudaMalloc(&d_Error, (Net->Layer[l]->Units + 1) * sizeof(REAL));
//     if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - AdjustWeights");
//     cuda_ret = cudaMalloc(&d_Weight, weightSize);
//     if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - AdjustWeights");
//     cuda_ret = cudaMalloc(&d_dWeight, weightSize);
//     if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - AdjustWeights");
        
//     // Copy host variables to device

//     cuda_ret = cudaMemcpy(d_Output, Net->Layer[l-1]->Output, (Net->Layer[l-1]->Units + 1) * sizeof(REAL), cudaMemcpyHostToDevice);
//     if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - AdjustWeights");
//     cuda_ret = cudaMemcpy(d_Error, Net->Layer[l]->Error, (Net->Layer[l]->Units + 1) * sizeof(REAL), cudaMemcpyHostToDevice);  
//     if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - AdjustWeights");
//     cuda_ret = cudaMemcpy(d_Weight, Net->Layer[l]->Weight[0], weightSize, cudaMemcpyHostToDevice);
//     if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - AdjustWeights");
//     cuda_ret = cudaMemcpy(d_dWeight, Net->Layer[l]->dWeight[0], weightSize, cudaMemcpyHostToDevice);
//     if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - AdjustWeights");  

//     // Launch kernel

//     dim3 DimGrid((Net->Layer[l-1]->Units + BLOCK_SIZE - 1) / BLOCK_SIZE, (Net->Layer[l]->Units + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);
//     adjustWeightsKernel<<<DimGrid, DimBlock>>>(d_Output, d_Error, d_Weight, d_dWeight, Net->Eta, Net->Alpha, Net->Layer[l-1]->Units, Net->Layer[l]->Units);

//     // Copy device variables to host

//     cuda_ret = cudaMemcpy(Net->Layer[l]->Weight[0], d_Weight, weightSize, cudaMemcpyDeviceToHost);
//     if (cuda_ret != cudaSuccess) printf("Unable to copy memory to host - AdjustWeights");
//     cuda_ret = cudaMemcpy(Net->Layer[l]->dWeight[0], d_dWeight, weightSize, cudaMemcpyDeviceToHost);
//     if (cuda_ret != cudaSuccess) printf("Unable to copy memory to host - AdjustWeights");

//     // Free memory

//     cudaFree(d_Output);
//     cudaFree(d_Error);
//     cudaFree(d_Weight);
//     cudaFree(d_dWeight);
//   }
// }

// Parallelization with OpenMP
void AdjustWeights(NET* Net)
{
  INT  l,i,j;
  REAL Out, Err, dWeight;

  #pragma omp parallel for
  for (l=1; l<NUM_LAYERS; l++) {
    
    #pragma omp parallel
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Out = Net->Layer[l-1]->Output[j];
        Err = Net->Layer[l]->Error[i];
        dWeight = Net->Layer[l]->dWeight[i][j];
        Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
        Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
      }
    }
  }
}

// Original C code
// void AdjustWeights(NET* Net)
// {
//   INT  l,i,j;
//   REAL Out, Err, dWeight;

//   for (l=1; l<NUM_LAYERS; l++) {
    
//     for (i=1; i<=Net->Layer[l]->Units; i++) {
//       for (j=0; j<=Net->Layer[l-1]->Units; j++) {
//         Out = Net->Layer[l-1]->Output[j];
//         Err = Net->Layer[l]->Error[i];
//         dWeight = Net->Layer[l]->dWeight[i][j];
//         Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
//         Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
//       }
//     }
//   }
// }