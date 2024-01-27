/******************************************************************************
                          I N I T I A L I Z A T I O N
 ******************************************************************************/

void GenerateNetwork(NET* Net)
{
  INT l,i;

  Net->Layer = (LAYER**) calloc(NUM_LAYERS, sizeof(LAYER*));
   
  for (l=0; l<NUM_LAYERS; l++) {
    Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));
      
    Net->Layer[l]->Units      = Units[l];
    Net->Layer[l]->Output     = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    Net->Layer[l]->Error      = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    Net->Layer[l]->Weight     = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    Net->Layer[l]->WeightSave = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    Net->Layer[l]->dWeight    = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    Net->Layer[l]->Output[0]  = BIAS;
      
    if (l != 0) {
      for (i=1; i<=Units[l]; i++) {
        Net->Layer[l]->Weight[i]     = (REAL*) calloc(Units[l-1]+1, sizeof(REAL));
        Net->Layer[l]->WeightSave[i] = (REAL*) calloc(Units[l-1]+1, sizeof(REAL));
        Net->Layer[l]->dWeight[i]    = (REAL*) calloc(Units[l-1]+1, sizeof(REAL));
      }
    }
  }
  Net->InputLayer  = Net->Layer[0];
  Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
  Net->Alpha       = 0.9;
  Net->Eta         = 0.25;
  Net->Gain        = 1;
}

void RandomWeights(NET* Net)
{
  INT l,i,j;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[i][j] = RandomEqualREAL(-0.5, 0.5);
      }
    }
  }
}

// Parallelization with CUDA
__global__ void SetInputKernel(INT units, REAL *output, REAL *input) {
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  if (i <= units) {
    output[i] = input[i-1];
  }
}

void SetInput(NET *Net, REAL *Input) {
  
  // Initialize host variables

  REAL *output;
  REAL *input;
  cudaError_t cuda_ret;

  // Allocate device variables

  cuda_ret = cudaMalloc(&output, (Net->InputLayer->Units + 1) * sizeof(REAL));
  if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - SetInput");
  cuda_ret = cudaMalloc(&input, (Net->InputLayer->Units + 1) * sizeof(REAL));
  if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - SetInput");

  // Copy host variables to device

  cuda_ret = cudaMemcpy(output, Net->InputLayer->Output, (Net->InputLayer->Units + 1) * sizeof(REAL), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - SetInput");
  cuda_ret = cudaMemcpy(input, Input, (Net->InputLayer->Units + 1) * sizeof(REAL), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - SetInput");

  // Launch kernel

  dim3 DimGrid((Net->InputLayer->Units + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  SetInputKernel<<<DimGrid, DimBlock>>>(Net->InputLayer->Units, output, input);

  // Copy device variables to host

  cuda_ret = cudaMemcpy(Net->InputLayer->Output, output, (Net->InputLayer->Units + 1) * sizeof(REAL), cudaMemcpyDeviceToHost);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to host - SetInput");

  // Free memory

  cudaFree(output);
  cudaFree(input);
}

// Original C code
// void SetInput(NET* Net, REAL* Input)
// {
//   INT i;
   
//   for (i=1; i<=Net->InputLayer->Units; i++) {
//     Net->InputLayer->Output[i] = Input[i-1];
//   }
// }

// Parallelization with CUDA
__global__ void GetOutputKernel(INT units, REAL *output, REAL *OutputLayerOutput) {
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  if (i <= units) {
    output[i-1] = OutputLayerOutput[i];
  }
}

void GetOutput(NET* Net, REAL* Output) {

  // Initialize host variables

  REAL *output;
  REAL *OutputLayerOutput;
  cudaError_t cuda_ret;

  // Allocate device variables

  cuda_ret = cudaMalloc(&output, (Net->OutputLayer->Units + 1) * sizeof(REAL));
  if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - GetOutput");
  cuda_ret = cudaMalloc(&OutputLayerOutput, (Net->OutputLayer->Units + 1) * sizeof(REAL));
  if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory - GetOutput");

  // Copy host variables to device

  cuda_ret = cudaMemcpy(output, Output, (Net->OutputLayer->Units + 1) * sizeof(REAL), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - GetOutput");
  cuda_ret = cudaMemcpy(OutputLayerOutput, Net->OutputLayer->Output, (Net->OutputLayer->Units + 1) * sizeof(REAL), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device - GetOutput");

  // Launch kernel

  dim3 DimGrid((Net->InputLayer->Units + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  GetOutputKernel<<<DimGrid, DimBlock>>>(Net->OutputLayer->Units, output, OutputLayerOutput);

  // Copy device variables to host

  cuda_ret = cudaMemcpy(Output, output, (Net->OutputLayer->Units + 1) * sizeof(REAL), cudaMemcpyDeviceToHost);
  if (cuda_ret != cudaSuccess) printf("Unable to copy memory to host - GetOutput");

  // Free memory

  cudaFree(output);
  cudaFree(OutputLayerOutput);
}

// Original C code
// void GetOutput(NET* Net, REAL* Output)
// {
//   INT i;
   
//   for (i=1; i<=Net->OutputLayer->Units; i++) {
//     Output[i-1] = Net->OutputLayer->Output[i];
//   }
// }