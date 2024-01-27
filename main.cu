/******************************************************************************
                                    M A I N
 ******************************************************************************/
#include "support.h"
#include "random.cu"
#include "app.cu"
#include "init.cu"
#include "weight.cu"
#include "propagate.cu"
#include "back.cu"
#include "sim.cu"

int main()
{
  Timer timer;
  NET  Net;
  BOOL Stop;
  REAL MinTestError;

  startTime(&timer); // record the start time

  InitializeRandoms();
  GenerateNetwork(&Net);
  RandomWeights(&Net);
  InitializeApplication(&Net);

  Stop = FALSE;
  MinTestError = MAX_REAL;
  do {
    TrainNet(&Net, 10);
    TestNet(&Net);
    if (TestError < MinTestError) {
      fprintf(f, " - saving Weights ...");
      MinTestError = TestError;
      SaveWeights(&Net);
    }
    else if (TestError > 1.2 * MinTestError) {
      fprintf(f, " - stopping Training and restoring Weights ...");
      Stop = TRUE;
      RestoreWeights(&Net);
    }
  } while (NOT Stop);

  TestNet(&Net);
  EvaluateNet(&Net);
  FinalizeApplication(&Net);

  stopTime(&timer); printf("Execution time main: %f s\n", elapsedTime(timer)); // record and print execution time

  return 0 ;
}