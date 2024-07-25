#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "poclu.h"

#define ARRAY_SIZE 32

#ifdef __cplusplus
#  define CALLAPI "C"
#else
#  define CALLAPI
#endif

extern CALLAPI int
exec_mma_kernel (cl_context context, cl_device_id device,
                    cl_command_queue cmd_queue, cl_program program,
                    int n, int wg_size, cl_float *srcA, cl_float *srcB,
                    cl_float *dst);

int
main (int argc, char **argv)
{
  cl_float *srcA, *srcB;
  cl_float *dst;
  int i, err;

  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_platform_id platform = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;

  err = poclu_get_any_device2 (&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN ("clCreateContext");

  const char *basename = "mma";
  err = poclu_load_program (context, device, basename, 0, 0, 0, NULL, NULL,
                            &program);
  if (err != CL_SUCCESS)
    goto FINISH;

  int vec_width = ARRAY_SIZE;
  int wg_size = ARRAY_SIZE;

  srcA = (cl_float *) malloc (vec_width * sizeof (cl_float));
  srcB = (cl_float *) malloc (vec_width * sizeof (cl_float));
  dst = (cl_float *) malloc (vec_width * sizeof (cl_float));

  for (i = 0; i < vec_width; ++i)
    {
      srcA[i] = (cl_float)i;
      srcB[i] = (cl_float)(vec_width - i);
      dst[i] = (cl_float)i;
    }

  err = 0;

  if (exec_mma_kernel (context, device, queue, program, vec_width,
                          wg_size, srcA, srcB, dst))
    {
      printf ("Error running the tests\n");
      err = 1;
      goto FINISH;
    }

  for (i = 0; i < vec_width; ++i)
    {
      if ((int)srcA[i] + (int)srcB[i] != (int)dst[i])
        {
          printf ("%d FAIL: %f + %f != %f\n", i, srcA[i], srcB[i], dst[i]);
          err = 1;
          goto FINISH;
        }
    }
  free (srcA);
  free (srcB);
  free (dst);

  printf ("OK\n");

FINISH:
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  return err;
}
