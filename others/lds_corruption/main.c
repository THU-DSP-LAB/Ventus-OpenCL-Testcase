#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const char *label;
  const char *kernel_name;
  size_t local_size;
  size_t groups;
  int rounds;
  unsigned local_arg_count;
  size_t local_arg_bytes[2];
} case_config_t;

typedef struct {
  int bad_items;
  int total_errors;
} case_result_t;

static const size_t kDefaultLocalSize = 64;
static const size_t kDefaultGroups = 4;
static const int kDefaultRounds = 16;
static const size_t kDynamicOnlyBytes = 2048 * sizeof(int);
static const size_t kDynamicPairBytes = 1024 * sizeof(int);
static const size_t kMixedStaticDynamicBytes = 1024 * sizeof(int);
static const size_t kMixedStaticDynamicPairBytes = 512 * sizeof(int);

static void die(const char *what, cl_int err) {
  fprintf(stderr, "%s failed: %d\n", what, err);
  exit(1);
}

static size_t get_env_size_t(const char *name, size_t fallback) {
  const char *value = getenv(name);
  char *end = NULL;
  unsigned long long parsed = 0;

  if (!value || !value[0]) return fallback;
  parsed = strtoull(value, &end, 0);
  if (!end || *end != '\0') {
    fprintf(stderr, "invalid integer for %s: %s\n", name, value);
    exit(1);
  }
  return (size_t)parsed;
}

static int get_env_int(const char *name, int fallback) {
  const char *value = getenv(name);
  char *end = NULL;
  long parsed = 0;

  if (!value || !value[0]) return fallback;
  parsed = strtol(value, &end, 0);
  if (!end || *end != '\0') {
    fprintf(stderr, "invalid integer for %s: %s\n", name, value);
    exit(1);
  }
  return (int)parsed;
}

static int case_selected(const char *label, const char *selector) {
  if (!selector || !selector[0]) return 1;
  return strcmp(label, selector) == 0;
}

static char *read_text_file(const char *path, size_t *size_out) {
  FILE *fp = fopen(path, "rb");
  char *data = NULL;
  long size = 0;
  size_t read_size = 0;

  if (!fp) {
    fprintf(stderr, "fopen failed: %s\n", path);
    exit(1);
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    fprintf(stderr, "fseek end failed: %s\n", path);
    exit(1);
  }
  size = ftell(fp);
  if (size < 0) {
    fclose(fp);
    fprintf(stderr, "ftell failed: %s\n", path);
    exit(1);
  }
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fclose(fp);
    fprintf(stderr, "fseek set failed: %s\n", path);
    exit(1);
  }

  data = malloc((size_t)size + 1);
  if (!data) {
    fclose(fp);
    fprintf(stderr, "malloc failed for %s\n", path);
    exit(1);
  }
  read_size = fread(data, 1, (size_t)size, fp);
  fclose(fp);
  if (read_size != (size_t)size) {
    free(data);
    fprintf(stderr, "fread failed: %s\n", path);
    exit(1);
  }

  data[size] = '\0';
  if (size_out) *size_out = (size_t)size;
  return data;
}

static void print_group_errors(
    const char *label, const int *out, size_t local, size_t groups
) {
  for (size_t group = 0; group < groups; ++group) {
    int group_errors = 0;
    for (size_t i = group * local; i < (group + 1) * local; ++i) {
      group_errors += out[i];
    }
    printf("case=%s group%zu_errors=%d\n", label, group, group_errors);
  }
}

static void set_kernel_args(
    cl_kernel kernel, cl_mem outbuf, const case_config_t *cfg
) {
  cl_int err = CL_SUCCESS;
  unsigned arg_index = 0;

  err = clSetKernelArg(kernel, arg_index++, sizeof(outbuf), &outbuf);
  if (err != CL_SUCCESS) die("clSetKernelArg(out)", err);

  for (unsigned local_index = 0; local_index < cfg->local_arg_count;
       ++local_index) {
    err = clSetKernelArg(
        kernel, arg_index++, cfg->local_arg_bytes[local_index], NULL);
    if (err != CL_SUCCESS) die("clSetKernelArg(local)", err);
  }

  err = clSetKernelArg(kernel, arg_index, sizeof(cfg->rounds), &cfg->rounds);
  if (err != CL_SUCCESS) die("clSetKernelArg(rounds)", err);
}

static case_result_t run_case(
    cl_context context, cl_command_queue queue, cl_program program,
    const case_config_t *cfg
) {
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, cfg->kernel_name, &err);
  size_t global_size = cfg->local_size * cfg->groups;
  case_result_t result = {0, 0};
  cl_mem outbuf = NULL;
  int *out = NULL;

  if (err != CL_SUCCESS) die("clCreateKernel", err);

  out = calloc(global_size, sizeof(int));
  if (!out) {
    fprintf(stderr, "calloc failed for %s\n", cfg->label);
    exit(1);
  }

  outbuf = clCreateBuffer(
      context, CL_MEM_READ_WRITE, sizeof(int) * global_size, NULL, &err);
  if (err != CL_SUCCESS) die("clCreateBuffer", err);

  set_kernel_args(kernel, outbuf, cfg);

  printf(
      "case=%s kernel=%s global=%zu local=%zu groups=%zu rounds=%d "
      "local_arg_count=%u local0=0x%zx local1=0x%zx\n",
      cfg->label, cfg->kernel_name, global_size, cfg->local_size, cfg->groups,
      cfg->rounds, cfg->local_arg_count, cfg->local_arg_bytes[0],
      cfg->local_arg_bytes[1]);

  err = clEnqueueNDRangeKernel(
      queue, kernel, 1, NULL, &global_size, &cfg->local_size, 0, NULL, NULL);
  if (err != CL_SUCCESS) die("clEnqueueNDRangeKernel", err);
  err = clFinish(queue);
  if (err != CL_SUCCESS) die("clFinish", err);

  err = clEnqueueReadBuffer(
      queue, outbuf, CL_TRUE, 0, sizeof(int) * global_size, out, 0, NULL,
      NULL);
  if (err != CL_SUCCESS) die("clEnqueueReadBuffer", err);

  for (size_t i = 0; i < global_size; ++i) {
    result.total_errors += out[i];
    if (out[i] != 0) result.bad_items++;
  }

  printf(
      "case=%s bad_items=%d total_errors=%d\n", cfg->label, result.bad_items,
      result.total_errors);
  print_group_errors(cfg->label, out, cfg->local_size, cfg->groups);

  clReleaseMemObject(outbuf);
  clReleaseKernel(kernel);
  free(out);
  return result;
}

static cl_program build_program_from_file(
    cl_context context, cl_device_id device, const char *source_path
) {
  cl_int err = CL_SUCCESS;
  size_t source_size = 0;
  char *source = read_text_file(source_path, &source_size);
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&source, &source_size, &err);

  if (err != CL_SUCCESS) die("clCreateProgramWithSource", err);

  err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(
        program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 0) {
      char *log = malloc(log_size + 1);
      if (log) {
        clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "%s\n", log);
        free(log);
      }
    }
    die("clBuildProgram", err);
  }

  free(source);
  return program;
}

int main(int argc, char **argv) {
  cl_int err = CL_SUCCESS;
  cl_uint nplatforms = 0;
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  const char *source_path = argc > 1 ? argv[1] : "./kernel.cl";
  const char *selected_case = getenv("LDS_SUITE_CASE");
  const case_config_t cases[] = {
      {"static_only", "lds_stomp_static_only", kDefaultLocalSize,
       kDefaultGroups, kDefaultRounds, 0, {0, 0}},
      {"dynamic_only", "lds_stomp_dynamic_only", kDefaultLocalSize,
       kDefaultGroups, kDefaultRounds, 1, {kDynamicOnlyBytes, 0}},
      {"dynamic_pair", "lds_stomp_dynamic_pair", kDefaultLocalSize,
       kDefaultGroups, kDefaultRounds, 2,
       {kDynamicPairBytes, kDynamicPairBytes}},
      {"mixed_static_dynamic", "lds_stomp_mixed_static_dynamic",
       kDefaultLocalSize, kDefaultGroups, kDefaultRounds, 1,
       {kMixedStaticDynamicBytes, 0}},
      {"mixed_static_dynamic_pair", "lds_stomp_mixed_static_dynamic_pair",
       kDefaultLocalSize, kDefaultGroups, kDefaultRounds, 2,
       {kMixedStaticDynamicPairBytes, kMixedStaticDynamicPairBytes}},
  };
  int suite_failures = 0;

  err = clGetPlatformIDs(1, &platform, &nplatforms);
  if (err != CL_SUCCESS || nplatforms == 0) die("clGetPlatformIDs", err);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  if (err != CL_SUCCESS) die("clGetDeviceIDs", err);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) die("clCreateContext", err);

  {
    const cl_queue_properties props[] = {0};
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
  }
  if (err != CL_SUCCESS) die("clCreateCommandQueueWithProperties", err);

  printf("kernel_source=%s\n", source_path);
  program = build_program_from_file(context, device, source_path);

  for (size_t case_index = 0; case_index < sizeof(cases) / sizeof(cases[0]);
       ++case_index) {
    case_config_t cfg = cases[case_index];
    case_result_t result = {0, 0};

    if (!case_selected(cfg.label, selected_case)) continue;
    cfg.local_size = get_env_size_t("LDS_SUITE_LOCAL", cfg.local_size);
    cfg.groups = get_env_size_t("LDS_SUITE_GROUPS", cfg.groups);
    cfg.rounds = get_env_int("LDS_SUITE_ROUNDS", cfg.rounds);
    result = run_case(context, queue, program, &cfg);
    if (result.total_errors != 0) suite_failures++;
  }

  printf(
      "suite_result=%s failing_cases=%d total_cases=%zu\n",
      suite_failures == 0 ? "PASS" : "FAIL", suite_failures,
      sizeof(cases) / sizeof(cases[0]));

  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return suite_failures == 0 ? 0 : 1;
}
