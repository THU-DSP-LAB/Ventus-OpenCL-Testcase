static inline int signed_mix(int tag, int group, int lid, int round) {
  return tag + group * 100000 + lid * 257 + round;
}

static inline int static_window_errors(
    __local volatile int *buf, int elems_per_wg, int tag, int group, int lid,
    int rounds
) {
  const int idx = lid * (elems_per_wg / 64);
  int errors = 0;

  for (int round = 0; round < rounds; ++round) {
    const int expect = signed_mix(tag, group, lid, round);
    buf[idx] = expect;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (buf[idx] != expect) errors++;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  return errors;
}

static inline int dynamic_pair_errors(
    __local volatile int *buf_a, int elems_a, int tag_a,
    __local volatile int *buf_b, int elems_b, int tag_b,
    int group, int lid, int rounds
) {
  const int idx_a = lid * (elems_a / 64);
  const int idx_b = lid * (elems_b / 64);
  int errors = 0;

  for (int round = 0; round < rounds; ++round) {
    const int expect_a = signed_mix(tag_a, group, lid, round);
    const int expect_b = signed_mix(tag_b, group, lid, round);
    buf_a[idx_a] = expect_a;
    buf_b[idx_b] = expect_b;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (buf_a[idx_a] != expect_a) errors++;
    if (buf_b[idx_b] != expect_b) errors++;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  return errors;
}

__kernel void lds_stomp_static_only(__global int *out, int rounds) {
  __local volatile int buf[2048];
  const int gid = (int)get_global_id(0);
  const int lid = (int)get_local_id(0);
  const int group = (int)get_group_id(0);
  out[gid] = static_window_errors(buf, 2048, 0x21000000, group, lid, rounds);
}

__kernel void lds_stomp_dynamic_only(
    __global int *out, __local volatile int *buf, int rounds
) {
  const int gid = (int)get_global_id(0);
  const int lid = (int)get_local_id(0);
  const int group = (int)get_group_id(0);
  out[gid] = static_window_errors(buf, 2048, 0x31000000, group, lid, rounds);
}

__kernel void lds_stomp_dynamic_pair(
    __global int *out, __local volatile int *buf_a, __local volatile int *buf_b,
    int rounds
) {
  const int gid = (int)get_global_id(0);
  const int lid = (int)get_local_id(0);
  const int group = (int)get_group_id(0);
  out[gid] = dynamic_pair_errors(
      buf_a, 1024, 0x41000000, buf_b, 1024, 0x42000000, group, lid, rounds);
}

__kernel void lds_stomp_mixed_static_dynamic(
    __global int *out, __local volatile int *dyn_buf, int rounds
) {
  __local volatile int static_buf[1024];
  const int gid = (int)get_global_id(0);
  const int lid = (int)get_local_id(0);
  const int group = (int)get_group_id(0);
  int errors = 0;

  errors += static_window_errors(
      static_buf, 1024, 0x61000000, group, lid, rounds);
  errors += static_window_errors(
      dyn_buf, 1024, 0x62000000, group, lid, rounds);

  out[gid] = errors;
}

__kernel void lds_stomp_mixed_static_dynamic_pair(
    __global int *out, __local volatile int *buf_a, __local volatile int *buf_b,
    int rounds
) {
  __local volatile int static_buf[1024];
  const int gid = (int)get_global_id(0);
  const int lid = (int)get_local_id(0);
  const int group = (int)get_group_id(0);
  int errors = 0;

  errors += static_window_errors(
      static_buf, 1024, 0x71000000, group, lid, rounds);
  errors += dynamic_pair_errors(
      buf_a, 512, 0x72000000, buf_b, 512, 0x73000000, group, lid, rounds);

  out[gid] = errors;
}
