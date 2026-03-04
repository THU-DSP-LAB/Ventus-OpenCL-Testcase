#pragma once

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "../common/ventus_result_io.h"

struct RegressionOptions {
    double atol;
    double rtol;
    std::string cpu_ref_path;
};

struct RegressionStats {
    bool size_match = true;
    int compared = 0;
    int mismatch = 0;
    int worst_index = -1;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double worst_tol = 0.0;
    float worst_ref = 0.0f;
    float worst_got = 0.0f;
};

struct CpuConvConfig {
    int IC, IH, IW;
    int OC, KH, KW;
    int OH, OW;
    int SH, SW, PH, PW;
};

struct CpuPoolConfig {
    int C, IH, IW;
    int KH, KW, SH, SW, PH, PW;
    int OH, OW;
    int count_include_pad;
};

struct CpuGemmConfig {
    int M, K, N;
    int do_trans_a, do_trans_b;
    bool do_relu;
};

inline void print_usage(const char* prog) {
    std::printf("Usage: %s [--atol <value>] [--rtol <value>] [--cpu-ref <path>]\n", prog);
}

inline bool parse_non_negative_double(const char* text, double* out) {
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value) || value < 0.0) {
        return false;
    }
    *out = value;
    return true;
}

inline RegressionOptions parse_options_or_exit(int argc, char** argv, double default_atol, double default_rtol,
                                               const char* default_cpu_ref_path) {
    RegressionOptions options{default_atol, default_rtol, default_cpu_ref_path};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--atol" || arg == "--rtol" || arg == "--cpu-ref") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "缺少参数值: %s\n", arg.c_str());
                std::exit(1);
            }
            const char* value = argv[++i];
            if (arg == "--cpu-ref") {
                options.cpu_ref_path = value;
                continue;
            }
            double parsed = 0.0;
            if (!parse_non_negative_double(value, &parsed)) {
                std::fprintf(stderr, "非法浮点参数: %s=%s\n", arg.c_str(), value);
                std::exit(1);
            }
            if (arg == "--atol") options.atol = parsed;
            else options.rtol = parsed;
            continue;
        }
        std::fprintf(stderr, "未知参数: %s\n", arg.c_str());
        print_usage(argv[0]);
        std::exit(1);
    }
    return options;
}

inline void require_size(const std::vector<float>& data, size_t expected, const char* name) {
    if (data.size() == expected) return;
    std::fprintf(stderr, "%s size mismatch: got=%zu expected=%zu\n", name, data.size(), expected);
    std::exit(1);
}

inline void conv2d_relu_cpu(const std::vector<float>& in, const std::vector<float>& w,
                            const CpuConvConfig& cfg, std::vector<float>* out) {
    const size_t expected_in = (size_t)cfg.IC * cfg.IH * cfg.IW;
    const size_t expected_w = (size_t)cfg.OC * cfg.IC * cfg.KH * cfg.KW;
    require_size(in, expected_in, "conv input");
    require_size(w, expected_w, "conv weight");
    out->assign((size_t)cfg.OC * cfg.OH * cfg.OW, 0.0f);
    const int in_hw = cfg.IH * cfg.IW;
    const int out_hw = cfg.OH * cfg.OW;
    for (int oc = 0; oc < cfg.OC; ++oc) {
        for (int oh = 0; oh < cfg.OH; ++oh) {
            for (int ow = 0; ow < cfg.OW; ++ow) {
                float acc = 0.0f;
                for (int ic = 0; ic < cfg.IC; ++ic) {
                    for (int kh = 0; kh < cfg.KH; ++kh) {
                        const int ih = oh * cfg.SH - cfg.PH + kh;
                        if (ih < 0 || ih >= cfg.IH) continue;
                        for (int kw = 0; kw < cfg.KW; ++kw) {
                            const int iw = ow * cfg.SW - cfg.PW + kw;
                            if (iw < 0 || iw >= cfg.IW) continue;
                            const int in_idx = ic * in_hw + ih * cfg.IW + iw;
                            const int w_idx = ((oc * cfg.IC + ic) * cfg.KH + kh) * cfg.KW + kw;
                            acc += in[in_idx] * w[w_idx];
                        }
                    }
                }
                if (acc < 0.0f) acc = 0.0f;
                (*out)[oc * out_hw + oh * cfg.OW + ow] = acc;
            }
        }
    }
}

inline void avg_pool2d_cpu(const std::vector<float>& in, const CpuPoolConfig& cfg,
                           std::vector<float>* out) {
    const size_t expected_in = (size_t)cfg.C * cfg.IH * cfg.IW;
    require_size(in, expected_in, "pool input");
    out->assign((size_t)cfg.C * cfg.OH * cfg.OW, 0.0f);
    const int in_hw = cfg.IH * cfg.IW;
    const int out_hw = cfg.OH * cfg.OW;
    for (int c = 0; c < cfg.C; ++c) {
        for (int oy = 0; oy < cfg.OH; ++oy) {
            for (int ox = 0; ox < cfg.OW; ++ox) {
                float acc = 0.0f;
                int count = 0;
                for (int ky = 0; ky < cfg.KH; ++ky) {
                    const int iy = oy * cfg.SH - cfg.PH + ky;
                    for (int kx = 0; kx < cfg.KW; ++kx) {
                        const int ix = ox * cfg.SW - cfg.PW + kx;
                        const bool valid = (iy >= 0 && iy < cfg.IH && ix >= 0 && ix < cfg.IW);
                        if (!cfg.count_include_pad && !valid) continue;
                        if (valid) acc += in[c * in_hw + iy * cfg.IW + ix];
                        ++count;
                    }
                }
                (*out)[c * out_hw + oy * cfg.OW + ox] = (count > 0) ? (acc / (float)count) : 0.0f;
            }
        }
    }
}

inline void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B,
                     const CpuGemmConfig& cfg, std::vector<float>* C) {
    const size_t expected_a = (size_t)cfg.M * cfg.K;
    const size_t expected_b = (size_t)cfg.K * cfg.N;
    require_size(A, expected_a, "gemm A");
    require_size(B, expected_b, "gemm B");
    C->assign((size_t)cfg.M * cfg.N, 0.0f);
    for (int m = 0; m < cfg.M; ++m) {
        for (int n = 0; n < cfg.N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < cfg.K; ++k) {
                const int a_idx = cfg.do_trans_a ? (k * cfg.M + m) : (m * cfg.K + k);
                const int b_idx = cfg.do_trans_b ? (n * cfg.K + k) : (k * cfg.N + n);
                acc += A[a_idx] * B[b_idx];
            }
            (*C)[m * cfg.N + n] = cfg.do_relu ? fmaxf(0.0f, acc) : acc;
        }
    }
}

inline bool exceeds_tolerance(float ref, float got, double rtol, double atol,
                              double* abs_err, double* rel_err, double* tol) {
    const double abs_err_local = std::fabs((double)got - (double)ref);
    const double rel_err_local = abs_err_local / (std::fabs((double)ref) + 1e-12);
    const double tol_local = atol + rtol * std::fabs((double)ref);
    if (abs_err) *abs_err = abs_err_local;
    if (rel_err) *rel_err = rel_err_local;
    if (tol) *tol = tol_local;
    return abs_err_local > tol_local;
}

inline RegressionStats compare_outputs(const std::vector<float>& ref, const std::vector<float>& got,
                                       double rtol, double atol) {
    RegressionStats stats;
    stats.size_match = (ref.size() == got.size());
    const int n = (int)((ref.size() < got.size()) ? ref.size() : got.size());
    stats.compared = n;
    for (int i = 0; i < n; ++i) {
        double abs_err = 0.0;
        double rel_err = 0.0;
        double tol = 0.0;
        if (exceeds_tolerance(ref[i], got[i], rtol, atol, &abs_err, &rel_err, &tol)) ++stats.mismatch;
        if (abs_err >= stats.max_abs_err) {
            stats.max_abs_err = abs_err;
            stats.max_rel_err = rel_err;
            stats.worst_index = i;
            stats.worst_tol = tol;
            stats.worst_ref = ref[i];
            stats.worst_got = got[i];
        }
    }
    if (!stats.size_match) stats.mismatch += std::abs((int)ref.size() - (int)got.size());
    return stats;
}

inline bool write_hex_file(const std::vector<float>& data, const std::string& path) {
    std::ofstream ofs(path.c_str(), std::ios::binary);
    if (!ofs.is_open()) {
        std::fprintf(stderr, "无法写入文件: %s\n", path.c_str());
        return false;
    }
    for (size_t i = 0; i < data.size(); ++i) {
        ofs << ventus_float_to_hex(data[i]);
        if (i + 1 != data.size()) ofs << ' ';
    }
    ofs << '\n';
    if (!ofs.good()) {
        std::fprintf(stderr, "写入文件失败: %s\n", path.c_str());
        return false;
    }
    return true;
}
