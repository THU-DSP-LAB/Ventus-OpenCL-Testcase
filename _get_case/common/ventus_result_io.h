#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <cstdlib>

// 将 float 转为 "hxxxxxxxx" 十六进制（便于跨后端一致、精确可比）
inline std::string ventus_float_to_hex(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(uint32_t));
    std::ostringstream oss;
    oss << 'h' << std::hex << u;
    return oss.str();
}

// 取得结果文件路径（优先环境变量 VENTUS_RESULT_FILE，缺省 "result.hex"）
inline std::string ventus_result_path() {
    const char* p = std::getenv("VENTUS_RESULT_FILE");
    if (p && *p) return std::string(p);
    return "result.hex";
}

// 只写“最终数据”的工具：一行保存为 hex，空格分隔。不会往 stdout 打任何东西。
inline void ventus_write_final_hex(const std::vector<float>& data) {
    std::string path = ventus_result_path();
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        // 即使失败也避免影响后续流程：不抛异常、不打印大量日志
        return;
    }
    for (size_t i = 0; i < data.size(); ++i) {
        ofs << ventus_float_to_hex(data[i]);
        if (i + 1 != data.size()) ofs << ' ';
    }
    ofs << '\n';
    ofs.close();
}

inline void ventus_write_final_hex(const float* data, size_t n) {
    if (!data || n == 0) return;
    std::string path = ventus_result_path();
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return;
    for (size_t i = 0; i < n; ++i) {
        ofs << ventus_float_to_hex(data[i]);
        if (i + 1 != n) ofs << ' ';
    }
    ofs << '\n';
    ofs.close();
}


// 追加：C 数组便捷重载
template <size_t N>
inline void ventus_write_final_hex(const float (&arr)[N]) {
    ventus_write_final_hex(arr, N);
}