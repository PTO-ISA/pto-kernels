#ifndef BENCH_UTILS_HPP
#define BENCH_UTILS_HPP

#include <cstdint>
#include <cmath>
#include <type_traits>

#ifdef __linx__
inline __half linx_half_fixture_value(unsigned value) {
    value &= 15u;
    if (value == 0)
        return __builtin_bit_cast(__half, static_cast<uint16_t>(0));
    const unsigned exponent = 31u - __builtin_clz(value);
    const uint16_t bits = static_cast<uint16_t>(
        ((exponent + 15u) << 10) |
        ((value - (1u << exponent)) << (10u - exponent)));
    return __builtin_bit_cast(__half, bits);
}
#endif

template <typename T>
void fill_seq(T *p, int n, T base = (T)0) {
#ifdef __linx__
    if constexpr (std::is_same_v<T, __half>) {
        (void)base;
        for (int i = 0; i < n; ++i)
            p[i] = linx_half_fixture_value(static_cast<unsigned>(i));
        return;
    }
#endif
    for (int i = 0; i < n; ++i) {
        p[i] = (T)(base + (T)i * (T)0.1);
    }
}

template <typename T>
void fill_const(T *p, int n, T v) {
    for (int i = 0; i < n; ++i) p[i] = v;
}

template <typename T>
void zero(T *p, int n) { fill_const(p, n, (T)0); }

template <typename T>
void fill_idx(T *p, int n, T base = (T)0) {
    for (int i = 0; i < n; ++i) p[i] = (T)((i * 7) % n) + base;
}

template <typename T>
bool verify(const T *got, const T *ref, int n, T eps = (T)1e-3) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs((double)(got[i] - ref[i])) > (double)eps) return false;
    }
    return true;
}

#endif
