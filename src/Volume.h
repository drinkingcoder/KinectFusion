#pragma once

#include <iostream>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

#include "Utility/cutil_math.h"
#include "Utility/CudaToolKit.h"

/// Volume contains two types of position representation:
///     1. Normalized position [0, 1]
///     2. Size-range positoin [0, size]
/// TSDF in Volume also can be representated by two value:
///     1. Normalized value [-1.0f, 1.0f]
///     2. Short int value [-32768, 32767]
///     We need to process it in float but store it in float???
class Volume {
public:
    Volume():m_size(make_uint3(0)),
            m_dim(make_float3(1)),
            m_data(NULL){
    }

    __device__ float2 operator[](const uint3 &pos) const {
        const short2 d = m_data[pos.x + pos.y * m_size.x + pos.z * m_size.x * m_size.y];
        return make_float2(d.x * 0.00003051944088f, d.y); // 1/32766
    }

    __device__ float v(const uint3 &pos) const {
        return operator[](pos).x;
    }

    __device__ float vs(const uint3 &pos) const {
        return m_data[pos.x + pos.y * m_size.x + pos.z * m_size.x * m_size.y].x;
    }

    __device__ void set(const uint3 &pos, const float2 &d) {
        m_data[pos.x + pos.y * m_size.x + pos.z * m_size.x * m_size.y] = make_short2(d.x * 32766.0f, d.y);
    }

    ///transfer unit position to the volume space position
    __device__ float3 pos(const uint3 & p) {
        return make_float3(
                (p.x + 0.5f) * m_dim.x / m_size.x,
                (p.y + 0.5f) * m_dim.y / m_size.y,
                (p.z + 0.5f) * m_dim.z / m_size.z
                );
    }
    /// linear interpolation
    __device__ float interp(const float3 &pos) const {
        const float3 scaled_pos = make_float3(
                (pos.x * m_size.x / m_dim.x) - 0.5f,
                (pos.y * m_size.y / m_dim.y) - 0.5f,
                (pos.z * m_size.z / m_dim.z) - 0.5f
                );
        const int3 base = make_int3(floorf(scaled_pos));
        const float3 factor = fracf(scaled_pos);
        const int3 lower = max(base, make_int3(0));
        const int3 upper = min(base + make_int3(1), make_int3(m_size) - make_int3(1));
        return (
              ((vs(make_uint3(lower.x, lower.y, lower.z)) * (1-factor.x) + vs(make_uint3(upper.x, lower.y, lower.z)) * factor.x) * (1-factor.y)
             + (vs(make_uint3(lower.x, upper.y, lower.z)) * (1-factor.x) + vs(make_uint3(upper.x, upper.y, lower.z)) * factor.x) * factor.y) * (1-factor.z)
            + ((vs(make_uint3(lower.x, lower.y, upper.z)) * (1-factor.x) + vs(make_uint3(upper.x, lower.y, upper.z)) * factor.x) * (1-factor.y)
             + (vs(make_uint3(lower.x, upper.y, upper.z)) * (1-factor.x) + vs(make_uint3(upper.x, upper.y, upper.z)) * factor.x) * factor.y) * factor.z
            ) * 0.00003051944088f;
    }

    __device__ float3 grad( const float3 & pos ) const {
        const float3 scaled_pos = make_float3((pos.x * m_size.x / m_dim.x) - 0.5f, (pos.y * m_size.y / m_dim.y) - 0.5f, (pos.z * m_size.z / m_dim.z) - 0.5f);
        const int3 base = make_int3(floorf(scaled_pos));
        const float3 factor = fracf(scaled_pos);
        const int3 lower_lower = max(base - make_int3(1), make_int3(0));
        const int3 lower_upper = max(base, make_int3(0));
        const int3 upper_lower = min(base + make_int3(1), make_int3(m_size) - make_int3(1));
        const int3 upper_upper = min(base + make_int3(2), make_int3(m_size) - make_int3(1));
        const int3 & lower = lower_upper;
        const int3 & upper = upper_lower;

        float3 gradient;

        gradient.x =
              (((vs(make_uint3(upper_lower.x, lower.y, lower.z)) - vs(make_uint3(lower_lower.x, lower.y, lower.z))) * (1-factor.x)
            + (vs(make_uint3(upper_upper.x, lower.y, lower.z)) - vs(make_uint3(lower_upper.x, lower.y, lower.z))) * factor.x) * (1-factor.y)
            + ((vs(make_uint3(upper_lower.x, upper.y, lower.z)) - vs(make_uint3(lower_lower.x, upper.y, lower.z))) * (1-factor.x)
            + (vs(make_uint3(upper_upper.x, upper.y, lower.z)) - vs(make_uint3(lower_upper.x, upper.y, lower.z))) * factor.x) * factor.y) * (1-factor.z)
            + (((vs(make_uint3(upper_lower.x, lower.y, upper.z)) - vs(make_uint3(lower_lower.x, lower.y, upper.z))) * (1-factor.x)
            + (vs(make_uint3(upper_upper.x, lower.y, upper.z)) - vs(make_uint3(lower_upper.x, lower.y, upper.z))) * factor.x) * (1-factor.y)
            + ((vs(make_uint3(upper_lower.x, upper.y, upper.z)) - vs(make_uint3(lower_lower.x, upper.y, upper.z))) * (1-factor.x)
            + (vs(make_uint3(upper_upper.x, upper.y, upper.z)) - vs(make_uint3(lower_upper.x, upper.y, upper.z))) * factor.x) * factor.y) * factor.z;

        gradient.y =
              (((vs(make_uint3(lower.x, upper_lower.y, lower.z)) - vs(make_uint3(lower.x, lower_lower.y, lower.z))) * (1-factor.x)
            + (vs(make_uint3(upper.x, upper_lower.y, lower.z)) - vs(make_uint3(upper.x, lower_lower.y, lower.z))) * factor.x) * (1-factor.y)
            + ((vs(make_uint3(lower.x, upper_upper.y, lower.z)) - vs(make_uint3(lower.x, lower_upper.y, lower.z))) * (1-factor.x)
            + (vs(make_uint3(upper.x, upper_upper.y, lower.z)) - vs(make_uint3(upper.x, lower_upper.y, lower.z))) * factor.x) * factor.y) * (1-factor.z)
            + (((vs(make_uint3(lower.x, upper_lower.y, upper.z)) - vs(make_uint3(lower.x, lower_lower.y, upper.z))) * (1-factor.x)
            + (vs(make_uint3(upper.x, upper_lower.y, upper.z)) - vs(make_uint3(upper.x, lower_lower.y, upper.z))) * factor.x) * (1-factor.y)
            + ((vs(make_uint3(lower.x, upper_upper.y, upper.z)) - vs(make_uint3(lower.x, lower_upper.y, upper.z))) * (1-factor.x)
            + (vs(make_uint3(upper.x, upper_upper.y, upper.z)) - vs(make_uint3(upper.x, lower_upper.y, upper.z))) * factor.x) * factor.y) * factor.z;

        gradient.z =
              (((vs(make_uint3(lower.x, lower.y, upper_lower.z)) - vs(make_uint3(lower.x, lower.y, lower_lower.z))) * (1-factor.x)
            + (vs(make_uint3(upper.x, lower.y, upper_lower.z)) - vs(make_uint3(upper.x, lower.y, lower_lower.z))) * factor.x) * (1-factor.y)
            + ((vs(make_uint3(lower.x, upper.y, upper_lower.z)) - vs(make_uint3(lower.x, upper.y, lower_lower.z))) * (1-factor.x)
            + (vs(make_uint3(upper.x, upper.y, upper_lower.z)) - vs(make_uint3(upper.x, upper.y, lower_lower.z))) * factor.x) * factor.y) * (1-factor.z)
            + (((vs(make_uint3(lower.x, lower.y, upper_upper.z)) - vs(make_uint3(lower.x, lower.y, lower_upper.z))) * (1-factor.x)
            + (vs(make_uint3(upper.x, lower.y, upper_upper.z)) - vs(make_uint3(upper.x, lower.y, lower_upper.z))) * factor.x) * (1-factor.y)
            + ((vs(make_uint3(lower.x, upper.y, upper_upper.z)) - vs(make_uint3(lower.x, upper.y, lower_upper.z))) * (1-factor.x)
            + (vs(make_uint3(upper.x, upper.y, upper_upper.z)) - vs(make_uint3(upper.x, upper.y, lower_upper.z))) * factor.x) * factor.y) * factor.z;

        return gradient * make_float3(m_dim.x/m_size.x, m_dim.y/m_size.y, m_dim.z/m_size.z) * (0.5f * 0.00003051944088f);
    }
    /// Set the whole volume with initial value(1.0f).
    void Reset();

    void Init(uint3 size, float3 d) {
        m_size = size;
        m_dim = d;
        std::cout << "Volume size = (" << m_size.x << ", " << m_size.y << ")" << std::endl;
        std::cout << "Volume dimension = (" << d.x << ", " << d.y << ")" << std::endl;
        cudaMalloc(&m_data, m_size.x * m_size.y * m_size.z * sizeof(short2));
        Reset();
    }

    void SetVolumeWrap(const float val);

    void SetBoxWrap(const float3 min_corner, const float3 max_corner, const float val);

    void PrintInnerVoxels();

    void release() {
        cudaFree(m_data);
        m_data = NULL;
    }

public:
    uint3 m_size;
    float3 m_dim;
    short2 * m_data;
};
