/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

template <typename T>
inline __device__ __host__ T padUp(T m, T n)
{
    return divUp(m, n) * n;
}

inline __host__ __device__ int64_t nextPowerOf2(int64_t x)
{
    if (x < 1)
        return 1;
    int64_t n = x - 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

inline __host__ __device__ int64_t lastPowerOf2(int64_t x)
{
    int64_t next = nextPowerOf2(x);
    if (next == x)
        return next;
    return next / 2;
}

inline int64_t nearestInBuckets(int64_t x, std::vector<int64_t> const& buckets)
{
    if (buckets.empty())
        return x;
    int64_t clamped = std::max(nextPowerOf2(x), buckets.front());
    clamped = std::min(clamped, buckets.back());
    return clamped;
}

inline std::vector<int64_t> getPowerOf2NumTokensBuckets(int64_t maxNumTokens)
{
    std::vector<int64_t> buckets;
    int64_t m = nextPowerOf2(maxNumTokens);
    while (m >= 1)
    {
        buckets.push_back(m);
        m /= 2;
    }
    std::reverse(buckets.begin(), buckets.end());
    return buckets;
}

inline std::vector<int64_t> getLastPowerOf2NumTokensBuckets(int64_t maxNumTokens)
{
    std::vector<int64_t> buckets;
    int64_t m = lastPowerOf2(maxNumTokens);
    while (m >= 1)
    {
        buckets.push_back(m);
        m /= 2;
    }
    std::reverse(buckets.begin(), buckets.end());
    return buckets;
}

inline std::vector<int64_t> deepGemmGenTuningBuckets(int64_t x)
{
    std::vector<int64_t> buckets;
    for (int64_t i = 8; i < 128; i += 8)
    {
        buckets.push_back(i);
    }
    if (x >= 128)
    {
        x = std::min(x, static_cast<int64_t>(8192));
        x = std::max(x, static_cast<int64_t>(4096));
        for (int64_t i = 128; i < x; i += 128)
        {
            buckets.push_back(i);
        }
    }
    return buckets;
}

inline std::pair<int64_t, int64_t> computeSwizzledSfShape(int64_t row, int64_t col)
{
    return {padUp(row, static_cast<int64_t>(128)), padUp(col, static_cast<int64_t>(4))};
}

inline int64_t dimToTrtAxes(std::vector<int64_t> const& dims)
{
    int64_t axes = 0;
    for (auto d : dims)
    {
        axes |= (int64_t{1} << d);
    }
    return axes;
}

inline std::vector<int64_t> trtAxesToDim(int64_t axes)
{
    std::vector<int64_t> dims;
    for (int64_t i = 0; i < 32; ++i)
    {
        if (axes & (int64_t{1} << i))
        {
            dims.push_back(i);
        }
    }
    return dims;
}

inline std::vector<int64_t> dimResolveNegative(std::vector<int64_t> const& dims, int64_t ndim)
{
    std::vector<int64_t> result;
    result.reserve(dims.size());
    for (auto d : dims)
    {
        if (d < 0)
        {
            d = ndim + d;
        }
        result.push_back(d);
    }
    return result;
}

inline int64_t fp8ScaleInferShape(std::vector<std::vector<int64_t>> const& inputShapes)
{
    auto const& inputShape = inputShapes[0];
    bool hasBatch = inputShape.size() == 3;
    int64_t m = inputShape[inputShape.size() - 2];
    return hasBatch ? padUp(m, int64_t{4}) : m;
}

} // namespace common

TRTLLM_NAMESPACE_END
