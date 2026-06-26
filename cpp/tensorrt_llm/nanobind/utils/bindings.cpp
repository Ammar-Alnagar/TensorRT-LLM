/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bindings.h"

#include "tensorrt_llm/common/mathUtils.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace tensorrt_llm::nanobind::utils
{

void initBindings(nb::module_& m)
{
    // Version string - replaces Python version.py
    m.attr("__version__") = nb::str("1.3.0rc20");

    // Math utility functions
    // --- from tensorrt_llm/math_utils.py ---
    m.def("ceil_div", [](int64_t x, int64_t y) -> int64_t
    {
        if (y == 0) throw std::runtime_error("ceil_div: division by zero");
        return tensorrt_llm::common::divUp(x, y);
    });

    m.def("pad_up", [](int64_t x, int64_t y) -> int64_t
    {
        if (y == 0) throw std::runtime_error("pad_up: division by zero");
        return tensorrt_llm::common::padUp(x, y);
    });

    m.def("div_up", [](int64_t x, int64_t y) -> int64_t
    {
        if (y == 0) throw std::runtime_error("div_up: division by zero");
        return tensorrt_llm::common::divUp(x, y);
    });

    // --- from _torch/utils.py ---
    m.def("next_positive_power_of_2", [](int64_t x) -> int64_t
    {
        return tensorrt_llm::common::nextPowerOf2(x);
    });

    m.def("last_positive_power_of_2", [](int64_t x) -> int64_t
    {
        return tensorrt_llm::common::lastPowerOf2(x);
    });

    m.def("nearest_in_buckets", [](int64_t x, std::vector<int64_t> const& buckets) -> int64_t
    {
        return tensorrt_llm::common::nearestInBuckets(x, buckets);
    });

    m.def("get_power_of_2_num_tokens_buckets", [](int64_t max_num_tokens) -> std::vector<int64_t>
    {
        return tensorrt_llm::common::getPowerOf2NumTokensBuckets(max_num_tokens);
    });

    m.def("get_last_power_of_2_num_tokens_buckets", [](int64_t max_num_tokens) -> std::vector<int64_t>
    {
        return tensorrt_llm::common::getLastPowerOf2NumTokensBuckets(max_num_tokens);
    });

    m.def("deep_gemm_gen_tuning_buckets", [](int64_t x) -> std::vector<int64_t>
    {
        return tensorrt_llm::common::deepGemmGenTuningBuckets(x);
    });

    m.def("compute_swizzled_sf_shape", [](int64_t row, int64_t col) -> std::pair<int64_t, int64_t>
    {
        return tensorrt_llm::common::computeSwizzledSfShape(row, col);
    });

    // --- from _torch/memory_buffer_utils.py ---
    // --- from _utils.py (dimension utilities) ---
    m.def("dim_to_trt_axes", [](std::vector<int64_t> const& dims) -> int64_t
    {
        return tensorrt_llm::common::dimToTrtAxes(dims);
    });

    m.def("trt_axes_to_dim", [](int64_t axes) -> std::vector<int64_t>
    {
        return tensorrt_llm::common::trtAxesToDim(axes);
    });

    m.def("dim_resolve_negative", [](std::vector<int64_t> const& dims, int64_t ndim) -> std::vector<int64_t>
    {
        return tensorrt_llm::common::dimResolveNegative(dims, ndim);
    });

    // --- from _torch/utils.py (fp8 scale shape) ---
    m.def("fp8_scale_infer_shape", [](std::vector<std::vector<int64_t>> const& input_shapes) -> int64_t
    {
        return tensorrt_llm::common::fp8ScaleInferShape(input_shapes);
    });

    m.def("get_size_in_byte", [](std::vector<int64_t> const& shape, int64_t element_size) -> int64_t
    {
        int64_t total = 1;
        for (auto s : shape) total *= s;
        return total * element_size;
    });

    // --- from _torch/utils.py (is_gated_activation) ---
    // Values match ActivationType enum in cpp/tensorrt_llm/kernels/cutlass_kernels/include/common.h:
    // Swiglu=5, Geglu=6, SwigluBias=7
    m.def("is_gated_activation", [](int64_t activation_type) -> bool
    {
        return activation_type == 5 || activation_type == 6 || activation_type == 7;
    });
}

} // namespace tensorrt_llm::nanobind::utils
