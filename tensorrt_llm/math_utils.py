# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Math utility functions — implemented in C++ (tensorrt_llm.nanobind.utils).
"""
from typing import List, Tuple

from tensorrt_llm.bindings.internal import utils as _cpp_utils


def ceil_div(x: int, y: int) -> int:
    return _cpp_utils.ceil_div(x, y)


def pad_up(x: int, y: int) -> int:
    return _cpp_utils.pad_up(x, y)


def div_up(x: int, y: int) -> int:
    return _cpp_utils.div_up(x, y)


def next_positive_power_of_2(x: int) -> int:
    return _cpp_utils.next_positive_power_of_2(x)


def last_positive_power_of_2(x: int) -> int:
    return _cpp_utils.last_positive_power_of_2(x)


def nearest_in_buckets(x: int, buckets: List[int]) -> int:
    return _cpp_utils.nearest_in_buckets(x, buckets)


def get_power_of_2_num_tokens_buckets(max_num_tokens: int) -> Tuple[int, ...]:
    return tuple(_cpp_utils.get_power_of_2_num_tokens_buckets(max_num_tokens))


def get_last_power_of_2_num_tokens_buckets(max_num_tokens: int) -> Tuple[int, ...]:
    return tuple(_cpp_utils.get_last_power_of_2_num_tokens_buckets(max_num_tokens))


def deep_gemm_gen_tuning_buckets(x: int) -> Tuple[int, ...]:
    return tuple(_cpp_utils.deep_gemm_gen_tuning_buckets(x))


def compute_swizzled_sf_shape(row: int, col: int) -> Tuple[int, int]:
    return tuple(_cpp_utils.compute_swizzled_sf_shape(row, col))
