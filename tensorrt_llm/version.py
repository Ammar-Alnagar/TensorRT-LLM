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
Version string for Trinity LLM.

Defined in C++ when native bindings are available, with Python fallback.
"""
import functools


@functools.lru_cache(None)
def _get_cpp_version():
    try:
        from tensorrt_llm.bindings.internal import utils as _cpp_utils
        return str(_cpp_utils.__version__)
    except (ImportError, AttributeError):
        return None


_cpp_ver = _get_cpp_version()
if _cpp_ver is not None:
    __version__ = _cpp_ver
else:
    __version__ = "1.3.0rc20"
