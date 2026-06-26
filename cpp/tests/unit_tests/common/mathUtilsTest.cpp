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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/mathUtils.h"

using namespace tensorrt_llm::common;

namespace
{

TEST(MathUtilsTest, DivUp)
{
    EXPECT_EQ(divUp(10, 3), 4);
    EXPECT_EQ(divUp(9, 3), 3);
    EXPECT_EQ(divUp(0, 5), 0);
    EXPECT_EQ(divUp(7, 1), 7);
    EXPECT_EQ(divUp(100, 100), 1);
    EXPECT_EQ(divUp(1, 100), 1);
    EXPECT_EQ(divUp(1000, 333), 4);
}

TEST(MathUtilsTest, PadUp)
{
    EXPECT_EQ(padUp(10, 3), 12);
    EXPECT_EQ(padUp(9, 3), 9);
    EXPECT_EQ(padUp(7, 5), 10);
    EXPECT_EQ(padUp(0, 5), 0);
    EXPECT_EQ(padUp(1, 1), 1);
    EXPECT_EQ(padUp(5, 8), 8);
    EXPECT_EQ(padUp(100, 32), 128);
}

TEST(MathUtilsTest, NextPowerOf2)
{
    EXPECT_EQ(nextPowerOf2(0), 1);
    EXPECT_EQ(nextPowerOf2(1), 1);
    EXPECT_EQ(nextPowerOf2(2), 2);
    EXPECT_EQ(nextPowerOf2(3), 4);
    EXPECT_EQ(nextPowerOf2(4), 4);
    EXPECT_EQ(nextPowerOf2(5), 8);
    EXPECT_EQ(nextPowerOf2(7), 8);
    EXPECT_EQ(nextPowerOf2(8), 8);
    EXPECT_EQ(nextPowerOf2(9), 16);
    EXPECT_EQ(nextPowerOf2(63), 64);
    EXPECT_EQ(nextPowerOf2(64), 64);
    EXPECT_EQ(nextPowerOf2(65), 128);
    EXPECT_EQ(nextPowerOf2(1023), 1024);
    EXPECT_EQ(nextPowerOf2(1024), 1024);
    EXPECT_EQ(nextPowerOf2(1025), 2048);
    EXPECT_EQ(nextPowerOf2(1LL << 40), 1LL << 40);
    EXPECT_EQ(nextPowerOf2((1LL << 40) + 1), 1LL << 41);
}

TEST(MathUtilsTest, LastPowerOf2)
{
    EXPECT_EQ(lastPowerOf2(0), 1LL << 63);
    EXPECT_EQ(lastPowerOf2(1), 1);
    EXPECT_EQ(lastPowerOf2(2), 2);
    EXPECT_EQ(lastPowerOf2(3), 2);
    EXPECT_EQ(lastPowerOf2(4), 4);
    EXPECT_EQ(lastPowerOf2(5), 4);
    EXPECT_EQ(lastPowerOf2(7), 4);
    EXPECT_EQ(lastPowerOf2(8), 8);
    EXPECT_EQ(lastPowerOf2(9), 8);
    EXPECT_EQ(lastPowerOf2(63), 32);
    EXPECT_EQ(lastPowerOf2(64), 64);
    EXPECT_EQ(lastPowerOf2(65), 64);
    EXPECT_EQ(lastPowerOf2(100), 64);
}

TEST(MathUtilsTest, NearestInBuckets)
{
    std::vector<int64_t> buckets = {1, 2, 4, 8, 16, 32, 64};
    EXPECT_EQ(nearestInBuckets(0, buckets), 1);
    EXPECT_EQ(nearestInBuckets(1, buckets), 1);
    EXPECT_EQ(nearestInBuckets(3, buckets), 4);
    EXPECT_EQ(nearestInBuckets(5, buckets), 8);
    EXPECT_EQ(nearestInBuckets(17, buckets), 32);
    EXPECT_EQ(nearestInBuckets(100, buckets), 64);
}

TEST(MathUtilsTest, GetPowerOf2NumTokensBuckets)
{
    auto result = getPowerOf2NumTokensBuckets(10);
    std::vector<int64_t> expected = {1, 2, 4, 8, 16};
    EXPECT_EQ(result, expected);
}

TEST(MathUtilsTest, GetLastPowerOf2NumTokensBuckets)
{
    auto result = getLastPowerOf2NumTokensBuckets(10);
    std::vector<int64_t> expected = {1, 2, 4, 8};
    EXPECT_EQ(result, expected);
}

TEST(MathUtilsTest, DeepGemmGenTuningBucketsSmall)
{
    auto result = deepGemmGenTuningBuckets(64);
    // x < 128: just the initial 8-step buckets
    std::vector<int64_t> expected;
    for (int64_t i = 8; i < 128; i += 8)
    {
        expected.push_back(i);
    }
    EXPECT_EQ(result, expected);
}

TEST(MathUtilsTest, DeepGemmGenTuningBucketsLarge)
{
    auto result = deepGemmGenTuningBuckets(4096);
    EXPECT_GE(result.size(), 8);
    EXPECT_EQ(result[0], 8);
    EXPECT_EQ(result[14], 120);
    EXPECT_EQ(result[15], 128);
    for (size_t i = 15; i < result.size(); ++i)
    {
        EXPECT_EQ(result[i] % 128, 0);
    }
}

TEST(MathUtilsTest, ComputeSwizzledSfShape)
{
    auto [padded_row, padded_col] = computeSwizzledSfShape(64, 4);
    EXPECT_EQ(padded_row, 128);
    EXPECT_EQ(padded_col, 4);

    auto [pr2, pc2] = computeSwizzledSfShape(128, 4);
    EXPECT_EQ(pr2, 128);
    EXPECT_EQ(pc2, 4);

    auto [pr3, pc3] = computeSwizzledSfShape(130, 6);
    EXPECT_EQ(pr3, 256);
    EXPECT_EQ(pc3, 8);
}

} // namespace
