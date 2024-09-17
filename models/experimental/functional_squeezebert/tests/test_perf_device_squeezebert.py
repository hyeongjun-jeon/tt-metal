# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test",
    [
        [8, "wormhole_b0-True-384-8-squeezebert/squeezebert-uncased"],
    ],
)
def test_perf_device_bare_metal(batch_size, test):
    subdir = "ttnn_squeezebert"
    num_iterations = 3
    margin = 0.03
    expected_perf = 431.52

    command = f"pytest models/experimental/functional_squeezebert/tests/test_ttnn_optimized_squeezebert.py::test_squeezebert_for_question_answering_dp[{test}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_squeezebert_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
