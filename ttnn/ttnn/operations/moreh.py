# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

adam = ttnn._ttnn.operations.moreh.moreh_adam
arange = ttnn._ttnn.operations.moreh.moreh_arange
getitem = ttnn._ttnn.operations.moreh.moreh_getitem
logsoftmax = ttnn._ttnn.operations.moreh.moreh_logsoftmax
logsoftmax_backward = ttnn._ttnn.operations.moreh.moreh_logsoftmax_backward
matmul = ttnn._ttnn.operations.moreh.moreh_matmul
mean = ttnn._ttnn.operations.moreh.moreh_mean
mean_backward = ttnn._ttnn.operations.moreh.moreh_mean_backward
softmax = ttnn._ttnn.operations.moreh.moreh_softmax
softmax_backward = ttnn._ttnn.operations.moreh.moreh_softmax_backward
softmin = ttnn._ttnn.operations.moreh.moreh_softmin
softmin_backward = ttnn._ttnn.operations.moreh.moreh_softmin_backward
sum = ttnn._ttnn.operations.moreh.moreh_sum

SoftmaxBackwardOp = ttnn._ttnn.operations.moreh.MorehSoftmaxBackwardOp
SoftmaxBackwardOpParallelizationStrategy = ttnn._ttnn.operations.moreh.MorehSoftmaxBackwardOpParallelizationStrategy
SoftmaxOp = ttnn._ttnn.operations.moreh.MorehSoftmaxOpParallelizationStrategy
SoftmaxOpParallelizationStrategy = ttnn._ttnn.operations.moreh.MorehSoftmaxOpParallelizationStrategy
