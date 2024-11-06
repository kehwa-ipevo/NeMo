# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _rnnt_logprobs_fwd_kernel(
    logits_ptr,
    targets_ptr,
    source_lengths_ptr,
    target_lengths_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,
    blank_id: int,
    target_scores_ptr,
    blank_scores_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    source_len = tl.load(source_lengths_ptr + batch_i)
    target_len = tl.load(target_lengths_ptr + batch_i)
    if source_i >= source_len or target_i > target_len:
        return

    flat_index = ((batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i) * num_labels
    logits_ptr += flat_index
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    logits_max = tl.max(logits, axis=0)
    logits_minus_max = logits - logits_max
    denominator = tl.log(tl.sum(tl.exp(logits_minus_max), axis=0))
    blank_logit = tl.load(logits_ptr + blank_id).to(tl.float32)
    flat_index_output = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    tl.store(blank_scores_ptr + flat_index_output, blank_logit - logits_max - denominator)
    if target_i < target_len:
        target_id = tl.load(targets_ptr + batch_i * (max_target_len_plus_1 - 1) + target_i)
        target_logit = tl.load(logits_ptr + target_id).to(tl.float32)
        tl.store(target_scores_ptr + flat_index_output, target_logit - logits_max - denominator)


@triton.jit
def _rnnt_logprobs_bwd_kernel(
    logits_ptr,
    grad_logits_ptr,
    targets_ptr,
    source_lengths_ptr,
    target_lengths_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,
    blank_id: int,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    source_len = tl.load(source_lengths_ptr + batch_i)
    target_len = tl.load(target_lengths_ptr + batch_i)
    if source_i >= source_len or target_i > target_len:
        return

    flat_index = ((batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i) * num_labels
    logits_ptr += flat_index
    grad_logits_ptr += flat_index

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    logits_max = tl.max(logits, axis=0)
    logits_minus_max = logits - logits_max
    denominator = tl.log(tl.sum(tl.exp(logits_minus_max), axis=0))
    log_softmax = logits_minus_max - denominator
    softmax = tl.exp(log_softmax)

    flat_index_grad = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    blank_grad = tl.load(grad_blank_scores_ptr + flat_index_grad).to(tl.float32)
    target_i_valid = target_i < target_len
    target_grad = tl.load(grad_target_scores_ptr + flat_index_grad, mask=target_i_valid, other=0.0).to(tl.float32)
    target_id = tl.load(targets_ptr + batch_i * (max_target_len_plus_1 - 1) + target_i, mask=target_i_valid, other=-1)

    grad_not_in_targets = (-softmax) * (blank_grad + target_grad)
    grad = tl.where(col_offsets == blank_id, blank_grad + grad_not_in_targets, grad_not_in_targets)
    grad = tl.where(col_offsets == target_id, target_grad + grad_not_in_targets, grad)
    tl.store(grad_logits_ptr + col_offsets, grad, mask=mask)


class RnntLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        blank_id: int,
        source_lengths: torch.Tensor | None,
        target_lengths: torch.Tensor | None,
    ):
        assert logits.is_contiguous()
        targets = targets.contiguous()
        device = logits.device
        float_dtype = torch.float32

        target_scores = torch.zeros(logits.shape[:-1], dtype=float_dtype, device=device)
        blank_scores = torch.zeros_like(target_scores)
        if source_lengths is None:
            source_lengths = torch.full([logits.shape[0]], fill_value=logits.shape[1], dtype=torch.int, device=device)
        else:
            source_lengths = source_lengths.contiguous()
        if target_lengths is None:
            target_lengths = torch.full(
                [logits.shape[0]], fill_value=logits.shape[2] - 1, dtype=torch.int, device=device
            )
        else:
            target_lengths = target_lengths.contiguous()
        _rnnt_logprobs_fwd_kernel[(logits.shape[0], logits.shape[1], logits.shape[2])](
            logits_ptr=logits,
            targets_ptr=targets,
            source_lengths_ptr=source_lengths,
            target_lengths_ptr=target_lengths,
            max_source_len=logits.shape[1],
            max_target_len_plus_1=logits.shape[2],
            num_labels=logits.shape[3],
            blank_id=blank_id,
            target_scores_ptr=target_scores,
            blank_scores_ptr=blank_scores,
            BLOCK_SIZE=triton.next_power_of_2(logits.shape[-1]),
        )

        # saving for backward
        ctx.save_for_backward(logits, targets, source_lengths, target_lengths)
        ctx.blank_id = blank_id
        return target_scores, blank_scores

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        (logits, targets, source_lengths, target_lengths) = ctx.saved_tensors
        blank_id = ctx.blank_id
        grad_logits = torch.zeros_like(logits)
        _rnnt_logprobs_bwd_kernel[(logits.shape[0], logits.shape[1], logits.shape[2])](
            logits_ptr=logits,
            grad_logits_ptr=grad_logits,
            source_lengths_ptr=source_lengths,
            target_lengths_ptr=target_lengths,
            targets_ptr=targets,
            max_source_len=logits.shape[1],
            max_target_len_plus_1=logits.shape[2],
            num_labels=logits.shape[3],
            blank_id=blank_id,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            BLOCK_SIZE=triton.next_power_of_2(logits.shape[-1]),
        )
        return grad_logits, None, None, None, None


def rnnt_logprobs_triton(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    source_lengths: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
):
    return RnntLogProbs.apply(logits, targets, blank_id, source_lengths, target_lengths)