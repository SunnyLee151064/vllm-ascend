    def generate_pcp_metadata(
        self,
        total_num_scheduled_tokens: int,
        query_lens: torch.Tensor,
        input_batch: "NPUInputBatch",
        num_scheduled_tokens: np.ndarray | None,
        block_table_tensor: torch.Tensor,
        num_reqs_padded: int,
        num_reqs: int,
        fixed_decode_seq_lens_cpu: np.ndarray | None = None,
    ):
        from vllm_ascend.attention.utils import AscendPrefillContextParallelMetadata

        if self.pcp_world_size > 1 and self.pcp_use_hybrid_attn:
            assert self.num_scheduled_tokens_padded is not None
            total_num_scheduled_tokens = self.num_scheduled_tokens_padded.sum()
        num_actual_tokens_pcp_padded = total_num_scheduled_tokens * self.pcp_world_size
        self.num_actual_tokens_pcp_padded = num_actual_tokens_pcp_padded
        long_seq_metadata = None
        ori_query_lens_cpu = self.query_lens_pcp_full.cpu[:num_reqs_padded]
        if self.pcp_world_size * self.dcp_world_size > 1:
            assert num_scheduled_tokens is not None
            if fixed_decode_seq_lens_cpu is not None:
                decode_context_lens = fixed_decode_seq_lens_cpu[: self.num_decode_reqs]
            else:
                decode_context_lens = (
                    input_batch.num_computed_tokens_cpu[: self.num_decode_reqs]
                    + num_scheduled_tokens[: self.num_decode_reqs]
                )
            prefill_context_lens = input_batch.num_computed_tokens_cpu[self.num_decode_reqs : self.num_reqs]
            context_lens = np.concatenate([decode_context_lens, prefill_context_lens])

            num_computed_tokens_of_pcp_dcp = self._get_cp_local_seq_lens(
                torch.tensor(context_lens),
                self.pcp_world_size,
                self.dcp_world_size,
                self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
            )

            pcp_unpad_mask = self.pcp_unpad_mask_cpu[: self.pcp_padded_tokens_length]
            long_seq_metadata = AscendPrefillContextParallelMetadata(
                pcp_use_hybrid_attn=self.pcp_use_hybrid_attn,
                num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp.numpy(),
                pcp_unpad_mask=torch.from_numpy(pcp_unpad_mask),
                pcp_padded_tokens_fla=self.pcp_padded_tokens_fla,
                query_lens_pcp_full_cpu=ori_query_lens_cpu,
                max_query_len_pcp_full=ori_query_lens_cpu.max().item(),
            )
            if self.pcp_world_size > 1:
                q_head_idx, q_tail_idx = [], []
                kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx = [], []
                kv_with_q_tail_nomask_idx, kv_with_q_tail_mask_idx = [], []
                kv_tail_proj_idx: list[int] = []
                kv_with_q_head_attn_idx_in_tail, kv_with_q_tail_attn_idx_in_tail = [], []
                split_with_q_head_nomask_idx_reqs = []
                split_kv_with_q_tail_nomask_idx_reqs = []
                chunk_seqlens = []
                kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = [], []
                head_actual_seq_lengths_kv, tail_actual_seq_lengths_kv = [], []
                q_req_offset = 0
                kv_req_offset = 0
                q_head_chunk_id = self.pcp_world_rank
                q_tail_chunk_id = self.pcp_world_size * 2 - 1 - self.pcp_world_rank
                for i, seq_len in enumerate(query_lens):
                    if i < self.num_decode_reqs:
                        continue
                    chunk_len = seq_len // 2
                    chunk_seqlens.append(chunk_len)
                    q_head_idx.extend(list(range(q_req_offset, q_req_offset + chunk_len)))
                    kv_with_q_head_nomask_idx.extend(
                        list(range(kv_req_offset, kv_req_offset + chunk_len * q_head_chunk_id))
                    )
                    kv_with_q_head_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_head_chunk_id,
                                kv_req_offset + chunk_len * (q_head_chunk_id + 1),
                            )
                        )
                    )
                    kv_with_q_head_nomask_seqlens.append(chunk_len * q_head_chunk_id)
                    split_with_q_head_nomask_idx_reqs.append(
                        list(range(kv_req_offset, kv_req_offset + chunk_len * q_head_chunk_id))
                    )
                    q_tail_idx.extend(list(range(q_req_offset + chunk_len, q_req_offset + chunk_len * 2)))
                    kv_with_q_tail_nomask_idx.extend(
                        list(range(kv_req_offset, kv_req_offset + chunk_len * q_tail_chunk_id))
                    )
                    kv_with_q_tail_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_tail_chunk_id,
                                kv_req_offset + chunk_len * (q_tail_chunk_id + 1),
                            )
                        )
                    )
                    kv_with_q_tail_nomask_seqlens.append(chunk_len * q_tail_chunk_id)
                    split_kv_with_q_tail_nomask_idx_reqs.append(
                        list(range(kv_req_offset, kv_req_offset + chunk_len * q_tail_chunk_id))
                    )
                    tail_proj_offset = len(kv_tail_proj_idx)
                    tail_proj_len = chunk_len * (q_tail_chunk_id + 1)
                    kv_tail_proj_idx.extend(list(range(kv_req_offset, kv_req_offset + tail_proj_len)))
                    kv_with_q_head_attn_idx_in_tail.extend(
                        list(range(tail_proj_offset, tail_proj_offset + chunk_len * (q_head_chunk_id + 1)))
                    )
                    kv_with_q_tail_attn_idx_in_tail.extend(
                        list(range(tail_proj_offset, tail_proj_offset + tail_proj_len))
                    )
                    head_actual_seq_lengths_kv.append(len(kv_with_q_head_attn_idx_in_tail))
                    tail_actual_seq_lengths_kv.append(len(kv_with_q_tail_attn_idx_in_tail))
                    q_req_offset += seq_len
                    kv_req_offset += seq_len * self.pcp_world_size

                q_head_idx_tensor = self._list_to_tensor(q_head_idx, self.device)
                q_tail_idx_tensor = self._list_to_tensor(q_tail_idx, self.device)
                self.q_head_idx_tensor = q_head_idx_tensor
                self.q_tail_idx_tensor = q_tail_idx_tensor

                q_full_idx = torch.cat([q_head_idx_tensor, q_tail_idx_tensor])
                q_full_idx = q_full_idx.to(torch.float32).argsort().to(torch.int32)
                self.q_full_idx = q_full_idx

                self.kv_idx_names = {
                    "kv_with_q_head_nomask_idx_tensor": kv_with_q_head_nomask_idx,
                    "kv_with_q_head_mask_idx_tensor": kv_with_q_head_mask_idx,
                    "kv_with_q_tail_nomask_idx_tensor": kv_with_q_tail_nomask_idx,
                    "kv_with_q_tail_mask_idx_tensor": kv_with_q_tail_mask_idx,
                    "kv_tail_proj_idx_tensor": kv_tail_proj_idx,
                    "kv_with_q_head_attn_idx_in_tail_tensor": kv_with_q_head_attn_idx_in_tail,
                    "kv_with_q_tail_attn_idx_in_tail_tensor": kv_with_q_tail_attn_idx_in_tail,
                }
                for key, value in self.kv_idx_names.items():
                    tensor_npu = self._list_to_tensor(value, self.device)
                    self.kv_idx_names[key] = tensor_npu

                attn_chunk_seqlens = torch.tensor(chunk_seqlens, dtype=torch.int32)
                attn_mask_seqlens = torch.cumsum(torch.tensor(chunk_seqlens, dtype=torch.int32), dim=0).tolist()
                head_attn_nomask_seqlens = torch.cumsum(
                    torch.tensor(kv_with_q_head_nomask_seqlens, dtype=torch.int32), dim=0
                ).tolist()
                tail_attn_nomask_seqlens = torch.cumsum(
                    torch.tensor(kv_with_q_tail_nomask_seqlens, dtype=torch.int32), dim=0
                ).tolist()

                self.extra_long_seq_kwargs = {
                    "attn_mask_seqlens": attn_mask_seqlens,
                    "head_attn_nomask_seqlens": head_attn_nomask_seqlens,
                    "tail_attn_nomask_seqlens": tail_attn_nomask_seqlens,
                    "head_actual_seq_lengths_kv": head_actual_seq_lengths_kv,
                    "tail_actual_seq_lengths_kv": tail_actual_seq_lengths_kv,
                }
                long_seq_metadata.pcp_allgather_restore_idx = self.pcp_allgather_restore_idx.gpu[
                    :num_actual_tokens_pcp_padded
                ]
                if self.pcp_use_hybrid_attn:
                    long_seq_metadata.pcp_exit_fa_scatter_idx = self.pcp_exit_fa_scatter_idx.gpu[
                        : num_scheduled_tokens.sum() - self.num_decode_tokens
                    ]
                    long_seq_metadata.pcp_fa_query_idx = self.pcp_fa_query_idx[
                        : num_actual_tokens_pcp_padded // self.pcp_world_size - self.num_decode_tokens
                    ]
                    long_seq_metadata.pcp_enter_fa_restore_idx = self.pcp_enter_fa_restore_idx[
                        : pcp_unpad_mask.sum() + self.num_decode_tokens * (self.pcp_world_size - 1)
                    ]
                    long_seq_metadata.max_num_tokens_across_pcp = self.max_num_tokens_across_pcp
                    long_seq_metadata.total_num_scheduled_tokens = self.total_num_scheduled_tokens
                long_seq_metadata.q_head_idx_tensor = self.q_head_idx_tensor
                long_seq_metadata.q_tail_idx_tensor = self.q_tail_idx_tensor
                long_seq_metadata.q_full_idx = self.q_full_idx
                long_seq_metadata.kv_with_q_head_nomask_idx_tensor = self.kv_idx_names[
                    "kv_with_q_head_nomask_idx_tensor"
                ]
                long_seq_metadata.kv_with_q_head_mask_idx_tensor = self.kv_idx_names["kv_with_q_head_mask_idx_tensor"]
                long_seq_metadata.kv_with_q_tail_nomask_idx_tensor = self.kv_idx_names[
                    "kv_with_q_tail_nomask_idx_tensor"
                ]
                long_seq_metadata.kv_with_q_tail_mask_idx_tensor = self.kv_idx_names["kv_with_q_tail_mask_idx_tensor"]
                long_seq_metadata.kv_tail_proj_idx_tensor = self.kv_idx_names["kv_tail_proj_idx_tensor"]
                long_seq_metadata.kv_with_q_head_attn_idx_in_tail_tensor = self.kv_idx_names[
                    "kv_with_q_head_attn_idx_in_tail_tensor"
                ]
                long_seq_metadata.kv_with_q_tail_attn_idx_in_tail_tensor = self.kv_idx_names[
                    "kv_with_q_tail_attn_idx_in_tail_tensor"
                ]
                long_seq_metadata.attn_mask_seqlens = self.extra_long_seq_kwargs["attn_mask_seqlens"]
                long_seq_metadata.head_attn_nomask_seqlens = self.extra_long_seq_kwargs["head_attn_nomask_seqlens"]
                long_seq_metadata.tail_attn_nomask_seqlens = self.extra_long_seq_kwargs["tail_attn_nomask_seqlens"]
                long_seq_metadata.head_actual_seq_lengths_kv = self.extra_long_seq_kwargs["head_actual_seq_lengths_kv"]
                long_seq_metadata.tail_actual_seq_lengths_kv = self.extra_long_seq_kwargs["tail_actual_seq_lengths_kv"]
                long_seq_metadata.attn_chunk_seqlens = attn_chunk_seqlens

            # Generate MTP attention masks for decode requests when cp_size > 1
            # with speculative decoding.
            if (
                self.dcp_world_size * self.pcp_world_size > 1
                and self.speculative_config
                and num_scheduled_tokens is not None
            ):
                # Generate the mask contents for the real decode requests.
                if self.num_decode_reqs > 0:
                    decode_num_scheduled_tokens = num_scheduled_tokens[: self.num_decode_reqs]
                    if fixed_decode_seq_lens_cpu is not None:
                        decode_num_computed_tokens = (
                            fixed_decode_seq_lens_cpu[: self.num_decode_reqs] - decode_num_scheduled_tokens
                        ).tolist()
                    else:
                        decode_num_computed_tokens = input_batch.num_computed_tokens_cpu[: self.num_decode_reqs].tolist()

                    dcp_mtp_attn_mask = self.generate_mtp_attention_mask_for_decode(
                        decode_num_computed_tokens, decode_num_scheduled_tokens
                    )
                    if dcp_mtp_attn_mask is not None:
                        self.dcp_mtp_attn_mask.np[: self.num_decode_reqs] = dcp_mtp_attn_mask
                        self.dcp_mtp_attn_mask.copy_to_gpu(self.num_decode_reqs)
                # NOTE: Always expose the (stable, pre-allocated) MTP mask buffer
                # for cp>1 + speculative decode, even when num_decode_reqs == 0.
                # FULL_DECODE_ONLY graph capture runs against a warmup batch whose
                # requests carry no context (num_decode_reqs == 0), so the content
                # generation above is skipped. Previously this left
                # dcp_mtp_attn_mask = None, which got baked into the captured
                # decode graph; on every replay the MTP-verify causal mask was
                # then dropped, corrupting attention over the spec query positions
                # and causing repetition-loop degeneration under pcp>1 + mtp +
                # graph. Exposing the buffer here makes the captured graph hold a
                # refreshable mask tensor whose contents are rewritten by
                # copy_to_gpu on every real decode step. At capture time
                # (num_decode_reqs == 0) we size it by num_reqs, which is fixed
                # per graph bucket (num_tokens == num_reqs * decode_threshold for
                # a uniform MTP-verify batch), so the slice is stable across
                # capture and replay.
                mask_n = self.num_decode_reqs if self.num_decode_reqs > 0 else num_reqs
                long_seq_metadata.dcp_mtp_attn_mask = self.dcp_mtp_attn_mask.gpu[:mask_n]
            else:
                long_seq_metadata.dcp_mtp_attn_mask = None

        self.long_seq_metadata = long_seq_metadata
        return long_seq_metadata, block_table_tensor
