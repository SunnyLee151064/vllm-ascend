    def generate_mtp_attention_mask_for_decode(
        self,
        decode_num_computed_tokens: list[int],
        decode_num_scheduled_tokens: np.ndarray,
    ) -> list[torch.Tensor | None]:
        """
        Generate MTP attention masks for decode requests in PCP mode.

        This function handles the case where decode requests with MTP (speculative decoding)
        need attention masks computed based on the local sequence after load balancing.

        New MTP token allocation logic (using position % cp_size):
        - History tokens are already split via DualChunkSwap
        - MTP tokens are allocated based on (history_len + mtp_idx) % cp_size
        - Each rank only computes mask for tokens assigned to itself

        Example:
            - pcp=1, dcp=2 (cp_size=2)
            - history_len=5: [a,b,c,d,e] split via DualChunkSwap
              - cp0: [a,b,c] (positions 0,1,2) -> 3 tokens
              - cp1: [d,e] (positions 3,4) -> 2 tokens
            - num_scheduled_tokens=4: [f,g,h,i] (positions 5,6,7,8)
            - MTP allocation by position % cp_size:
              - f: pos 5 % 2 = 1 -> rank1
              - g: pos 6 % 2 = 0 -> rank0
              - h: pos 7 % 2 = 1 -> rank1
              - i: pos 8 % 2 = 0 -> rank0
            - Final:
              - rank0: [a,b,c,g,i] positions [0,1,2,6,8] -> mask shape 4x5
              - rank1: [d,e,f,h] positions [3,4,5,7] -> mask shape 4x4

        Args:
            decode_num_computed_tokens: List of global history lengths for decode requests
            decode_num_scheduled_tokens: Array of scheduled token counts for decode requests
        """
        cp_rank = self.pcp_world_rank * self.dcp_world_size + self.dcp_world_rank
        cp_size = self.pcp_world_size * self.dcp_world_size
        assert cp_size > 1, "cp_size must be greater than 1"

        interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size

        q_lens = torch.tensor(decode_num_scheduled_tokens[: self.num_decode_reqs], dtype=torch.int32)
        global_histories = torch.tensor(decode_num_computed_tokens, dtype=torch.int32)
        total_lens = global_histories + q_lens
        context_lens = total_lens - q_lens

        # Interleave-aware per-rank KV length:
        # base = L // I // W * I, remainder = L - base * W,
        # local = base + clip(remainder - rank * I, 0, I)
        base_k = total_lens // interleave_size // cp_size * interleave_size
        remainder_k = total_lens - base_k * cp_size
        k_lens = base_k + torch.clamp(remainder_k - cp_rank * interleave_size, 0, interleave_size)
        valid = k_lens > 0

        if not valid.any():
            return self.dcp_mtp_attn_mask.cpu[: self.num_decode_reqs]

        k_lens = torch.where(valid, k_lens, torch.zeros_like(k_lens))

        mtp_attn_mask = self.dcp_mtp_attn_mask.cpu[: self.num_decode_reqs]
        mtp_attn_mask.zero_()

        num_valid = valid.sum().item()
        if num_valid == 0:
            return mtp_attn_mask

        max_q = int(q_lens[valid].max().item())
        max_k = int(k_lens[valid].max().item())

        # Generate indices up to max dimensions
        q_indices = torch.arange(max_q, dtype=torch.int32)
        k_indices = torch.arange(max_k, dtype=torch.int32)

        valid_q = valid[:, None] & (q_indices[None, :] < q_lens[:, None])
        valid_k = valid[:, None] & (k_indices[None, :] < k_lens[:, None])

        # Interleave-aware k_upper: for query token at global position P,
        # the number of rank-cp_rank KV tokens before P (exclusive upper bound).
        positions = context_lens[:, None] + q_indices[None, :]  # [num_decode_reqs, max_q]
        base_q = positions // interleave_size // cp_size * interleave_size
        remainder_q = positions - base_q * cp_size
        local_q = base_q + torch.clamp(remainder_q - cp_rank * interleave_size, 0, interleave_size)
        k_upper = local_q - 1  # inclusive upper KV index

        k_upper_expanded = k_upper[:, :, None]  # [num_decode_reqs, max_q, 1]
        k_idx_expanded = k_indices[None, None, :]  # [1, 1, max_k]
        full_mask = (k_idx_expanded > k_upper_expanded) & (k_upper_expanded >= 0)

        valid_mask_3d = valid_q[:, :, None] & valid_k[:, None, :]
        full_mask = full_mask & valid_mask_3d

        mtp_attn_mask[: self.num_decode_reqs, :max_q, :max_k] = full_mask

        return mtp_attn_mask
