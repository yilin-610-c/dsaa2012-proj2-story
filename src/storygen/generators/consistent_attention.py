from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class ConsistentAttentionState:
    """Run-scoped state for cross-panel consistent attention."""

    id_length: int = 4
    write_mode: bool = False
    cur_step: int = 0
    attn_count: int = 0
    total_count: int = 0
    height: int = 768
    width: int = 768
    sa32: bool = True
    sa64: bool = True
    character_order: list[str] = field(default_factory=list)
    id_bank: dict[str, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    indices1024: Any = None
    indices4096: Any = None


class ConsistentSelfAttentionProcessor(torch.nn.Module):
    """A lightweight StoryDiffusion-style consistent attention processor.

    This is a faithful refactor of the gradio prototype idea:
    - step 0 writes identity tokens into an id bank
    - later steps read tokens from other panels/characters and concatenate them
    - attention is computed with shared encoder states to encourage identity reuse
    """

    def __init__(self, *, state: ConsistentAttentionState, hidden_size=None, cross_attention_dim=None, device="cuda", dtype=torch.float16) -> None:
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("ConsistentSelfAttentionProcessor requires PyTorch 2.0+")
        self.state = state
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        state = self.state
        if state.total_count <= 0:
            return self._call2(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        write = state.write_mode
        cur_step = state.cur_step

        if cur_step < 1:
            hidden_states = self._call2(attn, hidden_states, None, attention_mask, temb)
        else:
            random_number = random.random()
            rand_num = 0.3 if cur_step < 20 else 0.1
            if random_number > rand_num:
                hidden_states = self._apply_consistency(attn, hidden_states, encoder_hidden_states, attention_mask, temb, write=write)
            else:
                hidden_states = self._call2(attn, hidden_states, None, attention_mask, temb)

        state.attn_count += 1
        if state.attn_count == state.total_count:
            state.attn_count = 0
            state.cur_step += 1
        return hidden_states

    def _apply_consistency(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *, write: bool):
        state = self.state
        if write:
            total_batch_size, nums_token, channel = hidden_states.shape
            img_nums = total_batch_size // 2
            hidden_states = hidden_states.reshape(-1, img_nums, nums_token, channel)
            for img_ind in range(img_nums):
                character = state.character_order[img_ind] if img_ind < len(state.character_order) else f"char_{img_ind}"
                state.id_bank.setdefault(character, {})[state.cur_step] = [
                    hidden_states[:, img_ind, :, :].clone(),
                ]
            hidden_states = hidden_states.reshape(-1, nums_token, channel)
            return self._call2(attn, hidden_states, None, attention_mask, temb)

        encoder_arr = []
        for character in state.character_order:
            if character in state.id_bank and state.cur_step in state.id_bank[character]:
                encoder_arr.extend(t.to(self.device) for t in state.id_bank[character][state.cur_step])
        if not encoder_arr:
            return self._call2(attn, hidden_states, None, attention_mask, temb)
        _, nums_token, channel = hidden_states.shape
        hidden_states = hidden_states.reshape(2, -1, nums_token, channel)
        encoder_hidden_states_tmp = torch.cat(encoder_arr + [hidden_states[:, 0, :, :]], dim=1)
        hidden_states[:, 0, :, :] = self._call2(attn, hidden_states[:, 0, :, :], encoder_hidden_states_tmp, None, temb)
        return hidden_states.reshape(-1, nums_token, channel)

    def _call2(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, channel = hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
