# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import faiss
import numpy as np
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.joint_multihead_attention_sum import JointMultiheadAttentionWeightedSum
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from fairseq.models.transformer import (
    TransformerConfig,
)
from fairseq.checkpoint_utils import load_model_ensemble
import time
import pickle

class TransformerDecoderSideNetLayer(nn.Module):
    """Decoder layer block for Side Network.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, memory=False
    ):
        super().__init__()
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            memory=memory,
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim
        self.num_heads = cfg.decoder.attention_heads
        scale_heads = utils.safe_getattr(cfg, "scale_heads", False)
        self.c_attn = (
            nn.Parameter(torch.ones((self.nh,)), requires_grad=True)
            if scale_heads
            else None
        )

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)

        self.normalize_before = cfg.decoder.normalize_before

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ln_1 = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False
        self.precompute_retrieval_output = None

        self.index_list = np.empty(self.num_heads,dtype=object)
        self.chunk_size = getattr(cfg, "chunk_size", 4)
        self.dimension = cfg.decoder.embed_dim
        self.head_dim_mem = int(self.dimension / self.num_heads)
        self.k = cfg.k

        self.cluster_allocation = np.empty(self.num_heads,dtype=object)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False, memory=False
    ):
        if memory:
            return JointMultiheadAttentionWeightedSum(
                embed_dim,
                cfg.decoder.attention_heads,
                dropout=cfg.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not cfg.cross_self_attention,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                bias=False,
            )
        else:
            return MultiheadAttention(
                embed_dim,
                cfg.decoder.attention_heads,
                dropout=cfg.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not cfg.cross_self_attention,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                bias=False,
            )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def clear_cluster_allocation(self, num_clusters):
        for i in range(self.num_heads):
            self.cluster_allocation[i] = np.zeros(num_clusters)

    def initalize(self, knn_config):
        centroid_list = knn_config["centroids"]
        assignments = knn_config["assignments"]
        keys_list = knn_config["keys_list"]
        values_list = knn_config["values_list"]
        num_clusters = knn_config["clusters"]
        keys_list_chunk = knn_config["keys_list_chunk"]
        values_list_chunk = knn_config["values_list_chunk"]
        self.chunk_size = knn_config["chunk_size"]

        for i in range(self.num_heads):
            self.cluster_allocation[i] = np.zeros(num_clusters)
        
        self.keys_assignment_store = np.zeros((self.num_heads, num_clusters), dtype = object)
        self.values_assignment_store = np.zeros((self.num_heads, num_clusters), dtype = object)

        self.keys_assignment_store_chunk = np.zeros((self.num_heads, num_clusters), dtype=object)
        self.values_assignment_store_chunk = np.zeros((self.num_heads, num_clusters), dtype=object)

        for i in range(self.num_heads):
            for j in range(num_clusters):
                self.keys_assignment_store[i][j] = np.empty((0, 64))
                self.values_assignment_store[i][j] = np.empty((0, 64))
                self.keys_assignment_store_chunk[i][j] = np.empty((0, self.chunk_size, 64))
                self.values_assignment_store_chunk[i][j] = np.empty((0, self.chunk_size, 64))

        # print("here1")
        for i in range(self.num_heads):
            print(i)
            for index, assignment in enumerate(assignments[i]):
                self.keys_assignment_store[i][assignment] = np.vstack((self.keys_assignment_store[i][assignment], keys_list[i][index]))
                self.values_assignment_store[i][assignment] = np.vstack((self.values_assignment_store[i][assignment], values_list[i][index]))

                self.keys_assignment_store_chunk[i][assignment] = np.vstack((self.keys_assignment_store_chunk[i][assignment], keys_list_chunk[i][index].reshape((1, self.chunk_size, 64))))
                self.values_assignment_store_chunk[i][assignment] = np.vstack((self.values_assignment_store_chunk[i][assignment], values_list_chunk[i][index].reshape((1, self.chunk_size, 64))))

        print(self.keys_assignment_store_chunk[0][0].shape)

        print('put index from cpu to gpu {}'.format(torch.cuda.current_device()))

        self.res = faiss.StandardGpuResources()
        for i in range(self.num_heads):
            gpu_index = faiss.IndexFlatIP(self.head_dim_mem)
            gpu_index = faiss.index_cpu_to_gpu(self.res, torch.cuda.current_device(), gpu_index)
            self.index_list[i] = gpu_index

        self.cluster_index = np.empty((self.num_heads, num_clusters), dtype = object)
        for i in range(self.num_heads):
            for j in range(num_clusters):
                index = faiss.IndexFlatIP(self.head_dim_mem)
                # index = faiss.IndexIVFPQ(index, self.head_dim_mem, 25, 4, 8)
                index = faiss.index_cpu_to_gpu(self.res, torch.cuda.current_device(), index)
                
                # index.train(self.keys_assignment_store[i][j].astype(np.float32))
                index.add(self.keys_assignment_store[i][j].astype(np.float32))
                self.cluster_index[i][j] = index

        
        for i, index in enumerate(centroid_list):
            self.index_list[i].add(index)

    def calc_retrieval_output(self, knn_config, queries):
        centroid_list = knn_config["centroids"]
        assignments = knn_config["assignments"]
        keys_list = knn_config["keys_list"]
        values_list = knn_config["values_list"]
        num_clusters = knn_config["clusters"]
        
        # if len(self.index_list) == 0:
        #     self.keys_assignment_store = np.zeros((self.num_heads, num_clusters), dtype = object)
        #     self.values_assignment_store = np.zeros((self.num_heads, num_clusters), dtype = object)

        #     for i in range(self.num_heads):
        #         for j in range(num_clusters):
        #             self.keys_assignment_store[i][j] = np.empty((0, 64))
        #             self.values_assignment_store[i][j] = np.empty((0, 64))

        #     # print("here1")
        #     for i in range(self.num_heads):
        #         print(i)
        #         for index, assignment in enumerate(assignments[i]):
        #             self.keys_assignment_store[i][assignment] = np.vstack((self.keys_assignment_store[i][assignment], keys_list[i][index]))
        #             self.values_assignment_store[i][assignment] = np.vstack((self.values_assignment_store[i][assignment], values_list[i][index]))
        #     print('put index from cpu to gpu {}'.format(torch.cuda.current_device()))

        #     self.res = faiss.StandardGpuResources()
        #     for i in range(self.num_heads):
        #         gpu_index = faiss.IndexFlatIP(self.head_dim_mem)
        #         gpu_index = faiss.index_cpu_to_gpu(self.res, torch.cuda.current_device(), gpu_index)
        #         self.index_list.append(gpu_index)

        #     self.cluster_index = np.empty((self.num_heads, num_clusters), dtype = object)
        #     for i in range(self.num_heads):
        #         for j in range(num_clusters):
        #             index = faiss.IndexFlatIP(self.head_dim_mem)
        #             index = faiss.index_cpu_to_gpu(self.res, torch.cuda.current_device(), index)
                    
        #             index.add(self.keys_assignment_store[i][j].astype(np.float32))
        #             self.cluster_index[i][j] = index

            
        #     for i, index in enumerate(centroid_list):
        #         self.index_list[i].add(index)

        seq_len, bsz, hid_size = queries.shape
        queries = queries.view(seq_len*bsz, self.num_heads, self.head_dim_mem).type(torch.float32)
        start = time.time()

        # for i in range(self.num_heads)
        indexs = [self.index_list[i].search(queries[:, i, :].contiguous().cpu(), 1)[1] for i in range(self.num_heads)]
        end = time.time()
        # print("search number 1:",end-start)
        
        start = time.time()
        indices = np.zeros((self.num_heads, seq_len * bsz, self.k // self.chunk_size))
        for i, index in enumerate(indexs):
            for j in range(seq_len * bsz):
                self.cluster_allocation[i][index[j]]+=1
                indices[i,j,:] = self.cluster_index[i][index[j]].search(
                                        queries[j, i, :].view(1,-1).contiguous().cpu(),
                                        self.k // self.chunk_size
                                    )[1]

        keys_tgt_index = []
        vals_tgt_index = []

        for i in range(self.num_heads):
            keys_list = []
            values_list = []
            
            for j in range(seq_len * bsz):
                indices_i_j = indices[i][j] # 16
                indices_i_j = [int(x) for x in indices_i_j]
                key_values = []
                value_values = []
                concatenated_keys = torch.tensor(self.keys_assignment_store_chunk[i][indexs[i][j]][indices_i_j].reshape(-1, self.k, self.head_dim_mem))
                concatenated_values = torch.tensor(self.values_assignment_store_chunk[i][indexs[i][j]][indices_i_j].reshape(-1, self.k, self.head_dim_mem))
                    
                keys_list.append(concatenated_keys)
                values_list.append(concatenated_values)
        #         print(1)

            keys_tgt_index.append(torch.cat(keys_list, dim=0))
            vals_tgt_index.append(torch.cat(values_list, dim=0))
        end = time.time()

        start = time.time()

        keys_tgt_index = torch.stack(keys_tgt_index).view(seq_len, bsz*self.num_heads, self.k, self.head_dim).transpose(0, 1)
        vals_tgt_index = torch.stack(vals_tgt_index).view(seq_len, bsz*self.num_heads, self.k, self.head_dim).transpose(0, 1)
        end = time.time()

        # print("final", end-start)
        return {'knn_index': indexs, 'tgt_index': {"k": keys_tgt_index, "v": vals_tgt_index}}




    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        position_encoding: Optional[torch.Tensor] = None,
        long_context_retrieval: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        external_memory = None,
        precompute_retrieval = None,
        calc_context = None,
        knn_config = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
            long_context_retrieval (Tensor. optional): input to the cross attenton `(seq_len, batch, k, embed_dim) 

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True
        
        # directly load residual connection for x_{n+1} - x_{n} + y_{n}
        ladder_residual = residual + x
        
        x = self.ln_1(x)
        
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)

        if calc_context is not None:
            if calc_context:
                self.precompute_retrieval_output = external_memory.retrieve(self.self_attn.q_proj(x))

        if external_memory is not None and calc_context is None:
            if external_memory.dstore_idx == 0 and knn_config is None:
                long_context_retrieval = None
            else:
                if precompute_retrieval is not None:
                    retrieval_output = precompute_retrieval
                else:
                    if knn_config is not None:
                        retrieval_output = self.calc_retrieval_output(knn_config, self.self_attn.q_proj(x))
                    else:    
                        retrieval_output = external_memory.retrieve(self.self_attn.q_proj(x))
                long_context_retrieval = retrieval_output['tgt_index'] #if retrieval_output else None

            attn_output, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_encoding=position_encoding,
                long_context_retrieval=long_context_retrieval,
            )
        else:
            attn_output, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_encoding=position_encoding,
            )

        mlp_output = self.fc2(self.activation_fn(self.fc1(x)))
        mlp_output = self.dropout_module(mlp_output)
        
        x = mlp_output + attn_output + ladder_residual

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        # modified by Weizhi
        try:
            return x, attn, retrieval_output['knn_index']
        except (TypeError, UnboundLocalError):
            return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

