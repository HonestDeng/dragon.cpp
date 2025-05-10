#pragma once

#include "operators.h"
#include "utils.h"
#include <vector>

// 注意力层的前向传播
struct dragon_tensor* attention_forward(
    struct dragon_context* ctx0,
    struct dragon_tensor* cur,
    const llama_layer& layer,
    struct dragon_tensor* memory_k,
    struct dragon_tensor* memory_v,
    int n_past,
    int n_embd,
    int n_head,
    int n_ctx,
    int N,
    int il);

// 前馈网络层的前向传播
struct dragon_tensor* ffn_forward(
    struct dragon_context* ctx0,
    struct dragon_tensor* inpFF,
    const llama_layer& layer);

// 层归一化
struct dragon_tensor* layer_norm(
    struct dragon_context* ctx0,
    struct dragon_tensor* inp,
    struct dragon_tensor* norm);

// 主评估函数
bool llama_eval(
    const llama_model& model,
    const int n_threads,
    const int n_past,
    const std::vector<gpt_vocab::id>& embd_inp,
    std::vector<float>& embd_w,
    size_t& mem_per_token); 