#include "operators.h"
#include "attention.h"
#include <cmath>
#include <vector>
#include <stdexcept>

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
    int il,
    dragon_cgraph* gf) {
    
    // 计算Q、K、V
    struct dragon_tensor* Qcur = dragon_mul_mat(ctx0, layer.wq, cur);
    struct dragon_tensor* Kcur = dragon_mul_mat(ctx0, layer.wk, cur);
    struct dragon_tensor* Vcur = dragon_mul_mat(ctx0, layer.wv, cur);

    // 存储key和value到memory
    if (N >= 1) {
        struct dragon_tensor* k = dragon_view_1d(ctx0, memory_k, N * n_embd,
            (dragon_element_size(memory_k) * n_embd) * (il * n_ctx + n_past));
        struct dragon_tensor* v = dragon_view_1d(ctx0, memory_v, N * n_embd,
            (dragon_element_size(memory_v) * n_embd) * (il * n_ctx + n_past));

        dragon_build_forward_expand(gf, dragon_cpy(ctx0, Kcur, k));
        dragon_build_forward_expand(gf, dragon_cpy(ctx0, Vcur, v));
    }

    // 计算注意力
    struct dragon_tensor* Q = dragon_permute(ctx0,
        dragon_rope(ctx0,
            dragon_cpy(ctx0, Qcur,
                dragon_new_tensor_3d(ctx0, DATA_TYPE_F32, n_embd / n_head, n_head, N)),
            n_past, n_embd / n_head, 0),
        0, 2, 1, 3);

    struct dragon_tensor* K = dragon_permute(ctx0,
        dragon_rope(ctx0,
            dragon_reshape_3d(ctx0,
                dragon_view_1d(ctx0, memory_k, (n_past + N) * n_embd,
                    il * n_ctx * dragon_element_size(memory_k) * n_embd),
                n_embd / n_head, n_head, n_past + N),
            n_past, n_embd / n_head, 1),
        0, 2, 1, 3);

    struct dragon_tensor* KQ = dragon_mul_mat(ctx0, K, Q);
    struct dragon_tensor* KQ_scaled = dragon_scale(ctx0, KQ,
        dragon_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head)));
    struct dragon_tensor* KQ_soft_max = dragon_soft_max(ctx0, KQ_scaled);

    struct dragon_tensor* V_trans = dragon_permute(ctx0,
        dragon_reshape_3d(ctx0,
            dragon_view_1d(ctx0, memory_v, (n_past + N) * n_embd,
                il * n_ctx * dragon_element_size(memory_v) * n_embd),
            n_embd / n_head, n_head, n_past + N),
        1, 2, 0, 3);

    struct dragon_tensor* KQV = dragon_mul_mat(ctx0, V_trans, KQ_soft_max);
    struct dragon_tensor* KQV_merged = dragon_permute(ctx0, KQV, 0, 2, 1, 3);
    struct dragon_tensor* cur_out = dragon_cpy(ctx0, KQV_merged,
        dragon_new_tensor_2d(ctx0, DATA_TYPE_F32, n_embd, N));

    // 投影
    return dragon_mul_mat(ctx0, layer.wo, cur_out);
}

// 前馈网络层的前向传播
struct dragon_tensor* ffn_forward(
    struct dragon_context* ctx0,
    struct dragon_tensor* inpFF,
    const llama_layer& layer) {
    
    struct dragon_tensor* cur = dragon_rms_norm(ctx0, inpFF);
    cur = dragon_mul(ctx0, dragon_repeat(ctx0, layer.ffn_norm, cur), cur);

    struct dragon_tensor* tmp = dragon_mul_mat(ctx0, layer.w3, cur);
    cur = dragon_mul_mat(ctx0, layer.w1, cur);
    cur = dragon_silu(ctx0, cur);
    cur = dragon_mul(ctx0, cur, tmp);

    return dragon_mul_mat(ctx0, layer.w2, cur);
}

// 层归一化
struct dragon_tensor* layer_norm(
    struct dragon_context* ctx0,
    struct dragon_tensor* inp,
    struct dragon_tensor* norm) {
    
    struct dragon_tensor* cur = dragon_rms_norm(ctx0, inp);
    return dragon_mul(ctx0, dragon_repeat(ctx0, norm, cur), cur);
}

// 主评估函数
bool llama_eval(
    const llama_model& model,
    const int n_threads,
    const int n_past,
    const std::vector<gpt_vocab::id>& embd_inp,
    std::vector<float>& embd_w,
    size_t& mem_per_token) {
    
    try {
        const int N = embd_inp.size();
        const auto& hparams = model.hparams;
        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_head = hparams.n_head;

        // 初始化计算上下文
        static size_t buf_size = 512u * 1024 * 1024;
        static void* buf = nullptr;
        
        if (mem_per_token > 0 && mem_per_token * N > buf_size) {
            const size_t buf_size_new = 1.1 * (mem_per_token * N);
            buf_size = buf_size_new;
            buf = realloc(buf, buf_size);
            if (buf == nullptr) {
                throw std::runtime_error("Failed to allocate memory for compute context");
            }
        }

        struct dragon_init_params params = {
            .mem_size = buf_size,
            .mem_buffer = buf,
        };

        struct dragon_context* ctx0 = dragon_init(params);
        dragon_cgraph gf = {};
        gf.n_threads = n_threads;

        struct dragon_tensor* embd = dragon_new_tensor_1d(ctx0, DATA_TYPE_I32, N);
        memcpy(embd->data, embd_inp.data(), N * dragon_element_size(embd));

        struct dragon_tensor* inpL = dragon_get_rows(ctx0, model.tok_embeddings, embd);

        for (int il = 0; il < n_layer; ++il) {
            struct dragon_tensor* inpSA = inpL;
            
            // 注意力层
            struct dragon_tensor* cur = layer_norm(ctx0, inpL, model.layers[il].attention_norm);
            cur = attention_forward(ctx0, cur, model.layers[il], model.memory_k, model.memory_v,
                n_past, n_embd, n_head, n_ctx, N, il, &gf);
            
            struct dragon_tensor* inpFF = dragon_add(ctx0, cur, inpSA);
            
            // 前馈网络
            cur = ffn_forward(ctx0, inpFF, model.layers[il]);
            
            inpL = cur;
        }

        // 最终层归一化
        inpL = layer_norm(ctx0, inpL, model.norm);
        
        // 输出层
        inpL = dragon_mul_mat(ctx0, model.output, inpL);

        // 执行计算图
        dragon_build_forward_expand(&gf, inpL);
        dragon_graph_compute(ctx0, &gf);

        // 处理输出
        const int n_vocab = model.hparams.n_vocab;
        embd_w.resize(n_vocab);
        memcpy(embd_w.data(), 
               (float*)dragon_get_data(inpL) + (n_vocab * (N - 1)),
               sizeof(float) * n_vocab);

        if (mem_per_token == 0) {
            mem_per_token = dragon_used_mem(ctx0) / N;
        }

        dragon_free(ctx0);
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error in llama_eval: %s\n", e.what());
        return false;
    }
} 