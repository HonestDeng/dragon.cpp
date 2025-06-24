// Various helper functions and utilities

#pragma once

#include <string>
#include <map>
#include <vector>
#include <random>
#include <thread>

//
// CLI argument parsing
//

struct gpt_params {
    int32_t seed      = 2025; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict = 128; // new tokens to predict
    int32_t repeat_last_n = 64;  // last n tokens to penalize
    int32_t n_ctx = 512; //context size

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    float   repeat_penalty  = 1.30f;

    int32_t n_batch = 8; // batch size for prompt processing

    std::string model = "models/lamma-7B/dragon-model.bin"; // model path
    std::string prompt;

    bool use_color = false; // use color to distinguish generations and inputs

    bool interactive = false; // interactive mode
    bool interactive_start = false; // reverse prompt immediately
    std::string antiprompt = ""; // string upon seeing which more user input is prompted
};

bool gpt_params_parse(int argc, char ** argv, gpt_params & params);

void gpt_print_usage(int argc, char ** argv, const gpt_params & params);

std::string gpt_random_prompt(std::mt19937 & rng);

//
// Vocab utils
//

struct gpt_vocab {
    using id    = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
};

void replace(std::string & str, const std::string & needle, const std::string & replacement);

// poor-man's JSON parsing
std::map<std::string, int32_t> json_parse(const std::string & fname);

// split text into tokens
//
// ref: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
//
// Regex (Python):
// r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
//
// Regex (C++):
// R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"
//
std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab & vocab, const std::string & text);

// TODO: this is probably wrong, but I cannot figure out how this tokenizer works ..
// ref: https://github.com/google/sentencepiece
std::vector<gpt_vocab::id> llama_tokenize(const gpt_vocab & vocab, const std::string & text, bool bos);

// load the tokens from encoder.json
bool gpt_vocab_init(const std::string & fname, gpt_vocab & vocab);

// sample next token given probabilities for each embedding
//
//   - consider only the top K tokens
//   - from them, consider only the top tokens with cumulative probability > P
//
gpt_vocab::id llama_sample_top_p_top_k(
        const gpt_vocab & vocab,
        const float * logits,
        std::vector<gpt_vocab::id> & last_n_tokens,
        double repeat_penalty,
        int top_k,
        double top_p,
        double temp,
        std::mt19937 & rng);

// filer to top K tokens from list of logits
void sample_top_k(std::vector<std::pair<double, gpt_vocab::id>> & logits_id, int top_k);

//
// Quantization
//

size_t dragon_quantize_q4_0(float * src, void * dst, int n, int k, int qk, int64_t * hist);
size_t dragon_quantize_q4_1(float * src, void * dst, int n, int k, int qk, int64_t * hist);

// Forward declarations (if struct definitions are not moved to utils.h or a common header)
struct llama_hparams;
struct llama_model;
struct gpt_vocab;
struct dragon_context;

// default hparams (LLaMA 7B)
struct llama_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx = 512;  // this is provided as user input?
    int32_t n_embd = 4096;
    int32_t n_mult = 256;
    int32_t n_head = 32;
    int32_t n_layer = 32;
    int32_t n_rot = 64;
    int32_t f16 = 1;
};

struct llama_layer {
    // normalization
    struct dragon_tensor *attention_norm;

    // attention
    struct dragon_tensor *wq;
    struct dragon_tensor *wk;
    struct dragon_tensor *wv;
    struct dragon_tensor *wo;

    // normalization
    struct dragon_tensor *ffn_norm;

    // ff
    struct dragon_tensor *w1;
    struct dragon_tensor *w2;
    struct dragon_tensor *w3;
};

struct llama_model {
    llama_hparams hparams;

    struct dragon_tensor *tok_embeddings;

    struct dragon_tensor *norm;
    struct dragon_tensor *output;

    std::vector<llama_layer> layers;

    // key + value memory
    struct dragon_tensor *memory_k;
    struct dragon_tensor *memory_v;

    //
    struct dragon_context *ctx;
    std::map<std::string, struct dragon_tensor *> tensors;
};

// Helper functions for model loading
bool llama_model_load(const std::string &fname, llama_model &model,
                      gpt_vocab &vocab, int user_n_ctx);

bool load_hparams(std::ifstream &fin, llama_hparams &hparams, int user_n_ctx,
                  int &n_ff, int &n_parts);

bool load_vocab(std::ifstream &fin, gpt_vocab &vocab,
                const llama_hparams &hparams);

bool create_model_context_and_allocate_tensors(const std::string &fname,
                                             llama_model &model, int n_ff);

bool load_model_weights(const std::string &fname, int n_parts,
                        llama_model &model);
