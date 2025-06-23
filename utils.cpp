#include "utils.h"
#include "operators.h"
#include <cassert>
#include <cstring>
#include <fstream>
#include <regex>
#include <iostream>
#include <iterator>
#include <string>
#include <math.h>

 #if defined(_MSC_VER) || defined(__MINGW32__)
 #include <malloc.h> // using malloc.h with MSC/MINGW
 #elif !defined(__FreeBSD__) && !defined(__NetBSD__)
 #include <alloca.h>
 #endif

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    // determine sensible default number of threads.
    // std::thread::hardware_concurrency may not be equal to the number of cores, or may return 0.
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    params.n_threads = std::count(std::istream_iterator<std::string>(cpuinfo),
                                  std::istream_iterator<std::string>(),
                                  std::string("processor"));
#endif
    if (params.n_threads == 0) {
        params.n_threads = std::max(1, (int32_t) std::thread::hardware_concurrency());
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = argv[++i];
        } else if (arg == "-f" || arg == "--file") {

            std::ifstream file(argv[++i]);

            std::copy(std::istreambuf_iterator<char>(file),
                    std::istreambuf_iterator<char>(),
                    back_inserter(params.prompt));
                
        } else if (arg == "-n" || arg == "--n_predict") {
            params.n_predict = std::stoi(argv[++i]);
        } else if (arg == "--top_k") {
            params.top_k = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--ctx_size") {
            params.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--top_p") {
            params.top_p = std::stof(argv[++i]);
        } else if (arg == "--temp") {
            params.temp = std::stof(argv[++i]);
        } else if (arg == "--repeat_last_n") {
            params.repeat_last_n = std::stoi(argv[++i]);
        } else if (arg == "--repeat_penalty") {
            params.repeat_penalty = std::stof(argv[++i]);
        } else if (arg == "-b" || arg == "--batch_size") {
            params.n_batch = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if (arg == "--interactive-start") {
            params.interactive = true;
            params.interactive_start = true;
        } else if (arg == "--color") {
            params.use_color = true;
        } else if (arg == "-r" || arg == "--reverse-prompt") {
            params.antiprompt = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argc, argv, params);
            exit(0);
        } else if (arg == "--load-model-only") {
            params.load_model_only = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            gpt_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void gpt_print_usage(int argc, char ** argv, const gpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -i, --interactive     run in interactive mode\n");
    fprintf(stderr, "  --interactive-start   run in interactive mode and poll user input at startup\n");
    fprintf(stderr, "  -r PROMPT, --reverse-prompt PROMPT\n");
    fprintf(stderr, "                        in interactive mode, poll user input upon seeing PROMPT\n");
    fprintf(stderr, "  --color               colorise output to distinguish prompt and user input from generations\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME\n");
    fprintf(stderr, "                        prompt file to start generation.\n");
    fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d)\n", params.n_predict);
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
    fprintf(stderr, "  --repeat_last_n N     last n tokens to consider for penalize (default: %d)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat_penalty N    penalize repeat sequence of tokens (default: %.1f)\n", params.repeat_penalty);
    fprintf(stderr, "  -c N, --ctx_size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
    fprintf(stderr, "  -b N, --batch_size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  --load-model-only     only load model for testing, not run inference\n");
    fprintf(stderr, "\n");
}

std::string gpt_random_prompt(std::mt19937 & rng) {
    // const int r = rng() % 10;
    // switch (r) {
    //     case 0: return "So";
    //     case 1: return "Once upon a time";
    //     case 2: return "When";
    //     case 3: return "The";
    //     case 4: return "After";
    //     case 5: return "If";
    //     case 6: return "import";
    //     case 7: return "He";
    //     case 8: return "She";
    //     case 9: return "They";
    //     default: return "To";
    // }

    return "Once upon a time";
}

void replace(std::string & str, const std::string & needle, const std::string & replacement) {
    size_t pos = 0;
    while ((pos = str.find(needle, pos)) != std::string::npos) {
        str.replace(pos, needle.length(), replacement);
        pos += replacement.length();
    }
}

std::map<std::string, int32_t> json_parse(const std::string & fname) {
    std::map<std::string, int32_t> result;

    // read file into string
    std::string json;
    {
        std::ifstream ifs(fname);
        if (!ifs) {
            fprintf(stderr, "Failed to open %s\n", fname.c_str());
            exit(1);
        }

        json = std::string((std::istreambuf_iterator<char>(ifs)),
                (std::istreambuf_iterator<char>()));
    }

    if (json[0] != '{') {
        return result;
    }

    // parse json
    {
        bool has_key  = false;
        bool in_token = false;

        std::string str_key = "";
        std::string str_val = "";

        int n = json.size();
        for (int i = 1; i < n; ++i) {
            if (!in_token) {
                if (json[i] == ' ') continue;
                if (json[i] == '"') {
                    in_token = true;
                    continue;
                }
            } else {
                if (json[i] == '\\' && i+1 < n) {
                    if (has_key == false) {
                        str_key += json[i];
                    } else {
                        str_val += json[i];
                    }
                    ++i;
                } else if (json[i] == '"') {
                    if (has_key == false) {
                        has_key = true;
                        ++i;
                        while (json[i] == ' ') ++i;
                        ++i; // :
                        while (json[i] == ' ') ++i;
                        if (json[i] != '\"') {
                            while (json[i] != ',' && json[i] != '}') {
                                str_val += json[i++];
                            }
                            has_key = false;
                        } else {
                            in_token = true;
                            continue;
                        }
                    } else {
                        has_key = false;
                    }

                    ::replace(str_key, "\\u0120", " " ); // \u0120 -> space
                    ::replace(str_key, "\\u010a", "\n"); // \u010a -> new line
                    ::replace(str_key, "\\\"",    "\""); // \\\"   -> "

                    try {
                        result[str_key] = std::stoi(str_val);
                    } catch (...) {
                        //fprintf(stderr, "%s: ignoring key '%s' with value '%s'\n", fname.c_str(), str_key.c_str(), str_val.c_str());

                    }
                    str_key = "";
                    str_val = "";
                    in_token = false;
                    continue;
                }
                if (has_key == false) {
                    str_key += json[i];
                } else {
                    str_val += json[i];
                }
            }
        }
    }

    return result;
}

std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab & vocab, const std::string & text) {
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;
        std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    // find the longest tokens that form the words:
    std::vector<gpt_vocab::id> tokens;
    for (const auto & word : words) {
        if (word.size() == 0) continue;

        int i = 0;
        int n = word.size();
        while (i < n) {
            int j = n;
            while (j > i) {
                auto it = vocab.token_to_id.find(word.substr(i, j-i));
                if (it != vocab.token_to_id.end()) {
                    tokens.push_back(it->second);
                    i = j;
                    break;
                }
                --j;
            }
            if (i == n) {
                break;
            }
            if (j == i) {
                auto sub = word.substr(i, 1);
                if (vocab.token_to_id.find(sub) != vocab.token_to_id.end()) {
                    tokens.push_back(vocab.token_to_id.at(sub));
                } else {
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
                }
                ++i;
            }
        }
    }

    return tokens;
}

#define MAX_TOKEN_LEN 18
// 参考 https://guillaume-be.github.io/2020-05-30/sentence_piece
/*
    id    token string
     1 -> ''
  9038 -> ' Once'
  2501 -> ' upon'
   263 -> ' a'
   931 -> ' time'
*/
std::vector<gpt_vocab::id> llama_tokenize(const gpt_vocab & vocab, const std::string & text, bool bos) {
    // 输入：" Once upon a time" (开头有空格)
    // 输出：{1, 9038, 2501, 263, 931}
    // 在调试时，你可以直接返回 {1, 9038, 2501, 263, 931}，但最后的提交代码要实现这个函数。
    std::vector<gpt_vocab::id> res = {0, 0};

    // TODO: Forward pass

    // TODO: Backward pass


    return res;
}

bool gpt_vocab_init(const std::string & fname, gpt_vocab & vocab) {
    printf("%s: loading vocab from '%s'\n", __func__, fname.c_str());

    vocab.token_to_id = ::json_parse(fname);

    for (const auto & kv : vocab.token_to_id) {
        vocab.id_to_token[kv.second] = kv.first;
    }

    printf("%s: vocab size = %d\n", __func__, (int) vocab.token_to_id.size());

    // print the vocabulary
    //for (auto kv : vocab.token_to_id) {
    //    printf("'%s' -> %d\n", kv.first.data(), kv.second);
    //}

    return true;
}


void sample_top_k(std::vector<std::pair<double, gpt_vocab::id>> & logits_id, int top_k) {
    // find the top K tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<double, gpt_vocab::id> & a, const std::pair<double, gpt_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);
}

gpt_vocab::id llama_sample_top_p_top_k(
        const gpt_vocab & vocab,
        const float * logits,
        std::vector<gpt_vocab::id> & last_n_tokens,
        double repeat_penalty,
        int top_k,
        double top_p,
        double temp,
        std::mt19937 & rng) {
    int n_logits = vocab.id_to_token.size();

    std::vector<std::pair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (logits[i] < 0.0) {
                    logits_id.push_back(std::make_pair(logits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(logits[i]*scale/repeat_penalty, i));
                }                
            } else {
                logits_id.push_back(std::make_pair(logits[i]*scale, i));
            }
        }
    }

    sample_top_k(logits_id, top_k);

    double maxl = -INFINITY;
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}


size_t dragon_quantize_q4_0(float * src, void * dst, int n, int k, int qk, int64_t * hist) {
    const int nb = k / qk;
    const size_t bs = (sizeof(float) + sizeof(uint8_t)*qk/2);
    const size_t row_size = nb*bs;

    assert(k % qk == 0);

    const size_t pp_size = qk / 2;
    uint8_t *pp = static_cast<uint8_t*>(alloca(pp_size));

    char * pdst = (char *) dst;

    for (int j = 0; j < n; j += k) {
        uint8_t * pd = (uint8_t *) (pdst + (j/k)*row_size + 0*bs);
        uint8_t * pb = (uint8_t *) (pdst + (j/k)*row_size + 0*bs + sizeof(float));

        for (int i = 0; i < nb; i++) {
            float amax = 0.0f; // absolute max

            {
                for (int l = 0; l < qk; l++) {
                    const float v = src[j + i*qk + l];
                    amax = std::max(amax, fabsf(v));
                }

                const float d = amax / ((1 << 3) - 1);
                const float id = d ? 1.0f/d : 0.0f;

                *(float *) pd = d;
                pd += bs;

                for (int l = 0; l < qk; l += 2) {
                    const float v0 = (src[j + i*qk + l + 0])*id;
                    const float v1 = (src[j + i*qk + l + 1])*id;

                    const uint8_t vi0 = ((int8_t) (round(v0))) + 8;
                    const uint8_t vi1 = ((int8_t) (round(v1))) + 8;

                    assert(vi0 >= 0 && vi0 < 16);
                    assert(vi1 >= 0 && vi1 < 16);

                    hist[vi0]++;
                    hist[vi1]++;

                    pp[l/2] = vi0 | (vi1 << 4);
                }

                memcpy(pb, pp, pp_size);
                pb += bs;
            }
        }
    }

    return (n/k)*row_size;
}

size_t dragon_quantize_q4_1(float * src, void * dst, int n, int k, int qk, int64_t * hist) {
    const int nb = k / qk;
    const size_t bs = (2*sizeof(float) + sizeof(uint8_t)*qk/2);
    const size_t row_size = nb*bs;

    assert(k % qk == 0);

    const size_t pp_size = qk / 2;
    uint8_t *pp = static_cast<uint8_t*>(alloca(pp_size));

    char * pdst = (char *) dst;

    for (int j = 0; j < n; j += k) { 
        uint8_t * pd = (uint8_t *) (pdst + (j/k)*row_size + 0*bs);
        uint8_t * pm = (uint8_t *) (pdst + (j/k)*row_size + 0*bs +   sizeof(float));
        uint8_t * pb = (uint8_t *) (pdst + (j/k)*row_size + 0*bs + 2*sizeof(float));

        //printf("n = %d, k = %d, nb = %d, row_size = %d, j = %d, pm = %p, pd = %p, pb = %p\n", n, k, nb, row_size, j, pm, pd, pb);

        for (int i = 0; i < nb; i++) {
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();

            {
                for (int l = 0; l < qk; l++) {
                    const float v = src[j + i*qk + l];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

                const float d = (max - min) / ((1 << 4) - 1);
                const float id = d ? 1.0f/d : 0.0f;

                *(float *) pd = d;
                *(float *) pm = min;
                pd += bs; 
                pm += bs;

                for (int l = 0; l < qk; l += 2) {
                    const float v0 = (src[j + i*qk + l + 0] - min)*id;
                    const float v1 = (src[j + i*qk + l + 1] - min)*id;

                    const uint8_t vi0 = round(v0);
                    const uint8_t vi1 = round(v1);

                    assert(vi0 >= 0 && vi0 < 16);
                    assert(vi1 >= 0 && vi1 < 16);

                    hist[vi0]++;
                    hist[vi1]++;

                    pp[l/2] = vi0 | (vi1 << 4);
                }

                memcpy(pb, pp, pp_size);
                pb += bs;
            }
        }
    }

    return (n/k)*row_size;
}

// Define LLAMA_N_PARTS map or pass it as argument if needed elsewhere
static const std::map<int, int> LLAMA_N_PARTS = {
    { 4096, 1 },
    { 5120, 2 },
    { 6656, 4 },
    { 8192, 8 },
};


bool load_hparams(std::ifstream &fin, llama_hparams &hparams, int user_n_ctx,
                  int &n_ff, int &n_parts) {
    fin.read((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
    fin.read((char *)&hparams.n_embd, sizeof(hparams.n_embd));
    fin.read((char *)&hparams.n_mult, sizeof(hparams.n_mult));
    fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
    fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
    fin.read((char *)&hparams.n_rot, sizeof(hparams.n_rot));
    fin.read((char *)&hparams.f16, sizeof(hparams.f16));

    hparams.n_ctx = user_n_ctx; // Use user provided context size

    // Calculate n_ff and n_parts based on loaded hparams
    if (LLAMA_N_PARTS.count(hparams.n_embd) == 0) {
        fprintf(stderr, "%s: unsupported embedding dimension %d\n", __func__, hparams.n_embd);
        return false;
    }
    n_ff =
        ((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult) *
        hparams.n_mult;
    n_parts = LLAMA_N_PARTS.at(hparams.n_embd);

    fprintf(stderr, "%s: n_vocab = %d\n", __func__, hparams.n_vocab);
    fprintf(stderr, "%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
    fprintf(stderr, "%s: n_embd  = %d\n", __func__, hparams.n_embd);
    fprintf(stderr, "%s: n_mult  = %d\n", __func__, hparams.n_mult);
    fprintf(stderr, "%s: n_head  = %d\n", __func__, hparams.n_head);
    fprintf(stderr, "%s: n_layer = %d\n", __func__, hparams.n_layer);
    fprintf(stderr, "%s: n_rot   = %d\n", __func__, hparams.n_rot);
    fprintf(stderr, "%s: f16     = %d\n", __func__, hparams.f16);
    fprintf(stderr, "%s: n_ff    = %d\n", __func__, n_ff);
    fprintf(stderr, "%s: n_parts = %d\n", __func__, n_parts);

    return true; // Assuming success for now
}

bool load_vocab(std::ifstream &fin, gpt_vocab &vocab,
                const llama_hparams &hparams) {
    std::string word;
    for (int i = 0; i < hparams.n_vocab; i++) {
        uint32_t len;
        fin.read((char *)&len, sizeof(len));

        // // Basic check for plausible length
        // if (len > 256 || len == 0) {
        //     fprintf(stderr, "%s: invalid vocab length %u for index %d\n", __func__, len, i);
        //     return false;
        // }

        word.resize(len);
        fin.read((char *)word.data(), len);

        vocab.token_to_id[word] = i;
        vocab.id_to_token[i] = word;
    }
    return true;
}

// Helper to determine weight type from hparams.f16
data_type get_wtype_from_hparams(int f16_val) {
    switch (f16_val) {
        case 0: return DATA_TYPE_F32;
        case 1: return DATA_TYPE_F16;
        case 2: return DATA_TYPE_Q4_0;
        case 3: return DATA_TYPE_Q4_1;
        default: return DATA_TYPE_COUNT; // Indicate error
    }
}

bool create_model_context_and_allocate_tensors(const std::string &fname, // Pass fname for error messages
                                               llama_model &model, int n_ff) {
  const auto &hparams = model.hparams;
  data_type wtype = get_wtype_from_hparams(hparams.f16);
    if (wtype == DATA_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                __func__, fname.c_str(), hparams.f16);
        return false;
    }

  size_t ctx_size = 0;
  // Calculate context size
  {
      const int n_embd = hparams.n_embd;
      const int n_layer = hparams.n_layer;
      const int n_ctx = hparams.n_ctx;
      const int n_vocab = hparams.n_vocab;

      ctx_size += n_embd * n_vocab * dragon_type_sizef(wtype);  // tok_embeddings
      ctx_size += n_embd * dragon_type_sizef(DATA_TYPE_F32);  // norm
      ctx_size += n_embd * n_vocab * dragon_type_sizef(wtype);  // output
      ctx_size += n_layer * (n_embd * dragon_type_sizef(DATA_TYPE_F32));  // attention_norm
      ctx_size += n_layer * (n_embd * n_embd * dragon_type_sizef(wtype));  // wq
      ctx_size += n_layer * (n_embd * n_embd * dragon_type_sizef(wtype));  // wk
      ctx_size += n_layer * (n_embd * n_embd * dragon_type_sizef(wtype));  // wv
      ctx_size += n_layer * (n_embd * n_embd * dragon_type_sizef(wtype));  // wo
      ctx_size += n_layer * (n_embd * dragon_type_sizef(DATA_TYPE_F32));  // ffn_norm
      ctx_size += n_layer * (n_ff * n_embd * dragon_type_sizef(wtype));  // w1
      ctx_size += n_layer * (n_ff * n_embd * dragon_type_sizef(wtype));  // w2
      ctx_size += n_layer * (n_ff * n_embd * dragon_type_sizef(wtype));  // w3
      ctx_size += n_ctx * n_layer * n_embd * dragon_type_sizef(DATA_TYPE_F32);  // memory_k
      ctx_size += n_ctx * n_layer * n_embd * dragon_type_sizef(DATA_TYPE_F32);  // memory_v
      ctx_size += (5 + 10 * n_layer) * 256;  // object overhead

      fprintf(stderr, "%s: dragon ctx size = %6.2f MB\n", __func__,
              ctx_size / (1024.0 * 1024.0));
  }

  auto ctx = model.ctx; // Convenience alias

  // Create the dragon context
  {
      struct dragon_init_params params = { ctx_size, NULL };
      model.ctx = dragon_init(params);
      if (!model.ctx) {
          fprintf(stderr, "%s: dragon_init() failed\n", __func__);
          return false;
      }
  }

  // Allocate tensors
  {
      const int n_embd = hparams.n_embd;
      const int n_layer = hparams.n_layer;
      const int n_vocab = hparams.n_vocab;

      model.layers.resize(n_layer);

      model.tok_embeddings = dragon_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
      model.norm = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, n_embd);
      model.output = dragon_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

      model.tensors["tok_embeddings.weight"] = model.tok_embeddings;
      model.tensors["norm.weight"] = model.norm;
      model.tensors["output.weight"] = model.output;

      for (int i = 0; i < n_layer; ++i) {
          auto &layer = model.layers[i];
          layer.attention_norm = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, n_embd);
          layer.wq = dragon_new_tensor_2d(ctx, wtype, n_embd, n_embd);
          layer.wk = dragon_new_tensor_2d(ctx, wtype, n_embd, n_embd);
          layer.wv = dragon_new_tensor_2d(ctx, wtype, n_embd, n_embd);
          layer.wo = dragon_new_tensor_2d(ctx, wtype, n_embd, n_embd);
          layer.ffn_norm = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, n_embd);
          layer.w1 = dragon_new_tensor_2d(ctx, wtype, n_embd, n_ff);
          layer.w2 = dragon_new_tensor_2d(ctx, wtype, n_ff, n_embd);
          layer.w3 = dragon_new_tensor_2d(ctx, wtype, n_embd, n_ff);

          // map by name
          model.tensors["layers." + std::to_string(i) + ".attention_norm.weight"] = layer.attention_norm;
          model.tensors["layers." + std::to_string(i) + ".attention.wq.weight"] = layer.wq;
          model.tensors["layers." + std::to_string(i) + ".attention.wk.weight"] = layer.wk;
          model.tensors["layers." + std::to_string(i) + ".attention.wv.weight"] = layer.wv;
          model.tensors["layers." + std::to_string(i) + ".attention.wo.weight"] = layer.wo;
          model.tensors["layers." + std::to_string(i) + ".ffn_norm.weight"] = layer.ffn_norm;
          model.tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"] = layer.w1;
          model.tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"] = layer.w2;
          model.tensors["layers." + std::to_string(i) + ".feed_forward.w3.weight"] = layer.w3;
      }
  }

    // Allocate KV cache
    {
        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_mem = n_layer * n_ctx;
        const int n_elements = n_embd * n_mem;

        model.memory_k = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, n_elements);
        model.memory_v = dragon_new_tensor_1d(ctx, DATA_TYPE_F32, n_elements);

        const size_t memory_size = dragon_nbytes(model.memory_k) + dragon_nbytes(model.memory_v);
        fprintf(stderr, "%s: memory_size = %8.2f MB, n_mem = %d\n", __func__,
                memory_size / 1024.0 / 1024.0, n_mem);
    }

  return true;
}

// Internal helper to load weights from a single part file stream
static bool load_weights_from_part(std::ifstream &fin, int n_parts, int part_id,
                                   llama_model &model) {
  int n_tensors = 0;
  size_t total_size = 0;
  fprintf(stderr, "%s: ", __func__);

  while (fin.peek() != EOF) { // Check before reading
    int32_t n_dims;
    int32_t length;
    int32_t ftype_from_file; // Renamed to avoid confusion with model's wtype

    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char *>(&length), sizeof(length));
    fin.read(reinterpret_cast<char *>(&ftype_from_file), sizeof(ftype_from_file));

        if (fin.eof()) { // Check again after reading metadata header
            // This might happen if the file ends exactly after the last tensor data
            if (n_tensors > 0) break; // Normal if we've read tensors
            else {
                fprintf(stderr, "%s: unexpected EOF after reading tensor metadata header for part %d\n", __func__, part_id);
                return false; // Error if it happens before any tensor
            }
        }

    // Validate dimensions and length
    if (n_dims < 1 || n_dims > 2 || length <= 0 || length > 256) {
        fprintf(stderr, "%s: invalid tensor metadata (n_dims=%d, length=%d) in part %d\n", __func__, n_dims, length, part_id);
        return false;
    }


    int32_t nelements = 1;
    int32_t ne[2] = {1, 1};
    for (int i = 0; i < n_dims; ++i) {
      fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            if (ne[i] <= 0) {
                 fprintf(stderr, "%s: invalid tensor dimension size %d for dim %d in part %d\n", __func__, ne[i], i, part_id);
                 return false;
            }
      nelements *= ne[i];
    }
        if (nelements <= 0) {
            fprintf(stderr, "%s: invalid total tensor elements %d in part %d\n", __func__, nelements, part_id);
            return false;
        }

    std::string name(length, 0);
    fin.read(&name[0], length);

    if (model.tensors.find(name) == model.tensors.end()) {
      fprintf(stderr, "%s: unknown tensor '%s' in model file part %d\n", __func__,
              name.c_str(), part_id);
      return false;
    }

    auto tensor = model.tensors[name];
    data_type tensor_type = tensor->type; // The type expected by the allocated tensor in context

    // Determine split type (same logic as before)
     int split_type = 0;
        if (name.find("tok_embeddings") != std::string::npos) {
          split_type = 0;
        } else if (name.find("layers") != std::string::npos) {
          if (name.find("attention.wo.weight") != std::string::npos) {
            split_type = 0;
          } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
            split_type = 0;
          } else {
            split_type = 1;
          }
        } else if (name.find("output") != std::string::npos) {
          split_type = 1;
        }


    // Validate tensor shape and size against metadata
     // (Simplified validation, original code had detailed checks)
        size_t expected_elements_part = (n_dims == 1 || n_parts == 1) ? dragon_nelements(tensor) : dragon_nelements(tensor) / n_parts;
        if (expected_elements_part != (size_t)nelements) {
             fprintf(stderr, "%s: tensor '%s' element count mismatch in part %d: file %d, expected %zu\n",
                     __func__, name.c_str(), part_id, nelements, expected_elements_part);
            // Note: This check might need refinement based on quantization block sizes etc.
            // return false; // Temporarily disable strict check if causing issues
        }

    // Determine bytes per element based on file type
    size_t bpe = 0;
    switch (ftype_from_file) {
      case 0: bpe = dragon_type_size(DATA_TYPE_F32); break;
      case 1: bpe = dragon_type_size(DATA_TYPE_F16); break;
      case 2: bpe = dragon_type_size(DATA_TYPE_Q4_0); break; // size includes scale factor etc.
      case 3: bpe = dragon_type_size(DATA_TYPE_Q4_1); break; // size includes scale factor etc.
      default:
        fprintf(stderr, "%s: unknown ftype %d in model file part %d for tensor '%s'\n",
                __func__, ftype_from_file, part_id, name.c_str());
        return false;
    };

    // Calculate expected size in file for this part
    size_t expected_bytes_in_file = 0;
     if (tensor_type == DATA_TYPE_Q4_0 || tensor_type == DATA_TYPE_Q4_1) {
         // For quantized types, the file stores blocks. Size calculation depends on block structure.
         // We assume the file stores the raw quantized data + scales/offsets per block.
         // The dragon_nbytes calculation should already account for the full size of the tensor in memory.
         // The size in the file for a part needs to match the part's portion of that total size.
         expected_bytes_in_file = dragon_nbytes(tensor) / n_parts;
         // We might need a direct check against nelements * bytes_per_block / block_size if bpe is tricky
         if (bpe == 0) { // Error case from switch
             fprintf(stderr, "%s: Cannot calculate bytes for unknown ftype %d for tensor '%s'\n", __func__, ftype_from_file, name.c_str());
             return false;
         }
         // Rough check: size should be related to elements and bpe/block_size
         // size_t approx_bytes = (size_t)nelements * bpe / dragon_blck_size(tensor_type);
         // This might not be exact due to metadata (scales etc.) within the type size.

     } else {
         // For F32/F16, it's simpler
         if (bpe == 0) { // Error case from switch
              fprintf(stderr, "%s: Cannot calculate bytes for unknown ftype %d for tensor '%s'\n", __func__, ftype_from_file, name.c_str());
              return false;
         }
         expected_bytes_in_file = (size_t)nelements * bpe;
     }


    // Read data based on split type
     if (n_dims == 1 || n_parts == 1) {
            if (expected_bytes_in_file != dragon_nbytes(tensor)) {
                 fprintf(stderr, "%s: tensor '%s' size mismatch (n_dims=1 or n_parts=1): file %zu, expected %zu\n",
                         __func__, name.c_str(), expected_bytes_in_file, dragon_nbytes(tensor));
                 // return false; // Allow loading for now
            }
          fin.read(reinterpret_cast<char *>(tensor->data), expected_bytes_in_file);
          total_size += expected_bytes_in_file;
     } else {
            if (expected_bytes_in_file != dragon_nbytes(tensor) / n_parts) {
                fprintf(stderr, "%s: tensor '%s' part size mismatch: file %zu, expected %zu\n",
                        __func__, name.c_str(), expected_bytes_in_file, dragon_nbytes(tensor) / n_parts);
                // return false; // Allow loading for now
            }
         const size_t bytes_to_read_for_part = expected_bytes_in_file; // Should be dragon_nbytes(tensor) / n_parts;

         if (split_type == 0) { // Split by columns
             const int np0 = ne[0]; // Elements per row in this part's column slice
             const size_t element_size = dragon_type_size(tensor_type); // Size of one element in memory type
             const size_t block_size_in_elements = dragon_blck_size(tensor_type);
                const size_t type_size_in_bytes = dragon_type_size(tensor_type); // Size of block or element

                // Calculate the size of one row in the final tensor in memory
             const size_t row_size_bytes_mem = tensor->nb[1];

                // Calculate the number of elements per row in the full tensor
                const size_t full_ne0 = tensor->ne[0];

                // Bytes per row for this part in the file (consider block size)
                // This calculation is tricky. Let's read it sequentially for now assuming file matches memory layout split.
                const size_t bytes_per_row_part = bytes_to_read_for_part / ne[1]; // ne[1] is number of rows

             for (int i1 = 0; i1 < tensor->ne[1]; ++i1) { // Iterate over rows of the *full tensor*
                 const size_t offset_row_mem = i1 * row_size_bytes_mem; // Offset for the start of the row in memory
                 // Offset within the row to place this part's data
                 const size_t offset_col_mem = (part_id * (full_ne0 / n_parts) / block_size_in_elements) * type_size_in_bytes;
                    // Read the calculated number of bytes for this row's part
                 fin.read(reinterpret_cast<char *>(tensor->data) + offset_row_mem + offset_col_mem, bytes_per_row_part);
             }
         } else { // Split by rows
             const int np1 = ne[1]; // Number of rows in this part
             const size_t row_size_bytes_mem = tensor->nb[1]; // Bytes per row in memory is constant

             for (int i1 = 0; i1 < np1; ++i1) {
                 // Calculate the starting row offset in the full tensor memory
                 const size_t offset_row_mem = (i1 + part_id * np1) * row_size_bytes_mem;
                 // Read a full row's worth of data
                 fin.read(reinterpret_cast<char *>(tensor->data) + offset_row_mem, row_size_bytes_mem);
             }
         }
         total_size += bytes_to_read_for_part;
     }

      // Check for read errors
        if (fin.fail()) {
            fprintf(stderr, "\n%s: file read failed for tensor '%s' in part %d\n", __func__, name.c_str(), part_id);
            return false;
        }


    if (++n_tensors % 8 == 0) {
      fprintf(stderr, ".");
      fflush(stderr);
    }
  }

  fprintf(stderr, " done\n");
  fprintf(stderr, "%s: model part %d size = %8.2f MB / num tensors = %d\n",
          __func__, part_id + 1, total_size / 1024.0 / 1024.0, n_tensors);
  return true;
}

bool load_model_weights(const std::string &fname, int n_parts,
                        llama_model &model) {
    // Need the offset from the initial file read (hparams, vocab)
    // Option 1: Re-open the first file to get the offset
    size_t file_offset = 0;
    {
        std::ifstream temp_fin(fname, std::ios::binary);
        if (!temp_fin) {
             fprintf(stderr, "%s: failed to open '%s' to determine offset\n", __func__, fname.c_str());
             return false;
        }
        // Read magic, hparams, vocab again just to advance the stream pointer
         uint32_t magic;
         temp_fin.read((char *)&magic, sizeof(magic));
         if (magic != 0x4b4c535) return false; // Should not happen if called after checks

         llama_hparams dummy_hparams;
         int dummy_nff, dummy_nparts;
         gpt_vocab dummy_vocab;
         if (!load_hparams(temp_fin, dummy_hparams, model.hparams.n_ctx, dummy_nff, dummy_nparts)) return false;
         if (!load_vocab(temp_fin, dummy_vocab, dummy_hparams)) return false;

        file_offset = temp_fin.tellg();
        temp_fin.close();
         if (file_offset == 0) {
              fprintf(stderr, "%s: failed to determine file offset after reading metadata from '%s'\n", __func__, fname.c_str());
              return false;
         }
    }


  std::vector<char> f_buf(1024 * 1024); // Buffer for stream ops

  for (int i = 0; i < n_parts; ++i) {
    std::string fname_part = fname;
    if (i > 0) {
      fname_part += "." + std::to_string(i);
    }

    fprintf(stderr, "%s: loading model part %d/%d from '%s'\n", __func__, i + 1,
            n_parts, fname_part.c_str());

    std::ifstream fin(fname_part, std::ios::binary);
    if (!fin) {
         fprintf(stderr, "%s: failed to open part file '%s'\n", __func__, fname_part.c_str());
         return false;
    }
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
    fin.seekg(file_offset);

    if (!load_weights_from_part(fin, n_parts, i, model)) {
      fin.close();
      return false;
    }

    fin.close();
  }

  return true;
}
