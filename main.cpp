#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "operators.h"
#include "utils.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <signal.h>
#endif

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_BOLD "\x1b[1m"

// determine number of model parts based on the dimension
static const std::map<int, int> LLAMA_N_PARTS = {
    {4096, 1},
    {5120, 2},
    {6656, 4},
    {8192, 8},
};

// load the model's weights from a file
bool llama_model_load(const std::string &fname, llama_model &model,
                      gpt_vocab &vocab, int user_n_ctx) { // Changed n_ctx to user_n_ctx for clarity
  fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__,
          fname.c_str());

  // Open file and check magic number
  std::ifstream fin(fname, std::ios::binary);
  std::vector<char> f_buf(1024 * 1024); // Optional: buffer for initial reads
  fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());

  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    return false;
  }

  {
    uint32_t magic;
    fin.read((char *)&magic, sizeof(magic));
    if (magic != 0x67676d6c) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
              fname.c_str());
      return false;
    }
  }

  int n_ff = 0;
  int n_parts = 0;

  // Load hparams using helper
  if (!load_hparams(fin, model.hparams, user_n_ctx, n_ff, n_parts)) {
    fin.close();
    return false;
  }

  // Load vocab using helper
  if (!load_vocab(fin, vocab, model.hparams)) {
    fin.close();
    return false;
  }

  // Close the initial file stream as metadata is read
  fin.close();


  // Create context and allocate tensors using helper
  if (!create_model_context_and_allocate_tensors(fname, model, n_ff)) {
    // Context creation failed, model.ctx might be invalid, cleanup?
    // dragon_free(model.ctx); // Maybe add a cleanup helper too
    return false;
  }

  // Load weights using helper
  if (!load_model_weights(fname, n_parts, model)) {
    // Weight loading failed, cleanup?
    // dragon_free(model.ctx);
    return false;
  }

  // All steps successful
  fprintf(stderr, "%s: model loaded successfully\n", __func__);
  return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//
bool llama_eval(const llama_model &model, const int n_threads, const int n_past,
                const std::vector<gpt_vocab::id> &embd_inp,
                std::vector<float> &embd_w, size_t &mem_per_token) {
  const int N = embd_inp.size();

  const auto &hparams = model.hparams;

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = hparams.n_ctx;
  const int n_head = hparams.n_head;
  const int n_vocab = hparams.n_vocab;
  const int n_rot = hparams.n_embd / hparams.n_head;

  const int d_key = n_embd / n_head;

  // TODO: check if this size scales with n_ctx linearly and remove constant.
  // somehow I feel it wasn't the case
  // static size_t buf_size = hparams.n_ctx*1024*1024;
  static size_t buf_size = 512u * 1024 * 1024;
  static void *buf = malloc(buf_size);

  if (mem_per_token > 0 && mem_per_token * N > buf_size) {
    const size_t buf_size_new =
        1.1 *
        (mem_per_token * N);  // add 10% to account for dragon object overhead
    // fprintf(stderr, "\n%s: reallocating buffer from %zu to %zu bytes\n",
    // __func__, buf_size, buf_size_new);

    // reallocate
    buf_size = buf_size_new;
    buf = realloc(buf, buf_size);
    if (buf == nullptr) {
      fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
      return false;
    }
  }

  struct dragon_init_params params = {
      /*.mem_size   =*/buf_size,
      /*.mem_buffer =*/buf,
  };

  struct dragon_context *ctx0 = dragon_init(params);
  dragon_cgraph gf = {};
  gf.n_threads = n_threads;

  struct dragon_tensor *embd = dragon_new_tensor_1d(ctx0, DATA_TYPE_I32, N);
  memcpy(embd->data, embd_inp.data(), N * dragon_element_size(embd));

  struct dragon_tensor *inpL =
      dragon_get_rows(ctx0, model.tok_embeddings, embd);

  for (int il = 0; il < n_layer; ++il) {
    struct dragon_tensor *inpSA = inpL;

    struct dragon_tensor *cur;

    // norm
    {
      cur = dragon_rms_norm(ctx0, inpL);

      // cur = attention_norm*cur
      cur = dragon_mul(
          ctx0, dragon_repeat(ctx0, model.layers[il].attention_norm, cur), cur);
    }

    // self-attention
    {
      struct dragon_tensor *Qcur =
          dragon_mul_mat(ctx0, model.layers[il].wq, cur);
      struct dragon_tensor *Kcur =
          dragon_mul_mat(ctx0, model.layers[il].wk, cur);
      struct dragon_tensor *Vcur =
          dragon_mul_mat(ctx0, model.layers[il].wv, cur);

      // store key and value to memory
      if (N >= 1) {
        struct dragon_tensor *k =
            dragon_view_1d(ctx0, model.memory_k, N * n_embd,
                           (dragon_element_size(model.memory_k) * n_embd) *
                               (il * n_ctx + n_past));
        struct dragon_tensor *v =
            dragon_view_1d(ctx0, model.memory_v, N * n_embd,
                           (dragon_element_size(model.memory_v) * n_embd) *
                               (il * n_ctx + n_past));

        dragon_build_forward_expand(&gf, dragon_cpy(ctx0, Kcur, k));
        dragon_build_forward_expand(&gf, dragon_cpy(ctx0, Vcur, v));
      }

      // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1,
      // 3)
      struct dragon_tensor *Q = dragon_permute(
          ctx0,
          dragon_rope(
              ctx0,
              dragon_cpy(ctx0, Qcur,
                         dragon_new_tensor_3d(ctx0, DATA_TYPE_F32,
                                              n_embd / n_head, n_head, N)),
              n_past, n_rot, 0),
          0, 2, 1, 3);

      // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
      struct dragon_tensor *K = dragon_permute(
          ctx0,
          dragon_rope(ctx0,
                      dragon_reshape_3d(
                          ctx0,
                          dragon_view_1d(
                              ctx0, model.memory_k, (n_past + N) * n_embd,
                              il * n_ctx * dragon_element_size(model.memory_k) *
                                  n_embd),
                          n_embd / n_head, n_head, n_past + N),
                      n_past, n_rot, 1),
          0, 2, 1, 3);

      // K * Q
      struct dragon_tensor *KQ = dragon_mul_mat(ctx0, K, Q);

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      struct dragon_tensor *KQ_scaled = dragon_scale(
          ctx0, KQ, dragon_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head)));

      // KQ_masked = mask_past(KQ_scaled)
      struct dragon_tensor *KQ_masked =
          dragon_diag_mask_inf(ctx0, KQ_scaled, n_past);

      // KQ = soft_max(KQ_masked)
      struct dragon_tensor *KQ_soft_max = dragon_soft_max(ctx0, KQ_masked);

      // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0,
      // 3).contiguous()
      struct dragon_tensor *V_trans = dragon_permute(
          ctx0,
          dragon_reshape_3d(
              ctx0,
              dragon_view_1d(
                  ctx0, model.memory_v, (n_past + N) * n_embd,
                  il * n_ctx * dragon_element_size(model.memory_v) * n_embd),
              n_embd / n_head, n_head, n_past + N),
          1, 2, 0, 3);

      // KQV = transpose(V) * KQ_soft_max
      struct dragon_tensor *KQV = dragon_mul_mat(ctx0, V_trans, KQ_soft_max);

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct dragon_tensor *KQV_merged = dragon_permute(ctx0, KQV, 0, 2, 1, 3);

      // cur = KQV_merged.contiguous().view(n_embd, N)
      cur = dragon_cpy(ctx0, KQV_merged,
                       dragon_new_tensor_2d(ctx0, DATA_TYPE_F32, n_embd, N));

      // projection (no bias)
      cur = dragon_mul_mat(ctx0, model.layers[il].wo, cur);
    }

    struct dragon_tensor *inpFF = dragon_add(ctx0, cur, inpSA);

    // feed-forward network
    {
      // norm
      {
        cur = dragon_rms_norm(ctx0, inpFF);

        // cur = ffn_norm*cur
        cur = dragon_mul(
            ctx0, dragon_repeat(ctx0, model.layers[il].ffn_norm, cur), cur);
      }

      struct dragon_tensor *tmp =
          dragon_mul_mat(ctx0, model.layers[il].w3, cur);

      cur = dragon_mul_mat(ctx0, model.layers[il].w1, cur);

      // SILU activation
      cur = dragon_silu(ctx0, cur);

      cur = dragon_mul(ctx0, cur, tmp);

      cur = dragon_mul_mat(ctx0, model.layers[il].w2, cur);
    }

    cur = dragon_add(ctx0, cur, inpFF);

    // input for next layer
    inpL = cur;
  }

  // norm
  {
    inpL = dragon_rms_norm(ctx0, inpL);

    // inpL = norm*inpL
    inpL = dragon_mul(ctx0, dragon_repeat(ctx0, model.norm, inpL), inpL);
  }

  // lm_head
  {
    inpL = dragon_mul_mat(ctx0, model.output, inpL);
  }

  // logits -> probs
  // inpL = dragon_soft_max(ctx0, inpL);

  // run the computation
  dragon_build_forward_expand(&gf, inpL);
  dragon_graph_compute(ctx0, &gf);

  // if (n_past%100 == 0) {
  //     dragon_graph_print   (&gf);
  //     dragon_graph_dump_dot(&gf, NULL, "gpt-2.dot");
  // }

  // embd_w.resize(n_vocab*N);
  // memcpy(embd_w.data(), dragon_get_data(inpL), sizeof(float)*n_vocab*N);

  // return result for just the last token
  embd_w.resize(n_vocab);
  memcpy(embd_w.data(), (float *)dragon_get_data(inpL) + (n_vocab * (N - 1)),
         sizeof(float) * n_vocab);

  if (mem_per_token == 0) {
    mem_per_token = dragon_used_mem(ctx0) / N;
  }
  // fprintf(stderr, "used_mem = %zu\n", dragon_used_mem(ctx0));

  dragon_free(ctx0);

  return true;
}

static bool is_interacting = false;

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || \
    defined(_WIN32)
void sigint_handler(int signo) {
  printf(ANSI_COLOR_RESET);
  if (signo == SIGINT) {
    if (!is_interacting) {
      is_interacting = true;
    } else {
      _exit(130);
    }
  }
}
#endif

const char *llama_print_system_info(void) {
  static std::string s;

  s = "";
  s += "AVX = " + std::to_string(dragon_cpu_has_avx()) + " | ";
  s += "AVX2 = " + std::to_string(dragon_cpu_has_avx2()) + " | ";
  s += "AVX512 = " + std::to_string(dragon_cpu_has_avx512()) + " | ";
  s += "FMA = " + std::to_string(dragon_cpu_has_fma()) + " | ";
  s += "NEON = " + std::to_string(dragon_cpu_has_neon()) + " | ";
  s += "ARM_FMA = " + std::to_string(dragon_cpu_has_arm_fma()) + " | ";
  s += "F16C = " + std::to_string(dragon_cpu_has_f16c()) + " | ";
  s += "FP16_VA = " + std::to_string(dragon_cpu_has_fp16_va()) + " | ";
  s += "WASM_SIMD = " + std::to_string(dragon_cpu_has_wasm_simd()) + " | ";
  s += "BLAS = " + std::to_string(dragon_cpu_has_blas()) + " | ";
  s += "SSE3 = " + std::to_string(dragon_cpu_has_sse3()) + " | ";
  s += "VSX = " + std::to_string(dragon_cpu_has_vsx()) + " | ";

  return s.c_str();
}

int main(int argc, char **argv) {
  dragon_time_init();
  const int64_t t_main_start_us = dragon_time_us();

  gpt_params params;
  params.model = "models/llama-7B/dragon-model.bin";

  if (gpt_params_parse(argc, argv, params) == false) {
    return 1;
  }

  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

  std::mt19937 rng(params.seed);
  if (params.prompt.empty()) {
    params.prompt = gpt_random_prompt(rng);
  }

  //    params.prompt = R"(// this function checks if the number n is prime
  // bool is_prime(int n) {)";

  int64_t t_load_us = 0;

  gpt_vocab vocab;
  llama_model model;

  // load the model
  {
    const int64_t t_start_us = dragon_time_us();
    if (!llama_model_load(params.model, model, vocab, params.n_ctx)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__,
              params.model.c_str());
      return 1;
    }

    t_load_us = dragon_time_us() - t_start_us;
  }

  // print system information
  {
    fprintf(stderr, "\n");
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n", params.n_threads,
            std::thread::hardware_concurrency(), llama_print_system_info());
  }

  int n_past = 0;

  int64_t t_sample_us = 0;
  int64_t t_predict_us = 0;

  std::vector<float> logits;

  // Add a space in front of the first character to match OG llama tokenizer
  // behavior
  params.prompt.insert(0, 1, ' ');
  // tokenize the prompt
  std::vector<gpt_vocab::id> embd_inp =
      ::llama_tokenize(vocab, params.prompt, true);

  params.n_predict =
      std::min(params.n_predict, model.hparams.n_ctx - (int)embd_inp.size());

  // tokenize the reverse prompt
  std::vector<gpt_vocab::id> antiprompt_inp =
      ::llama_tokenize(vocab, params.antiprompt, false);

  fprintf(stderr, "\n");
  fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
  fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__,
          embd_inp.size());
  for (int i = 0; i < (int)embd_inp.size(); i++) {
    fprintf(stderr, "%6d -> '%s'\n", embd_inp[i],
            vocab.id_to_token.at(embd_inp[i]).c_str());
  }
  fprintf(stderr, "\n");
  if (params.interactive) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
    signal(SIGINT, sigint_handler);
#endif

    fprintf(stderr, "%s: interactive mode on.\n", __func__);

    if (antiprompt_inp.size()) {
      fprintf(stderr, "%s: reverse prompt: '%s'\n", __func__,
              params.antiprompt.c_str());
      fprintf(stderr, "%s: number of tokens in reverse prompt = %zu\n",
              __func__, antiprompt_inp.size());
      for (int i = 0; i < (int)antiprompt_inp.size(); i++) {
        fprintf(stderr, "%6d -> '%s'\n", antiprompt_inp[i],
                vocab.id_to_token.at(antiprompt_inp[i]).c_str());
      }
      fprintf(stderr, "\n");
    }
  }
  fprintf(stderr,
          "sampling parameters: temp = %f, top_k = %d, top_p = %f, "
          "repeat_last_n = %i, repeat_penalty = %f\n",
          params.temp, params.top_k, params.top_p, params.repeat_last_n,
          params.repeat_penalty);
  fprintf(stderr, "\n\n");

  std::vector<gpt_vocab::id> embd;

  // determine the required inference memory per token:
  size_t mem_per_token = 0;
  llama_eval(model, params.n_threads, 0, {0, 1, 2, 3}, logits, mem_per_token);

  int last_n_size = params.repeat_last_n;
  std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
  std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

  if (params.interactive) {
    fprintf(stderr,
            "== Running in interactive mode. ==\n"
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || \
    defined(_WIN32)
            " - Press Ctrl+C to interject at any time.\n"
#endif
            " - Press Return to return control to LLaMa.\n"
            " - If you want to submit another line, end your input in '\\'.\n");
  }

  int remaining_tokens = params.n_predict;
  int input_consumed = 0;
  bool input_noecho = false;

  // prompt user immediately after the starting prompt has been loaded
  if (params.interactive_start) {
    is_interacting = true;
  }

  // set the color for the prompt which will be output initially
  if (params.use_color) {
    printf(ANSI_COLOR_YELLOW);
  }

  while (remaining_tokens > 0) {
    // predict
    if (embd.size() > 0) {
      const int64_t t_start_us = dragon_time_us();

      if (!llama_eval(model, params.n_threads, n_past, embd, logits,
                      mem_per_token)) {
        fprintf(stderr, "Failed to predict\n");
        return 1;
      }

      t_predict_us += dragon_time_us() - t_start_us;
    }

    n_past += embd.size();
    embd.clear();

    if (embd_inp.size() <= input_consumed) {
      // out of user input, sample next token
      const float top_k = params.top_k;
      const float top_p = params.top_p;
      const float temp = params.temp;
      const float repeat_penalty = params.repeat_penalty;

      const int n_vocab = model.hparams.n_vocab;

      gpt_vocab::id id = 0;

      {
        const int64_t t_start_sample_us = dragon_time_us();

        id = llama_sample_top_p_top_k(
            vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens,
            repeat_penalty, top_k, top_p, temp, rng);

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);

        t_sample_us += dragon_time_us() - t_start_sample_us;
      }

      // add it to the context
      embd.push_back(id);

      // echo this to console
      input_noecho = false;

      // decrement remaining sampling budget
      --remaining_tokens;
    } else {
      // some user input remains from prompt or interaction, forward it to
      // processing
      while (embd_inp.size() > input_consumed) {
        embd.push_back(embd_inp[input_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(embd_inp[input_consumed]);
        ++input_consumed;
        if (embd.size() > params.n_batch) {
          break;
        }
      }

      // reset color to default if we there is no pending user input
      if (!input_noecho && params.use_color &&
          embd_inp.size() == input_consumed) {
        printf(ANSI_COLOR_RESET);
      }
    }

    // display text
    if (!input_noecho) {
      for (auto id : embd) {
        printf("%s", vocab.id_to_token[id].c_str());
      }
      fflush(stdout);
    }

    // in interactive mode, and not currently processing queued inputs;
    // check if we should prompt the user for more
    if (params.interactive && embd_inp.size() <= input_consumed) {
      // check for reverse prompt
      if (antiprompt_inp.size() &&
          std::equal(antiprompt_inp.rbegin(), antiprompt_inp.rend(),
                     last_n_tokens.rbegin())) {
        // reverse prompt found
        is_interacting = true;
      }
      if (is_interacting) {
        // currently being interactive
        bool another_line = true;
        while (another_line) {
          fflush(stdout);
          char buf[256] = {0};
          int n_read;
          if (params.use_color) printf(ANSI_BOLD ANSI_COLOR_GREEN);
          if (scanf("%255[^\n]%n%*c", buf, &n_read) <= 0) {
            // presumable empty line, consume the newline
            std::ignore = scanf("%*c");
            n_read = 0;
          }
          if (params.use_color) printf(ANSI_COLOR_RESET);

          if (n_read > 0 && buf[n_read - 1] == '\\') {
            another_line = true;
            buf[n_read - 1] = '\n';
            buf[n_read] = 0;
          } else {
            another_line = false;
            buf[n_read] = '\n';
            buf[n_read + 1] = 0;
          }

          std::vector<gpt_vocab::id> line_inp =
              ::llama_tokenize(vocab, buf, false);
          embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

          remaining_tokens -= line_inp.size();

          input_noecho = true;  // do not echo this again
        }

        is_interacting = false;
      }
    }

    // end of text token
    if (embd.back() == 2) {
      fprintf(stderr, " [end of text]\n");
      break;
    }
  }

#if defined(_WIN32)
  signal(SIGINT, SIG_DFL);
#endif

  // report timing
  {
    const int64_t t_main_end_us = dragon_time_us();

    fprintf(stderr, "\n\n");
    fprintf(stderr, "%s: mem per token = %8zu bytes\n", __func__,
            mem_per_token);
    fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__,
            t_load_us / 1000.0f);
    fprintf(stderr, "%s:   sample time = %8.2f ms\n", __func__,
            t_sample_us / 1000.0f);
    fprintf(stderr, "%s:  predict time = %8.2f ms / %.2f ms per token\n",
            __func__, t_predict_us / 1000.0f, t_predict_us / 1000.0f / n_past);
    fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__,
            (t_main_end_us - t_main_start_us) / 1000.0f);
  }

  dragon_free(model.ctx);

  if (params.use_color) {
    printf(ANSI_COLOR_RESET);
  }

  return 0;
}
