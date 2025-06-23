#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "operators.h"
#include "attention.h"
#include "utils.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <signal.h>
#endif


// load the model's weights from a file
bool llama_vocab_load(const std::string &fname,
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
    if (false) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
              fname.c_str());
      return false;
    }
  }

  int n_ff = 0;
  int n_parts = 0;
  // Load hparams using helper
  llama_hparams hparams;
  if (!load_hparams(fin, hparams, user_n_ctx, n_ff, n_parts)) {
    fin.close();
    return false;
  }

  // Load vocab using helper
  if (!load_vocab(fin, vocab, hparams)) {
    fin.close();
    return false;
  }

  // Close the initial file stream as metadata is read
  fin.close();

  return true;
}

int main(int argc, char **argv) {
  gpt_params params;

  // Read two arguments: model_path and prompt
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <model_path> <prompt>\n", argv[0]);
    return 1;
  }
  params.model = argv[1];
  params.prompt = argv[2];

  gpt_vocab vocab;

  // load the vocab
  {
    const int64_t t_start_us = dragon_time_us();
    if (!llama_vocab_load(params.model, vocab, params.n_ctx)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__,
              params.model.c_str());
      return 1;
    }
  }

  // Add a space in front of the first character to match OG llama tokenizer
  // behavior
  params.prompt.insert(0, 1, ' ');
  // tokenize the prompt
  std::vector<gpt_vocab::id> embd_inp =
      ::llama_tokenize(vocab, params.prompt, true);

  fprintf(stderr, "%s", params.prompt.c_str());
  // 打印ids，用空格分隔
  for (auto& id : embd_inp) {
    fprintf(stdout, "%d ", id);
  }

  return 0;
}
