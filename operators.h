#pragma once

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define DRAGON_MAX_DIMS     4
#define DRAGON_MAX_NODES    4096
#define DRAGON_MAX_PARAMS   16
#define DRAGON_MAX_CONTEXTS 64
#define DRAGON_MAX_OPT      4

#ifdef __ARM_NEON
// we use the built-in 16-bit float type
typedef __fp16 dragon_fp16_t;
#else
typedef uint16_t dragon_fp16_t;
#endif

// convert FP16 <-> FP32
float       dragon_fp16_to_fp32(dragon_fp16_t x);
dragon_fp16_t dragon_fp32_to_fp16(float x);

struct dragon_object;
struct dragon_context;

enum data_type {
    DATA_TYPE_Q4_0,
    DATA_TYPE_Q4_1,
    DATA_TYPE_I8,
    DATA_TYPE_I16,
    DATA_TYPE_I32,
    DATA_TYPE_F16,
    DATA_TYPE_F32,
    DATA_TYPE_COUNT,
};

// available tensor operations:
enum dragon_op {
    DRAGON_OP_NONE = 0,

    DRAGON_OP_DUP,
    DRAGON_OP_ADD,
    DRAGON_OP_SUB,
    DRAGON_OP_MUL,
    DRAGON_OP_DIV,
    DRAGON_OP_SQR,
    DRAGON_OP_SQRT,
    DRAGON_OP_SUM,
    DRAGON_OP_MEAN,
    DRAGON_OP_REPEAT,
    DRAGON_OP_ABS,
    DRAGON_OP_SGN,
    DRAGON_OP_NEG,
    DRAGON_OP_STEP,
    DRAGON_OP_RELU,
    DRAGON_OP_GELU,
    DRAGON_OP_SILU,
    DRAGON_OP_NORM, // normalize
    DRAGON_OP_RMS_NORM,

    DRAGON_OP_MUL_MAT,

    DRAGON_OP_SCALE,
    DRAGON_OP_CPY,
    DRAGON_OP_RESHAPE,
    DRAGON_OP_VIEW,
    DRAGON_OP_PERMUTE,
    DRAGON_OP_TRANSPOSE,
    DRAGON_OP_GET_ROWS,
    DRAGON_OP_DIAG_MASK_INF,
    DRAGON_OP_SOFT_MAX,
    DRAGON_OP_ROPE,
    DRAGON_OP_CONV_1D_1S,
    DRAGON_OP_CONV_1D_2S,

    DRAGON_OP_FLASH_ATTN,
    DRAGON_OP_FLASH_FF,

    DRAGON_OP_COUNT,
};

// n-dimensional tensor
struct dragon_tensor {
    enum data_type type;

    int    n_dims;
    int    ne[DRAGON_MAX_DIMS]; // number of elements
    size_t nb[DRAGON_MAX_DIMS]; // stride in bytes:
                              // nb[0] = sizeof(type)
                              // nb[1] = nb[0]   * ne[0] + padding
                              // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    enum dragon_op op;

    bool is_param;

    struct dragon_tensor * grad;
    struct dragon_tensor * src0;
    struct dragon_tensor * src1;
    struct dragon_tensor * opt[DRAGON_MAX_OPT];

    // thread scheduling
    int n_tasks;

    // performance
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;

    void * data;
    char padding[8];
};

// computation graph
struct dragon_cgraph {
    int n_nodes;
    int n_leafs;
    int n_threads;

    size_t work_size;
    struct dragon_tensor * work;

    struct dragon_tensor * nodes[DRAGON_MAX_NODES];
    struct dragon_tensor * grads[DRAGON_MAX_NODES];
    struct dragon_tensor * leafs[DRAGON_MAX_NODES];

    // performance
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;
};

// scratch buffer
struct dragon_scratch {
    size_t offs;
    size_t size;
    void * data;
};

struct dragon_init_params {
    // memory pool
    size_t mem_size;   // bytes
    void * mem_buffer; // if NULL, memory will be allocated internally
};

void    dragon_time_init(void); // call this once at the beginning of the program
int64_t dragon_time_ms(void);
int64_t dragon_time_us(void);
int64_t dragon_cycles(void);
int64_t dragon_cycles_per_ms(void);

void dragon_print_object (const struct dragon_object * obj);
void dragon_print_objects(const struct dragon_context * ctx);

int    dragon_nelements(const struct dragon_tensor * tensor);
size_t dragon_nbytes   (const struct dragon_tensor * tensor);

int    dragon_blck_size (enum data_type type);
size_t dragon_type_size (enum data_type type); // size in bytes for all elements in a block
float  dragon_type_sizef(enum data_type type); // dragon_type_size()/dragon_blck_size() as float

size_t dragon_element_size(const struct dragon_tensor * tensor);

struct dragon_context * dragon_init(struct dragon_init_params params);
void dragon_free(struct dragon_context * ctx);

size_t dragon_used_mem(const struct dragon_context * ctx);

size_t dragon_set_scratch(struct dragon_context * ctx, struct dragon_scratch scratch);

struct dragon_tensor * dragon_new_tensor(
        struct dragon_context * ctx,
        enum   data_type type,
        int    n_dims,
        const int *ne);

struct dragon_tensor * dragon_new_tensor_1d(
        struct dragon_context * ctx,
        enum   data_type type,
        int    ne0);

struct dragon_tensor * dragon_new_tensor_2d(
        struct dragon_context * ctx,
        enum   data_type type,
        int    ne0,
        int    ne1);

struct dragon_tensor * dragon_new_tensor_3d(
        struct dragon_context * ctx,
        enum   data_type type,
        int    ne0,
        int    ne1,
        int    ne2);

struct dragon_tensor * dragon_new_tensor_4d(
        struct dragon_context * ctx,
        enum   data_type type,
        int    ne0,
        int    ne1,
        int    ne2,
        int    ne3);

struct dragon_tensor * dragon_new_i32(struct dragon_context * ctx, int32_t value);
struct dragon_tensor * dragon_new_f32(struct dragon_context * ctx, float value);

struct dragon_tensor * dragon_dup_tensor (struct dragon_context * ctx, const struct dragon_tensor * src);
struct dragon_tensor * dragon_view_tensor(struct dragon_context * ctx, const struct dragon_tensor * src);

struct dragon_tensor * dragon_set_zero(struct dragon_tensor * tensor);
struct dragon_tensor * dragon_set_i32 (struct dragon_tensor * tensor, int32_t value);
struct dragon_tensor * dragon_set_f32 (struct dragon_tensor * tensor, float value);

int32_t dragon_get_i32_1d(const struct dragon_tensor * tensor, int i);
void    dragon_set_i32_1d(const struct dragon_tensor * tensor, int i, int32_t value);

float dragon_get_f32_1d(const struct dragon_tensor * tensor, int i);
void  dragon_set_f32_1d(const struct dragon_tensor * tensor, int i, float value);

 void * dragon_get_data    (const struct dragon_tensor * tensor);
float * dragon_get_data_f32(const struct dragon_tensor * tensor);

//
// operations on tensors with backpropagation
//

struct dragon_tensor * dragon_dup(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

struct dragon_tensor * dragon_add(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

struct dragon_tensor * dragon_sub(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

struct dragon_tensor * dragon_mul(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

struct dragon_tensor * dragon_div(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

struct dragon_tensor * dragon_sqr(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

struct dragon_tensor * dragon_sqrt(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

// return scalar
// TODO: compute sum along rows
struct dragon_tensor * dragon_sum(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

// mean along rows
struct dragon_tensor * dragon_mean(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
struct dragon_tensor * dragon_repeat(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

struct dragon_tensor * dragon_abs(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

struct dragon_tensor * dragon_sgn(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

struct dragon_tensor * dragon_neg(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

struct dragon_tensor * dragon_step(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

struct dragon_tensor * dragon_relu(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

// TODO: double-check this computation is correct
struct dragon_tensor * dragon_gelu(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

struct dragon_tensor * dragon_silu(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

// normalize along rows
// TODO: eps is hardcoded to 1e-5 for now
struct dragon_tensor * dragon_norm(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

struct dragon_tensor * dragon_rms_norm(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

// A: m rows, n columns
// B: p rows, n columns (i.e. we transpose it internally)
// result is m columns, p rows
struct dragon_tensor * dragon_mul_mat(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

//
// operations on tensors without backpropagation
//

// in-place, returns view(a)
struct dragon_tensor * dragon_scale(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

// a -> b, return view(b)
struct dragon_tensor * dragon_cpy(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

// return view(a), b specifies the new shape
// TODO: when we start computing gradient, make a copy instead of view
struct dragon_tensor * dragon_reshape(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
struct dragon_tensor * dragon_reshape_2d(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   ne0,
        int                   ne1);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
struct dragon_tensor * dragon_reshape_3d(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2);

// offset in bytes
struct dragon_tensor * dragon_view_1d(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   ne0,
        size_t                offset);

struct dragon_tensor * dragon_view_2d(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   ne0,
        int                   ne1,
        size_t                nb1, // row stride in bytes
        size_t                offset);

struct dragon_tensor * dragon_permute(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3);

// alias for dragon_permute(ctx, a, 1, 0, 2, 3)
struct dragon_tensor * dragon_transpose(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

struct dragon_tensor * dragon_get_rows(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

// set elements above the diagonal to -INF
// in-place, returns view(a)
struct dragon_tensor * dragon_diag_mask_inf(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   n_past);

// in-place, returns view(a)
struct dragon_tensor * dragon_soft_max(
        struct dragon_context * ctx,
        struct dragon_tensor  * a);

// rotary position embedding
// in-place, returns view(a)
// if mode == 1, skip n_past elements
// TODO: avoid creating a new tensor every time
struct dragon_tensor * dragon_rope(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        int                   n_past,
        int                   n_dims,
        int                   mode);

// padding = 1
// TODO: we don't support extra parameters for now
//       that's why we are hard-coding the stride, padding, and dilation
//       not great ..
struct dragon_tensor * dragon_conv_1d_1s(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

struct dragon_tensor * dragon_conv_1d_2s(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b);

struct dragon_tensor * dragon_flash_attn(
        struct dragon_context * ctx,
        struct dragon_tensor  * q,
        struct dragon_tensor  * k,
        struct dragon_tensor  * v,
        bool                  masked);

struct dragon_tensor * dragon_flash_ff(
        struct dragon_context * ctx,
        struct dragon_tensor  * a,
        struct dragon_tensor  * b0,
        struct dragon_tensor  * b1,
        struct dragon_tensor  * c0,
        struct dragon_tensor  * c1);

//
// automatic differentiation
//

void dragon_set_param(
        struct dragon_context * ctx,
        struct dragon_tensor * tensor);

void dragon_build_forward_expand(struct dragon_cgraph * cgraph, struct dragon_tensor * tensor);

struct dragon_cgraph dragon_build_forward (struct dragon_tensor * tensor);
struct dragon_cgraph dragon_build_backward(struct dragon_context * ctx, struct dragon_cgraph * gf, bool keep);

void dragon_graph_compute(struct dragon_context * ctx, struct dragon_cgraph * cgraph);
void dragon_graph_reset  (struct dragon_cgraph * cgraph);

// print info and performance information for the graph
void dragon_graph_print(const struct dragon_cgraph * cgraph);

// dump the graph into a file using the dot format
void dragon_graph_dump_dot(const struct dragon_cgraph * gb, const struct dragon_cgraph * gf, const char * filename);

//
// optimization
//

// optimization methods
enum dragon_opt_type {
    DRAGON_OPT_ADAM,
    DRAGON_OPT_LBFGS,
};

// linesearch methods
enum dragon_linesearch {
    DRAGON_LINESEARCH_DEFAULT = 1,

    DRAGON_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
    DRAGON_LINESEARCH_BACKTRACKING_WOLFE        = 1,
    DRAGON_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
};

// optimization return values
enum dragon_opt_result {
    DRAGON_OPT_OK = 0,
    DRAGON_OPT_DID_NOT_CONVERGE,
    DRAGON_OPT_NO_CONTEXT,
    DRAGON_OPT_INVALID_WOLFE,
    DRAGON_OPT_FAIL,

    DRAGON_LINESEARCH_FAIL = -128,
    DRAGON_LINESEARCH_MINIMUM_STEP,
    DRAGON_LINESEARCH_MAXIMUM_STEP,
    DRAGON_LINESEARCH_MAXIMUM_ITERATIONS,
    DRAGON_LINESEARCH_INVALID_PARAMETERS,
};

// optimization parameters
//
//   see operators.c (dragon_opt_default_params) for default values
//
struct dragon_opt_params {
    enum dragon_opt_type type;

    int n_threads;

    // delta-based convergence test
    //
    //   if past == 0 - disabled
    //   if past > 0:
    //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
    //
    int past;
    float delta;

    // maximum number of iterations without improvement
    //
    //   if 0 - disabled
    //   if > 0:
    //     assume convergence if no cost improvement in this number of iterations
    //
    int max_no_improvement;

    bool print_forward_graph;
    bool print_backward_graph;

    // ADAM parameters
    struct {
        int n_iter;

        float alpha; // learning rate
        float beta1;
        float beta2;
        float eps;   // epsilon for numerical stability
        float eps_f; // epsilon for convergence test
        float eps_g; // epsilon for convergence test
    } adam;

    // LBFGS parameters
    struct {
        int m; // number of corrections to approximate the inv. Hessian
        int n_iter;
        int max_linesearch;

        float eps;      // convergence tolerance
        float ftol;     // line search tolerance
        float wolfe;
        float min_step;
        float max_step;

        enum dragon_linesearch linesearch;
    } lbfgs;
};

struct dragon_opt_params dragon_opt_default_params(enum dragon_opt_type type);

// optimize the function defined by the tensor f
enum dragon_opt_result dragon_opt(
        struct dragon_context * ctx,
        struct dragon_opt_params params,
        struct dragon_tensor * f);

//
// system info
//

int dragon_cpu_has_avx(void);
int dragon_cpu_has_avx2(void);
int dragon_cpu_has_avx512(void);
int dragon_cpu_has_fma(void);
int dragon_cpu_has_neon(void);
int dragon_cpu_has_arm_fma(void);
int dragon_cpu_has_f16c(void);
int dragon_cpu_has_fp16_va(void);
int dragon_cpu_has_wasm_simd(void);
int dragon_cpu_has_blas(void);
int dragon_cpu_has_sse3(void);
int dragon_cpu_has_vsx(void);

#ifdef  __cplusplus
}
#endif
