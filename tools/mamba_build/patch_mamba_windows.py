"""Universal SR Studio — Windows/MSVC portability patch for mamba_ssm v2.2.2.

Run once on a fresh `git clone --depth 1 --branch v2.2.2 https://github.com/state-spaces/mamba.git mamba_src`:
    python patch_mamba_windows.py <path to mamba_src>

1. setup.py: _USE_MATH_DEFINES (MSVC has no M_LOG2E without it) and our own gencode list
   (upstream hard-codes sm_53/62/70/72/80/87/90 and ignores TORCH_CUDA_ARCH_LIST: no sm_61 =
   "no kernel image is available" on desktop Pascal). PTX for compute_89 is embedded too, so
   newer GPUs (Blackwell sm_120) can JIT-compile it until a native build exists.
2. selective_scan_{fwd,bwd}_kernel.cuh: the `#ifndef USE_ROCM / #else / #endif` block sits
   inside the BOOL_SWITCH lambdas, i.e. inside macro arguments, which the MSVC preprocessor
   rejects ("#" not expected here). Only the CUDA branch is kept. No /Zc:preprocessor
   (it broke the constexpr template arguments: C2975), so every BOOL_SWITCH variant is
   still compiled — the kernel stays fully general (z, non-variable B/C...).
3. mamba_ssm/__init__.py: Mamba / Mamba2 / MambaLMHeadModel need triton and transformers,
   which SR never uses — imported only when available; selective_scan_fn stays mandatory.
"""
import os
import re
import sys

root = sys.argv[1]


def patch(rel, pairs):
    p = os.path.join(root, rel)
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        if callable(old):
            s2 = old(s)
            assert s2 != s, (rel, "regex patch did not apply")
            s = s2
            continue
        assert s.count(old) == 1, (rel, old[:60], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
    print("patched", rel)


GENCODE_OLD = re.compile(
    r'        cc_flag\.append\("-gencode"\)\n        cc_flag\.append\("arch=compute_53,code=sm_53"\)\n'
    r'.*?            cc_flag\.append\("arch=compute_90,code=sm_90"\)\n', re.S)
GENCODE_NEW = '''        # Universal SR Studio portability patch: real targets (upstream skips sm_61).
        # 61 = GTX 10 desktop, 75 = RTX 20, 86 = RTX 30, 89 = RTX 40, 90 = Hopper;
        # + compute_89 PTX so newer GPUs (RTX 50, sm_120) can JIT it.
        for _arch in ("61", "75", "86", "89"):
            cc_flag.append("-gencode")
            cc_flag.append(f"arch=compute_{_arch},code=sm_{_arch}")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_89,code=compute_89")
'''

patch("setup.py", [
    (lambda s: GENCODE_OLD.sub(lambda m: GENCODE_NEW, s, count=1), None),
    ('''        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",''',
     '''        _win = ["-D_USE_MATH_DEFINES"] if os.name == "nt" else []  # USS: M_LOG2E on MSVC
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"] + _win,
            "nvcc": append_nvcc_threads(
                _win + [
                    "-O3",
                    "-std=c++17",'''),
])

ROCM_IN_LAMBDA = re.compile(
    r'\n(?P<ind>[ \t]*)#ifndef USE_ROCM\n(?P<cuda>[ \t]*C10_CUDA_CHECK\(cudaFuncSetAttribute\(\n'
    r'[^\n]*kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize\)\);\n)'
    r'[ \t]*#else\n.*?[ \t]*#endif\n', re.S)
for f in ("selective_scan_fwd_kernel.cuh", "selective_scan_bwd_kernel.cuh"):
    patch(os.path.join("csrc", "selective_scan", f), [
        (lambda s: ROCM_IN_LAMBDA.sub(lambda m: "\n" + m.group("cuda"), s, count=1), None)])

# 4. MSVC (as nvcc host compiler) rejects the constexpr template arguments built inside the
#    nested BOOL_SWITCH lambdas (C2975 'kNRows_' / 'kIsEvenLen_'), with or without
#    /Zc:preprocessor. The launchers are rewritten without lambdas: a template helper per
#    variant + plain if/else dispatch. Variable B/C (what every SR Mamba arch uses) is
#    required and checked at run time; even/odd length, z and delta_softplus stay dispatched.
FWD_LAUNCH = re.compile(
    r"template<int kNThreads, int kNItems, typename input_t, typename weight_t>\n"
    r"void selective_scan_fwd_launch\(SSMParamsBase &params, cudaStream_t stream\) \{\n.*?\n\}\n", re.S)
FWD_NEW = r'''template<int kNThreads, int kNItems, bool kIsEvenLen, bool kHasZ, typename input_t, typename weight_t>
void selective_scan_fwd_launch_variant(SSMParamsBase &params, cudaStream_t stream) {
    // USS: lambda-free launcher (MSVC C2975 on the nested BOOL_SWITCH lambdas).
    using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, 1, kIsEvenLen, true, true, kHasZ, input_t, weight_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize + 1 * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
    dim3 grid(params.batch, params.dim / 1);
    auto kernel = &selective_scan_fwd_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    TORCH_CHECK(params.is_variable_B && params.is_variable_C,
                "selective_scan (USS Windows build): only variable B and C are compiled");
    const bool even = params.seqlen % (kNThreads * kNItems) == 0;
    const bool has_z = params.z_ptr != nullptr;
    if (even) {
        if (has_z) { selective_scan_fwd_launch_variant<kNThreads, kNItems, true, true, input_t, weight_t>(params, stream); }
        else { selective_scan_fwd_launch_variant<kNThreads, kNItems, true, false, input_t, weight_t>(params, stream); }
    } else {
        if (has_z) { selective_scan_fwd_launch_variant<kNThreads, kNItems, false, true, input_t, weight_t>(params, stream); }
        else { selective_scan_fwd_launch_variant<kNThreads, kNItems, false, false, input_t, weight_t>(params, stream); }
    }
}
'''
BWD_LAUNCH = re.compile(
    r"template<int kNThreads, int kNItems, typename input_t, typename weight_t>\n"
    r"void selective_scan_bwd_launch\(SSMParamsBwd &params, cudaStream_t stream\) \{\n.*?\n\}\n", re.S)
_bwd_calls = []
for even in ("true", "false"):
    for sp in ("true", "false"):
        for z in ("true", "false"):
            _bwd_calls.append(
                f"    if (even == {even} && softplus == {sp} && has_z == {z}) {{ "
                f"selective_scan_bwd_launch_variant<kNThreads, kNItems, {even}, {sp}, {z}, input_t, weight_t>(params, stream); return; }}")
BWD_NEW = r'''template<int kNThreads, int kNItems, bool kIsEvenLen, bool kDeltaSoftplus, bool kHasZ, typename input_t, typename weight_t>
void selective_scan_bwd_launch_variant(SSMParamsBwd &params, cudaStream_t stream) {
    // USS: lambda-free launcher (MSVC C2975 on the nested BOOL_SWITCH lambdas).
    using Ktraits = Selective_Scan_bwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, true, true, kDeltaSoftplus, kHasZ, input_t, weight_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize + MAX_DSTATE * sizeof(typename Ktraits::scan_t) + (kNThreads + 4 * MAX_DSTATE) * sizeof(typename Ktraits::weight_t);
    dim3 grid(params.batch, params.dim);
    auto kernel = &selective_scan_bwd_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_bwd_launch(SSMParamsBwd &params, cudaStream_t stream) {
    TORCH_CHECK(params.is_variable_B && params.is_variable_C,
                "selective_scan (USS Windows build): only variable B and C are compiled");
    const bool even = params.seqlen % (kNThreads * kNItems) == 0;
    const bool softplus = params.delta_softplus;
    const bool has_z = params.z_ptr != nullptr;
''' + "\n".join(_bwd_calls) + "\n}\n"

patch(os.path.join("csrc", "selective_scan", "selective_scan_fwd_kernel.cuh"), [
    (lambda s: FWD_LAUNCH.sub(lambda m: FWD_NEW, s, count=1), None)])
patch(os.path.join("csrc", "selective_scan", "selective_scan_bwd_kernel.cuh"), [
    (lambda s: BWD_LAUNCH.sub(lambda m: BWD_NEW, s, count=1), None)])

patch(os.path.join("mamba_ssm", "__init__.py"), [(
    '''from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel''',
    '''# USS: the language-model modules need triton / transformers (unused for SR).
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    Mamba = Mamba2 = MambaLMHeadModel = None''')])
