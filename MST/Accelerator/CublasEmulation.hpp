#ifndef CUBLAS_EMULATION_HPP
#define CUBLAS_EMULATION_HPP

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "acclib.hpp"
#include "CusolverEmulation.hpp"   // cusolverEmulationMantissaBits()

// Companion to CusolverEmulation.hpp for the cuBLAS side.
//
// Both read the same two environment variables, which EmulationModule.F90 sets
// from the 'Cublas-1' scheme in EmulationEnvVar.inc:
//
//     CUBLAS_EMULATE_DOUBLE_PRECISION = 1        enable
//     CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT = N
//     CUBLAS_EMULATION_STRATEGY = default|performant|eager
//
// The strategy is parsed by emulationStrategyFromEnv() in CusolverEmulation.hpp
// and passed explicitly, so the same variable steers both libraries and neither
// silently contradicts what EmulationEnvVar.inc sets.
//
// Unlike cuSOLVER, cuBLAS has supported FP64 fixed-point emulation since CUDA
// 13.0, so this path is live on the current toolkit while the cuSOLVER one
// waits for 13.2+.
//
// Mantissa control, per CUDA's library_types.h:
//   DYNAMIC - the library picks enough retained bits to be at least as accurate
//             as the format being emulated (i.e. >= FP64). Safe default.
//   FIXED   - the caller pins the count, which is what allows trading accuracy
//             for speed per energy-contour zone.
//
// A bit count of 0 from the environment means "library default"; we map that to
// DYNAMIC. Any positive count selects FIXED.

#if CUDART_VERSION >= 13000

// One mapped-pinned int that cuBLAS reads the live bit count through. Mapped
// rather than plain host or plain device memory because the documentation does
// not say which side dereferences it; mapped pinned is addressable from both.
inline int *cublasEmulationMantissaSlot() {
   static int *slot = nullptr;
   if (slot == nullptr) {
      checkCudaErrors(cudaHostAlloc((void **)&slot, sizeof(int), cudaHostAllocMapped));
      *slot = 0;
   }
   return slot;
}

// Change the retained mantissa bits WITHOUT reconfiguring the handle. Valid
// only once configureCublasEmulation has installed the pointer with a positive
// count. This is the cheap path for per-energy-point precision changes: one
// store, no cuBLAS call, no handle state change.
inline void setCublasEmulationMantissaBits(int bits) {
   *cublasEmulationMantissaSlot() = bits;
}

inline void configureCublasEmulation(cublasHandle_t handle, int mantissa_bits) {
   if (mantissa_bits < 0) {                       // emulation off -> native FP64
      checkCublasErrors(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
      return;
   }
   checkCublasErrors(cublasSetEmulationStrategy(
                        handle, static_cast<cublasEmulationStrategy_t>(emulationStrategyFromEnv())));
   if (mantissa_bits == 0) {
      checkCublasErrors(cublasSetFixedPointEmulationMantissaControl(
                           handle, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
   } else {
      int *slot = cublasEmulationMantissaSlot();
      *slot = mantissa_bits;
      checkCublasErrors(cublasSetFixedPointEmulationMantissaControl(
                           handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
      checkCublasErrors(cublasSetFixedPointEmulationMaxMantissaBitCount(handle, mantissa_bits));
      checkCublasErrors(cublasSetFixedPointEmulationMantissaBitCountPointer(handle, slot));
   }
   checkCublasErrors(cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
}

#else

inline void setCublasEmulationMantissaBits(int) {}
inline void configureCublasEmulation(cublasHandle_t handle, int mantissa_bits) {
   (void)handle;
   if (mantissa_bits >= 0) {
      std::fprintf(stderr, "cuBLAS FP64 emulation requires CUDA Toolkit 13.0 or newer\n");
      std::exit(EXIT_FAILURE);
   }
}

#endif
#endif
