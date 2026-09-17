#ifndef CUSOLVER_EMULATION_HPP
#define CUSOLVER_EMULATION_HPP

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "acclib.hpp"

// Match the cuBLAS enable switch. -1 means native FP64; 0 selects the
// library's default fixed mantissa count. Read afresh for each energy.
inline int cusolverEmulationMantissaBits() {
   const char *enabled = std::getenv("CUBLAS_EMULATE_DOUBLE_PRECISION");
   if (!enabled || std::strcmp(enabled, "1") != 0) return -1;

   const char *value = std::getenv("CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT");
   if (!value) return 0;

   char *end = nullptr;
   errno = 0;
   const long bits = std::strtol(value, &end, 10);
   if (errno == ERANGE || end == value || *end != '\0' || bits < 0 || bits > INT_MAX) {
      std::fprintf(stderr, "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT "
                           "must be a nonnegative integer, got '%s'\n", value);
      std::exit(EXIT_FAILURE);
   }
   return static_cast<int>(bits);
}

// Strategy, shared by the cuSOLVER and cuBLAS configurators. Both
// cudaEmulationStrategy and cublasEmulationStrategy_t use the same values
// (DEFAULT 0, PERFORMANT 1, EAGER 2), so one integer serves for both.
//
// DEFAULT is documented as "equivalent to PERFORMANT unless a library dependent
// environment variable is set", so passing it unchanged lets the library read
// CUBLAS_EMULATION_STRATEGY itself. We still parse the variable and pass the
// value explicitly, because an explicit call must not silently contradict what
// EmulationEnvVar.inc advertises for the 'Cublas-1' scheme.
//
//   unset        -> DEFAULT   (library decides)
//   "default"    -> DEFAULT
//   "performant" -> PERFORMANT
//   "eager"      -> EAGER
inline int emulationStrategyFromEnv() {
   const char *value = std::getenv("CUBLAS_EMULATION_STRATEGY");
   if (!value || *value == '\0') return 0;             // DEFAULT

   char lowered[32];
   std::size_t i = 0;
   for (; value[i] != '\0' && i < sizeof(lowered) - 1; ++i) {
      lowered[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(value[i])));
   }
   lowered[i] = '\0';

   if (std::strcmp(lowered, "default")    == 0) return 0;
   if (std::strcmp(lowered, "performant") == 0) return 1;
   if (std::strcmp(lowered, "eager")      == 0) return 2;

   std::fprintf(stderr, "CUBLAS_EMULATION_STRATEGY must be one of "
                        "default, performant, eager; got '%s'\n", value);
   std::exit(EXIT_FAILURE);
}

// cuSOLVER's internal cuBLAS handles ignore cuBLAS environment variables.
// Apply these settings before querying the LU workspace size.
inline void configureCusolverEmulation(cusolverDnHandle_t handle, int mantissa_bits) {
#if CUDART_VERSION >= 13020
   if (mantissa_bits < 0) {
      checkCusolverErrors(cusolverDnSetMathMode(handle, CUSOLVER_DEFAULT_MATH));
      return;
   }
   checkCusolverErrors(cusolverDnSetEmulationStrategy(
                         handle, static_cast<cudaEmulationStrategy_t>(emulationStrategyFromEnv())));
   checkCusolverErrors(cusolverDnSetFixedPointEmulationMantissaControl(
                         handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
   checkCusolverErrors(cusolverDnSetFixedPointEmulationMaxMantissaBitCount(handle, mantissa_bits));
   checkCusolverErrors(cusolverDnSetMathMode(handle, CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH));
#else
   (void)handle;
   if (mantissa_bits >= 0) {
      std::fprintf(stderr, "cuSOLVER FP64 emulation requires CUDA Toolkit 13.2 or newer\n");
      std::exit(EXIT_FAILURE);
   }
#endif
}

#endif
