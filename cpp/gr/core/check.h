// Precondition checks.
//
// The library does not use C++ exceptions. A violated precondition is a
// programming error: GR_CHECK prints the failed condition and aborts, like
// CHECK in glog / Abseil. Code that takes untrusted input (for example the
// Python bindings) validates it before calling into the library.

#ifndef GARAGE_RECON_GR_CORE_CHECK_H_
#define GARAGE_RECON_GR_CORE_CHECK_H_

#include <cstdio>
#include <cstdlib>

namespace gr::internal {

[[noreturn]] inline void CheckFailed(const char* file, int line,
                                     const char* condition) {
  std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, condition);
  std::abort();
}

}  // namespace gr::internal

#define GR_CHECK(condition)                                        \
  do {                                                             \
    if (!(condition)) {                                            \
      ::gr::internal::CheckFailed(__FILE__, __LINE__, #condition); \
    }                                                              \
  } while (false)

#endif  // GARAGE_RECON_GR_CORE_CHECK_H_
