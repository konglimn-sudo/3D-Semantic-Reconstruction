#include "gr/core/logging.h"

#include <cstdarg>
#include <cstdio>

namespace gr {

std::string StringPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  // A first pass measures the text, a second writes it.
  va_list measure_args;
  va_copy(measure_args, args);
  const int length = std::vsnprintf(nullptr, 0, format, measure_args);
  va_end(measure_args);
  std::string out(length > 0 ? static_cast<size_t>(length) : 0, '\0');
  if (length > 0) {
    // vsnprintf also writes the terminating null character.
    std::vsnprintf(out.data(), out.size() + 1, format, args);
  }
  va_end(args);
  return out;
}

}  // namespace gr
