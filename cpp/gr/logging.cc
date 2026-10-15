#include "gr/logging.h"

#include <cstdarg>
#include <cstdio>

namespace gr {

std::string StringPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  va_list copy;
  va_copy(copy, args);
  const int n = std::vsnprintf(nullptr, 0, format, copy);
  va_end(copy);
  std::string out(n > 0 ? static_cast<size_t>(n) : 0, '\0');
  if (n > 0) std::vsnprintf(out.data(), out.size() + 1, format, args);
  va_end(args);
  return out;
}

}  // namespace gr
