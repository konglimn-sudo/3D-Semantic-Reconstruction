// Progress logging for the stage-level functions.

#ifndef GARAGE_RECON_GR_CORE_LOGGING_H_
#define GARAGE_RECON_GR_CORE_LOGGING_H_

#include <functional>
#include <string>

namespace gr {

// Receives one line per call; an empty logger keeps a stage silent.
using Logger = std::function<void(const std::string&)>;

inline void LogLine(const Logger& log, const std::string& message) {
  if (log) log(message);
}

// printf-style formatting into a std::string. The compiler checks the
// arguments against `format` (parameter 1; the values start at parameter 2).
std::string StringPrintf(const char* format, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_LOGGING_H_
