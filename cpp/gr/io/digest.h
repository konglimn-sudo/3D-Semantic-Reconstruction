// Content hashes as the Python side computes them (pose_set._digest):
// SHA-256 over float64 array bytes and json.dumps(part, sort_keys=True)
// text, so that a pose set written by a C++ program has the ids the Python
// stages derive from it.

#ifndef GARAGE_RECON_GR_IO_DIGEST_H_
#define GARAGE_RECON_GR_IO_DIGEST_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gr/io/value.h"

namespace gr {

// SHA-256 (FIPS 180-4).
class Sha256 {
 public:
  // Bytes per message block, and 32-bit words of the hash state.
  static constexpr size_t kBlockBytes = 64;
  static constexpr size_t kStateWords = 8;

  Sha256();
  void Update(const void* data, size_t size);
  void Update(const std::string& text) { Update(text.data(), text.size()); }
  // Lowercase hex of the digest; the object is spent afterwards.
  std::string HexDigest();

 private:
  void Block(const uint8_t* block);

  std::array<uint32_t, kStateWords> state_;
  std::array<uint8_t, kBlockBytes> buffer_{};
  size_t buffered_ = 0;
  uint64_t length_ = 0;  // Bytes.
};

// json.dumps(value, sort_keys=True[, indent=indent]): by default one line
// with ", " and ": " separators; map keys sorted at every depth (ASCII
// text: non-ASCII stays UTF-8 where Python escapes it).
std::string PythonJsonDumps(const Value& value, int indent = kJsonOneLine);

// pose_set._digest: the parts in order, each either float64 values (an
// array's C-order bytes) or a JSON value; the first 16 hex digits.
class PythonDigest {
 public:
  // Hex digits of the short digest.
  static constexpr size_t kShortDigestLength = 16;

  void AddDoubles(const std::vector<double>& values);
  void AddJson(const Value& value);
  std::string Hex16();

 private:
  Sha256 sha_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_IO_DIGEST_H_
