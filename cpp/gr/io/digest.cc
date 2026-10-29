#include "gr/io/digest.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <utility>

namespace gr {
namespace {

// SHA-256 rounds per block, and the message words of a block.
constexpr int kRounds = 64;
constexpr int kBlockWords = 16;
constexpr int kWordBits = 32;
constexpr int kWordBytes = 4;

// The round constants K (FIPS 180-4 section 4.2.2): the first 32 bits of the
// fractional parts of the cube roots of the first 64 primes.
constexpr std::array<uint32_t, kRounds> kRoundConstants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// The initial hash value H(0) (section 5.3.3): the first 32 bits of the
// fractional parts of the square roots of the first 8 primes.
constexpr std::array<uint32_t, Sha256::kStateWords> kInitialHash = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

// Padding (section 5.1.1): a single set bit, zeros, then the message length
// in bits as a 64-bit big-endian number at the end of the last block.
constexpr uint8_t kPaddingStart = 0x80;
constexpr size_t kLengthBytes = sizeof(uint64_t);
constexpr size_t kLengthOffset = Sha256::kBlockBytes - kLengthBytes;

// Hex digits of a 32-bit word, plus the terminating null character.
constexpr size_t kWordHexBuffer = 2 * kWordBytes + 1;

uint32_t RotateRight(uint32_t x, int bits) {
  return (x >> bits) | (x << (kWordBits - bits));
}

// The functions of section 4.1.2.
uint32_t BigSigma0(uint32_t x) {
  return RotateRight(x, 2) ^ RotateRight(x, 13) ^ RotateRight(x, 22);
}
uint32_t BigSigma1(uint32_t x) {
  return RotateRight(x, 6) ^ RotateRight(x, 11) ^ RotateRight(x, 25);
}
uint32_t SmallSigma0(uint32_t x) {
  return RotateRight(x, 7) ^ RotateRight(x, 18) ^ (x >> 3);
}
uint32_t SmallSigma1(uint32_t x) {
  return RotateRight(x, 17) ^ RotateRight(x, 19) ^ (x >> 10);
}

// The big-endian 32-bit word at `bytes`.
uint32_t LoadBigEndian32(const uint8_t* bytes) {
  uint32_t word = 0;
  for (int k = 0; k < kWordBytes; ++k) word = word << CHAR_BIT | bytes[k];
  return word;
}

// A copy with the map members sorted by key at every depth.
Value Sorted(const Value& value) {
  if (value.is_list()) {
    Value out = Value::List();
    for (const Value& item : value.items()) out.Append(Sorted(item));
    return out;
  }
  if (!value.is_map()) return value;
  std::vector<std::pair<std::string, Value>> members = value.members();
  std::sort(members.begin(), members.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  Value out = Value::Map();
  for (const auto& [key, item] : members) out.Set(key, Sorted(item));
  return out;
}

}  // namespace

Sha256::Sha256() : state_(kInitialHash) {}

void Sha256::Block(const uint8_t* block) {
  // The message schedule W (section 6.2.2): the block's words, then
  // W[t] = s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16].
  std::array<uint32_t, kRounds> schedule;
  for (int k = 0; k < kBlockWords; ++k) {
    schedule[k] = LoadBigEndian32(block + kWordBytes * k);
  }
  for (int k = kBlockWords; k < kRounds; ++k) {
    schedule[k] = schedule[k - 16] + SmallSigma0(schedule[k - 15]) +
                  schedule[k - 7] + SmallSigma1(schedule[k - 2]);
  }
  // The working variables a..h of the standard.
  auto [a, b, c, d, e, f, g, h] = state_;
  for (int k = 0; k < kRounds; ++k) {
    const uint32_t choice = (e & f) ^ (~e & g);
    const uint32_t t1 =
        h + BigSigma1(e) + choice + kRoundConstants[k] + schedule[k];
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = BigSigma0(a) + majority;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  const std::array<uint32_t, kStateWords> working = {a, b, c, d, e, f, g, h};
  for (size_t k = 0; k < kStateWords; ++k) state_[k] += working[k];
}

void Sha256::Update(const void* data, size_t size) {
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  length_ += size;
  while (size > 0) {
    const size_t take = std::min(size, buffer_.size() - buffered_);
    std::memcpy(buffer_.data() + buffered_, bytes, take);
    buffered_ += take;
    bytes += take;
    size -= take;
    if (buffered_ == buffer_.size()) {
      Block(buffer_.data());
      buffered_ = 0;
    }
  }
}

std::string Sha256::HexDigest() {
  const uint64_t bits = length_ * CHAR_BIT;
  Update(&kPaddingStart, 1);
  const uint8_t zero = 0;
  while (buffered_ != kLengthOffset) Update(&zero, 1);
  std::array<uint8_t, kLengthBytes> length_field;
  for (size_t k = 0; k < kLengthBytes; ++k) {
    length_field[k] =
        static_cast<uint8_t>(bits >> (CHAR_BIT * (kLengthBytes - 1 - k)));
  }
  Update(length_field.data(), length_field.size());
  std::string out;
  char hex[kWordHexBuffer];
  for (uint32_t word : state_) {
    std::snprintf(hex, sizeof(hex), "%08x", word);
    out += hex;
  }
  return out;
}

std::string PythonJsonDumps(const Value& value, int indent) {
  return ToJson(Sorted(value), indent);
}

void PythonDigest::AddDoubles(const std::vector<double>& values) {
  if (!values.empty()) {
    sha_.Update(values.data(), values.size() * sizeof(double));
  }
}

void PythonDigest::AddJson(const Value& value) {
  sha_.Update(PythonJsonDumps(value));
}

std::string PythonDigest::Hex16() {
  return sha_.HexDigest().substr(0, kShortDigestLength);
}

}  // namespace gr
