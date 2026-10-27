#include "gr/io/digest.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>

namespace gr {
namespace {

constexpr std::array<uint32_t, 64> kRound = {
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

uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

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

Sha256::Sha256()
    : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::Block(const uint8_t* block) {
  std::array<uint32_t, 64> w;
  for (int k = 0; k < 16; ++k) {
    w[k] = static_cast<uint32_t>(block[4 * k]) << 24 |
           static_cast<uint32_t>(block[4 * k + 1]) << 16 |
           static_cast<uint32_t>(block[4 * k + 2]) << 8 | block[4 * k + 3];
  }
  for (int k = 16; k < 64; ++k) {
    const uint32_t s0 =
        Rotr(w[k - 15], 7) ^ Rotr(w[k - 15], 18) ^ (w[k - 15] >> 3);
    const uint32_t s1 =
        Rotr(w[k - 2], 17) ^ Rotr(w[k - 2], 19) ^ (w[k - 2] >> 10);
    w[k] = w[k - 16] + s0 + w[k - 7] + s1;
  }
  uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
  for (int k = 0; k < 64; ++k) {
    const uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
    const uint32_t choice = (e & f) ^ (~e & g);
    const uint32_t t1 = h + s1 + choice + kRound[k] + w[k];
    const uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
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
  const uint64_t bits = length_ * 8;
  const uint8_t one = 0x80;
  Update(&one, 1);
  const uint8_t zero = 0;
  while (buffered_ != 56) Update(&zero, 1);
  uint8_t tail[8];
  for (int k = 0; k < 8; ++k) {
    tail[k] = static_cast<uint8_t>(bits >> (56 - 8 * k));
  }
  Update(tail, 8);
  std::string out;
  char hex[9];
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
  if (!values.empty()) sha_.Update(values.data(), values.size() * 8);
}

void PythonDigest::AddJson(const Value& value) {
  sha_.Update(PythonJsonDumps(value));
}

std::string PythonDigest::Hex16() { return sha_.HexDigest().substr(0, 16); }

}  // namespace gr
