#include "gr/io/npy.h"

#include <zlib.h>

#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace gr {
namespace {

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

std::vector<uint8_t> ReadBytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) Fail("cannot read " + path);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

void WriteBytes(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) Fail("cannot write " + path);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!out) Fail("cannot write " + path);
}

uint64_t Le(const uint8_t* p, int bytes) {
  uint64_t v = 0;
  for (int k = bytes - 1; k >= 0; --k) v = (v << 8) | p[k];
  return v;
}

void PutLe(uint64_t v, int bytes, std::vector<uint8_t>* out) {
  for (int k = 0; k < bytes; ++k) {
    out->push_back(static_cast<uint8_t>(v >> (8 * k)));
  }
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  const uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {  // Subnormal: normalise.
      int e = -1;
      do {
        ++e;
        mantissa <<= 1;
      } while ((mantissa & 0x400) == 0);
      bits = sign | static_cast<uint32_t>(127 - 15 - e) << 23 |
             (mantissa & 0x3FF) << 13;
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000 | mantissa << 13;
  } else {
    bits = sign | (exponent + 127 - 15) << 23 | mantissa << 13;
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

template <typename T>
T Load(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

// Element k as a double.
double Element(const NpyArray& a, int64_t k) {
  const uint8_t* p = a.data.data() + k * a.item_size;
  switch (a.kind) {
    case 'f':
      if (a.item_size == 2) return HalfToFloat(Load<uint16_t>(p));
      if (a.item_size == 4) return Load<float>(p);
      if (a.item_size == 8) return Load<double>(p);
      break;
    case 'i':
      if (a.item_size == 1) return Load<int8_t>(p);
      if (a.item_size == 2) return Load<int16_t>(p);
      if (a.item_size == 4) return Load<int32_t>(p);
      if (a.item_size == 8) return static_cast<double>(Load<int64_t>(p));
      break;
    case 'u':
    case 'b':
      if (a.item_size == 1) return Load<uint8_t>(p);
      if (a.item_size == 2) return Load<uint16_t>(p);
      if (a.item_size == 4) return Load<uint32_t>(p);
      if (a.item_size == 8) return static_cast<double>(Load<uint64_t>(p));
      break;
  }
  Fail(std::string("unsupported dtype ") + a.kind +
       std::to_string(a.item_size));
}

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string out = "(";
  for (size_t k = 0; k < shape.size(); ++k) {
    if (k) out += ", ";
    out += std::to_string(shape[k]);
  }
  if (shape.size() == 1) out += ",";
  return out + ")";
}

std::vector<uint8_t> Inflate(const uint8_t* data, size_t size,
                             size_t expected) {
  std::vector<uint8_t> out(expected);
  z_stream stream{};
  if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) Fail("inflateInit2 failed");
  stream.next_in = const_cast<Bytef*>(data);
  stream.avail_in = static_cast<uInt>(size);
  stream.next_out = out.data();
  stream.avail_out = static_cast<uInt>(out.size());
  const int status = inflate(&stream, Z_FINISH);
  inflateEnd(&stream);
  if (status != Z_STREAM_END || stream.total_out != expected) {
    Fail("corrupt deflate stream");
  }
  return out;
}

std::vector<uint8_t> Deflate(const std::vector<uint8_t>& data) {
  z_stream stream{};
  if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK) {
    Fail("deflateInit2 failed");
  }
  std::vector<uint8_t> out(deflateBound(&stream, data.size()));
  stream.next_in = const_cast<Bytef*>(data.data());
  stream.avail_in = static_cast<uInt>(data.size());
  stream.next_out = out.data();
  stream.avail_out = static_cast<uInt>(out.size());
  const int status = deflate(&stream, Z_FINISH);
  deflateEnd(&stream);
  if (status != Z_STREAM_END) Fail("deflate failed");
  out.resize(stream.total_out);
  return out;
}

}  // namespace

int64_t NpyArray::size() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

int64_t NpyArray::columns() const {
  int64_t n = 1;
  for (size_t k = 1; k < shape.size(); ++k) n *= shape[k];
  return n;
}

std::vector<double> NpyArray::ToDoubles() const {
  std::vector<double> out(static_cast<size_t>(size()));
  for (int64_t k = 0; k < size(); ++k) out[k] = Element(*this, k);
  return out;
}

std::vector<float> NpyArray::ToFloats() const {
  std::vector<float> out(static_cast<size_t>(size()));
  for (int64_t k = 0; k < size(); ++k) {
    out[k] = static_cast<float>(Element(*this, k));
  }
  return out;
}

std::vector<int64_t> NpyArray::ToInts() const {
  if (kind == 'f') Fail("integer array expected");
  std::vector<int64_t> out(static_cast<size_t>(size()));
  const bool wide = item_size == 8;
  for (int64_t k = 0; k < size(); ++k) {
    const uint8_t* p = data.data() + k * item_size;
    out[k] = wide ? Load<int64_t>(p)
                  : static_cast<int64_t>(Element(*this, k));
  }
  return out;
}

NpyArray NpyArray::FromDoubles(const std::vector<double>& values,
                               std::vector<int64_t> shape) {
  NpyArray a;
  a.kind = 'f';
  a.item_size = 8;
  a.shape = std::move(shape);
  if (a.size() != static_cast<int64_t>(values.size())) Fail("shape mismatch");
  a.data.resize(values.size() * sizeof(double));
  std::memcpy(a.data.data(), values.data(), a.data.size());
  return a;
}

NpyArray NpyArray::FromFloats(const std::vector<float>& values,
                              std::vector<int64_t> shape) {
  NpyArray a;
  a.kind = 'f';
  a.item_size = 4;
  a.shape = std::move(shape);
  if (a.size() != static_cast<int64_t>(values.size())) Fail("shape mismatch");
  a.data.resize(values.size() * sizeof(float));
  std::memcpy(a.data.data(), values.data(), a.data.size());
  return a;
}

NpyArray NpyArray::FromInts(const std::vector<int32_t>& values,
                            std::vector<int64_t> shape) {
  NpyArray a;
  a.kind = 'i';
  a.item_size = 4;
  a.shape = std::move(shape);
  if (a.size() != static_cast<int64_t>(values.size())) Fail("shape mismatch");
  a.data.resize(values.size() * sizeof(int32_t));
  std::memcpy(a.data.data(), values.data(), a.data.size());
  return a;
}

NpyArray ParseNpy(const std::vector<uint8_t>& bytes) {
  static const char kMagic[] = "\x93NUMPY";
  if (bytes.size() < 10 || std::memcmp(bytes.data(), kMagic, 6) != 0) {
    Fail("not a .npy file");
  }
  const int major = bytes[6];
  const size_t length_bytes = major == 1 ? 2 : 4;
  if (bytes.size() < 8 + length_bytes) Fail("truncated .npy header");
  const size_t header_length =
      static_cast<size_t>(Le(bytes.data() + 8, static_cast<int>(length_bytes)));
  const size_t data_start = 8 + length_bytes + header_length;
  if (bytes.size() < data_start) Fail("truncated .npy header");
  const std::string header(bytes.begin() + 8 + length_bytes,
                           bytes.begin() + data_start);
  static const std::regex kDescr("'descr':\\s*'([<>|=])([a-zA-Z])(\\d+)'");
  static const std::regex kFortran("'fortran_order':\\s*(True|False)");
  static const std::regex kShape("'shape':\\s*\\(([^)]*)\\)");
  std::smatch descr, fortran, shape;
  if (!std::regex_search(header, descr, kDescr) ||
      !std::regex_search(header, fortran, kFortran) ||
      !std::regex_search(header, shape, kShape)) {
    Fail("unsupported .npy header " + header);
  }
  if (descr[1] == ">") Fail("big-endian .npy arrays are not supported");
  NpyArray a;
  a.kind = descr[2].str()[0];
  a.item_size = std::stoi(descr[3]);
  if (a.kind != 'f' && a.kind != 'i' && a.kind != 'u' && a.kind != 'b') {
    Fail("unsupported dtype " + descr[2].str() + descr[3].str());
  }
  std::istringstream dims(shape[1].str());
  std::string dim;
  while (std::getline(dims, dim, ',')) {
    if (dim.find_first_not_of(" ") != std::string::npos) {
      a.shape.push_back(std::stoll(dim));
    }
  }
  const size_t n = static_cast<size_t>(a.size()) * a.item_size;
  if (bytes.size() < data_start + n) Fail("truncated .npy data");
  a.data.assign(bytes.begin() + data_start, bytes.begin() + data_start + n);
  if (fortran[1] == "True" && a.shape.size() > 1) {
    // Column-major (numpy saves transposed views so): reorder to C order.
    const std::vector<uint8_t> column_major = std::move(a.data);
    a.data.assign(n, 0);
    const size_t dims = a.shape.size();
    std::vector<int64_t> index(dims, 0);
    for (int64_t c = 0; c < a.size(); ++c) {
      int64_t f = 0;  // Fortran offset of the C-order element `c`.
      for (size_t d = dims; d-- > 0;) f = f * a.shape[d] + index[d];
      std::memcpy(a.data.data() + c * a.item_size,
                  column_major.data() + f * a.item_size, a.item_size);
      for (size_t d = dims; d-- > 0;) {  // Next C-order index.
        if (++index[d] < a.shape[d]) break;
        index[d] = 0;
      }
    }
  }
  return a;
}

std::vector<uint8_t> SerializeNpy(const NpyArray& array) {
  std::string header = std::string("{'descr': '") +
                       (array.item_size == 1 ? '|' : '<') + array.kind +
                       std::to_string(array.item_size) +
                       "', 'fortran_order': False, 'shape': " +
                       ShapeText(array.shape) + ", }";
  // Magic, version and length take 10 bytes; pad to a multiple of 64.
  const size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header.push_back('\n');
  std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  PutLe(header.size(), 2, &out);
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), array.data.begin(), array.data.end());
  return out;
}

NpyArray ReadNpy(const std::string& path) {
  try {
    return ParseNpy(ReadBytes(path));
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

void WriteNpy(const std::string& path, const NpyArray& array) {
  WriteBytes(path, SerializeNpy(array));
}

NpzArchive ReadNpz(const std::string& path) {
  const std::vector<uint8_t> zip = ReadBytes(path);
  const uint8_t* z = zip.data();
  const auto fail = [&](const std::string& message) {
    Fail(path + ": " + message);
  };
  // End of central directory (followed by a comment of up to 64 KiB).
  if (zip.size() < 22) fail("not a zip archive");
  size_t eocd = zip.size() - 22;
  while (Le(z + eocd, 4) != 0x06054b50) {
    if (eocd == 0 || zip.size() - eocd > 22 + 0xFFFF) {
      fail("not a zip archive");
    }
    --eocd;
  }
  uint64_t entries = Le(z + eocd + 10, 2);
  uint64_t directory = Le(z + eocd + 16, 4);
  if ((entries == 0xFFFF || directory == 0xFFFFFFFF) && eocd >= 20 &&
      Le(z + eocd - 20, 4) == 0x07064b50) {
    const uint64_t record = Le(z + eocd - 20 + 8, 8);
    if (record + 56 > zip.size() || Le(z + record, 4) != 0x06064b50) {
      fail("bad ZIP64 end record");
    }
    entries = Le(z + record + 32, 8);
    directory = Le(z + record + 48, 8);
  }
  NpzArchive out;
  size_t at = static_cast<size_t>(directory);
  for (uint64_t k = 0; k < entries; ++k) {
    if (at + 46 > zip.size() || Le(z + at, 4) != 0x02014b50) {
      fail("bad central directory");
    }
    const int method = static_cast<int>(Le(z + at + 10, 2));
    const uint32_t crc = static_cast<uint32_t>(Le(z + at + 16, 4));
    uint64_t packed = Le(z + at + 20, 4);
    uint64_t size = Le(z + at + 24, 4);
    const size_t name_length = Le(z + at + 28, 2);
    const size_t extra_length = Le(z + at + 30, 2);
    const size_t comment_length = Le(z + at + 32, 2);
    uint64_t local = Le(z + at + 42, 4);
    const std::string name(reinterpret_cast<const char*>(z + at + 46),
                           name_length);
    // ZIP64 extra field: the 8-byte values of the saturated fields.
    size_t e = at + 46 + name_length;
    const size_t extra_end = e + extra_length;
    while (e + 4 <= extra_end) {
      const uint64_t id = Le(z + e, 2);
      const size_t length = Le(z + e + 2, 2);
      if (id == 0x0001) {
        size_t f = e + 4;
        if (size == 0xFFFFFFFF) {
          size = Le(z + f, 8);
          f += 8;
        }
        if (packed == 0xFFFFFFFF) {
          packed = Le(z + f, 8);
          f += 8;
        }
        if (local == 0xFFFFFFFF) local = Le(z + f, 8);
      }
      e += 4 + length;
    }
    at = extra_end + comment_length;
    if (local + 30 > zip.size() || Le(z + local, 4) != 0x04034b50) {
      fail("bad local header of " + name);
    }
    const size_t data = static_cast<size_t>(local) + 30 +
                        Le(z + local + 26, 2) + Le(z + local + 28, 2);
    if (data + packed > zip.size()) fail("truncated member " + name);
    std::vector<uint8_t> bytes;
    if (method == 0) {
      bytes.assign(z + data, z + data + packed);
    } else if (method == 8) {
      bytes = Inflate(z + data, static_cast<size_t>(packed),
                      static_cast<size_t>(size));
    } else {
      fail("unsupported compression of " + name);
    }
    if (crc32(0L, bytes.data(), static_cast<uInt>(bytes.size())) != crc) {
      fail("CRC mismatch in " + name);
    }
    std::string key = name;
    if (key.size() > 4 && key.compare(key.size() - 4, 4, ".npy") == 0) {
      key.resize(key.size() - 4);
    }
    try {
      out[key] = ParseNpy(bytes);
    } catch (const std::runtime_error& error) {
      fail(name + ": " + error.what());
    }
  }
  return out;
}

void WriteNpz(const std::string& path, const NpzArchive& arrays,
              bool compress) {
  std::vector<uint8_t> zip;
  std::vector<uint8_t> directory;
  const uint16_t method = compress ? 8 : 0;
  for (const auto& [key, array] : arrays) {
    const std::string name = key + ".npy";
    const std::vector<uint8_t> raw = SerializeNpy(array);
    const std::vector<uint8_t> packed = compress ? Deflate(raw) : raw;
    const uint32_t crc = static_cast<uint32_t>(
        crc32(0L, raw.data(), static_cast<uInt>(raw.size())));
    if (raw.size() >= 0xFFFFFFFFu || zip.size() >= 0xFFFFFFFFu) {
      Fail(path + ": members of 4 GiB or more are not supported");
    }
    const size_t local = zip.size();
    // Local header: version 2.0, no flags, DOS date 1980-01-01.
    PutLe(0x04034b50, 4, &zip);
    PutLe(20, 2, &zip);
    PutLe(0, 2, &zip);
    PutLe(method, 2, &zip);
    PutLe(0, 2, &zip);
    PutLe(0x21, 2, &zip);
    PutLe(crc, 4, &zip);
    PutLe(packed.size(), 4, &zip);
    PutLe(raw.size(), 4, &zip);
    PutLe(name.size(), 2, &zip);
    PutLe(0, 2, &zip);
    zip.insert(zip.end(), name.begin(), name.end());
    zip.insert(zip.end(), packed.begin(), packed.end());
    PutLe(0x02014b50, 4, &directory);
    PutLe(20, 2, &directory);
    PutLe(20, 2, &directory);
    PutLe(0, 2, &directory);
    PutLe(method, 2, &directory);
    PutLe(0, 2, &directory);
    PutLe(0x21, 2, &directory);
    PutLe(crc, 4, &directory);
    PutLe(packed.size(), 4, &directory);
    PutLe(raw.size(), 4, &directory);
    PutLe(name.size(), 2, &directory);
    PutLe(0, 2, &directory);  // Extra.
    PutLe(0, 2, &directory);  // Comment.
    PutLe(0, 2, &directory);  // Disk.
    PutLe(0, 2, &directory);  // Internal attributes.
    PutLe(0, 4, &directory);  // External attributes.
    PutLe(local, 4, &directory);
    directory.insert(directory.end(), name.begin(), name.end());
  }
  const size_t start = zip.size();
  zip.insert(zip.end(), directory.begin(), directory.end());
  PutLe(0x06054b50, 4, &zip);
  PutLe(0, 2, &zip);
  PutLe(0, 2, &zip);
  PutLe(arrays.size(), 2, &zip);
  PutLe(arrays.size(), 2, &zip);
  PutLe(directory.size(), 4, &zip);
  PutLe(start, 4, &zip);
  PutLe(0, 2, &zip);
  WriteBytes(path, zip);
}

}  // namespace gr
