#include "gr/io/npy.h"

#include <zlib.h>

#include <climits>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace gr {
namespace {

// --- The .npy format (numpy/lib/format.py) ----------------------------------

// "\x93NUMPY", the major and minor version, then the length of the header
// text: 2 bytes in version 1, 4 from version 2 on.
constexpr char kNpyMagic[] = "\x93NUMPY";
constexpr size_t kNpyMagicSize = sizeof(kNpyMagic) - 1;
constexpr size_t kNpyMajorVersionAt = kNpyMagicSize;
constexpr size_t kNpyHeaderLengthAt = kNpyMagicSize + 2;
constexpr size_t kNpyV1LengthBytes = 2;
constexpr size_t kNpyV2LengthBytes = 4;
constexpr size_t kNpyMinSize = kNpyHeaderLengthAt + kNpyV1LengthBytes;
// The version written, and the alignment numpy pads the header to.
constexpr uint8_t kNpyWrittenMajor = 1;
constexpr uint8_t kNpyWrittenMinor = 0;
constexpr size_t kNpyHeaderAlignment = 64;

// The item size of a dtype stored as T (float16 is stored as 16 bits).
template <typename T>
constexpr int kSizeOf = static_cast<int>(sizeof(T));

// --- IEEE 754 binary16 and binary32
// -------------------------------------------

constexpr int kHalfMantissaBits = 10;
constexpr int kFloatMantissaBits = 23;
// Shift between the mantissas, and between the sign bits.
constexpr int kMantissaShift = kFloatMantissaBits - kHalfMantissaBits;
constexpr int kSignShift = 16;
constexpr int kHalfExponentBias = 15;
constexpr int kFloatExponentBias = 127;
constexpr uint32_t kHalfSignBit = 0x8000;
constexpr uint32_t kHalfExponentMask = 0x1F;  // After the mantissa shift.
constexpr uint32_t kHalfMantissaMask = (1u << kHalfMantissaBits) - 1;
constexpr uint32_t kHalfImplicitBit = 1u << kHalfMantissaBits;
constexpr uint32_t kHalfInfinity = kHalfExponentMask << kHalfMantissaBits;
constexpr uint32_t kHalfQuietNan = kHalfInfinity | (kHalfImplicitBit >> 1);
constexpr uint32_t kFloatMagnitudeMask = 0x7FFFFFFF;
constexpr uint32_t kFloatMantissaMask = (1u << kFloatMantissaBits) - 1;
constexpr uint32_t kFloatImplicitBit = 1u << kFloatMantissaBits;
constexpr uint32_t kFloatInfinity = 0xFFu << kFloatMantissaBits;
// The float mantissa bits a half drops, and half of their range (the
// rounding point).
constexpr uint32_t kDroppedBitsMask = (1u << kMantissaShift) - 1;
constexpr uint32_t kDroppedHalfway = 1u << (kMantissaShift - 1);
// Biased float exponents: from this one on a float overflows a half; from
// this one on it is a normal half; and the float exponent that becomes the
// half exponent 0 after rebiasing.
constexpr uint32_t kFirstOverflowExponent =
    kFloatExponentBias + kHalfExponentBias + 1;
constexpr uint32_t kFirstNormalHalfExponent =
    kFloatExponentBias - kHalfExponentBias + 1;
constexpr uint32_t kHalfRebias = kFloatExponentBias - kHalfExponentBias;
// Subnormal halves count units of 2^-24; the shift that turns a float
// mantissa (with its implicit bit) into such units is
// kSubnormalShiftBase - exponent, and floats shifted further are zero.
constexpr int kHalfSubnormalUnitExponent = 24;
constexpr int kSubnormalShiftBase =
    kFloatExponentBias + kFloatMantissaBits - kHalfSubnormalUnitExponent;

// --- The zip format (PKWARE APPNOTE.TXT)
// --------------------------------------

namespace zip {

constexpr uint64_t kLocalHeaderSignature = 0x04034b50;
constexpr uint64_t kCentralHeaderSignature = 0x02014b50;
constexpr uint64_t kEndRecordSignature = 0x06054b50;
constexpr uint64_t kZip64EndRecordSignature = 0x06064b50;
constexpr uint64_t kZip64LocatorSignature = 0x07064b50;

// Local file header: fixed part, then the name and the extra field.
constexpr size_t kLocalHeaderSize = 30;
constexpr size_t kLocalNameLengthAt = 26;
constexpr size_t kLocalExtraLengthAt = 28;

// Central directory file header.
constexpr size_t kCentralHeaderSize = 46;
constexpr size_t kCentralMethodAt = 10;
constexpr size_t kCentralCrcAt = 16;
constexpr size_t kCentralPackedSizeAt = 20;
constexpr size_t kCentralSizeAt = 24;
constexpr size_t kCentralNameLengthAt = 28;
constexpr size_t kCentralExtraLengthAt = 30;
constexpr size_t kCentralCommentLengthAt = 32;
constexpr size_t kCentralLocalHeaderAt = 42;

// End of central directory record, followed by a comment of at most
// kMaxCommentSize bytes.
constexpr size_t kEndRecordSize = 22;
constexpr size_t kEndEntriesAt = 10;
constexpr size_t kEndDirectoryAt = 16;
constexpr size_t kMaxCommentSize = 0xFFFF;

// ZIP64 end of central directory locator (just before the end record) and
// record.
constexpr size_t kZip64LocatorSize = 20;
constexpr size_t kZip64LocatorRecordAt = 8;
constexpr size_t kZip64RecordSize = 56;
constexpr size_t kZip64EntriesAt = 32;
constexpr size_t kZip64DirectoryAt = 48;

// Extra fields: an id and a length (2 bytes each), then the data.
constexpr size_t kExtraHeaderSize = 4;
constexpr size_t kExtraLengthAt = 2;
constexpr uint64_t kZip64ExtraId = 0x0001;

// Fields that hold these values are saturated: the ZIP64 records have them.
constexpr uint64_t kSaturated16 = 0xFFFF;
constexpr uint64_t kSaturated32 = 0xFFFFFFFF;

constexpr int kStored = 0;
constexpr int kDeflated = 8;

// What WriteNpz records: version 2.0 (to make and to extract), no flags,
// the DOS time 00:00 of the DOS date 1980-01-01.
constexpr uint64_t kVersion = 20;
constexpr uint64_t kDosDate1980 = 0x21;

}  // namespace zip

// zlib's default memory level for deflate.
constexpr int kDeflateMemoryLevel = 8;

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

// The little-endian unsigned number of `width` bytes at `bytes`.
uint64_t LoadLittleEndian(const uint8_t* bytes, size_t width) {
  uint64_t value = 0;
  for (size_t k = width; k-- > 0;) value = (value << CHAR_BIT) | bytes[k];
  return value;
}
uint64_t Load16(const uint8_t* bytes) {
  return LoadLittleEndian(bytes, sizeof(uint16_t));
}
uint64_t Load32(const uint8_t* bytes) {
  return LoadLittleEndian(bytes, sizeof(uint32_t));
}
uint64_t Load64(const uint8_t* bytes) {
  return LoadLittleEndian(bytes, sizeof(uint64_t));
}

// Appends `value` as a little-endian number of `width` bytes.
void StoreLittleEndian(uint64_t value, size_t width,
                       std::vector<uint8_t>* out) {
  for (size_t k = 0; k < width; ++k) {
    out->push_back(static_cast<uint8_t>(value >> (CHAR_BIT * k)));
  }
}
void Store16(uint64_t value, std::vector<uint8_t>* out) {
  StoreLittleEndian(value, sizeof(uint16_t), out);
}
void Store32(uint64_t value, std::vector<uint8_t>* out) {
  StoreLittleEndian(value, sizeof(uint32_t), out);
}

float HalfToFloat(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & kHalfSignBit)
                        << kSignShift;
  const uint32_t exponent = (half >> kHalfMantissaBits) & kHalfExponentMask;
  uint32_t mantissa = half & kHalfMantissaMask;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {  // Subnormal: normalise.
      int shifts = -1;
      do {
        ++shifts;
        mantissa <<= 1;
      } while ((mantissa & kHalfImplicitBit) == 0);
      bits =
          sign |
          static_cast<uint32_t>(kFloatExponentBias - kHalfExponentBias - shifts)
              << kFloatMantissaBits |
          (mantissa & kHalfMantissaMask) << kMantissaShift;
    }
  } else if (exponent == kHalfExponentMask) {  // Infinity or NaN.
    bits = sign | kFloatInfinity | mantissa << kMantissaShift;
  } else {
    bits = sign |
           (exponent + kFloatExponentBias - kHalfExponentBias)
               << kFloatMantissaBits |
           mantissa << kMantissaShift;
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Round to nearest even, as numpy's float -> float16 cast.
uint16_t FloatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign =
      static_cast<uint16_t>((bits >> kSignShift) & kHalfSignBit);
  const uint32_t magnitude = bits & kFloatMagnitudeMask;
  const uint32_t exponent = magnitude >> kFloatMantissaBits;
  if (magnitude > kFloatInfinity) {  // NaN: keep it quiet.
    return static_cast<uint16_t>(
        sign | kHalfQuietNan |
        ((magnitude >> kMantissaShift) & kHalfMantissaMask));
  }
  if (exponent >= kFirstOverflowExponent) {
    return static_cast<uint16_t>(sign | kHalfInfinity);
  }
  if (exponent >= kFirstNormalHalfExponent) {
    // Normal half (a carry may round up to infinity).
    uint32_t half = ((exponent - kHalfRebias) << kHalfMantissaBits) |
                    ((magnitude >> kMantissaShift) & kHalfMantissaMask);
    const uint32_t dropped = magnitude & kDroppedBitsMask;
    if (dropped > kDroppedHalfway ||
        (dropped == kDroppedHalfway && (half & 1))) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  // Subnormal half: units of 2^-24.
  const int shift = kSubnormalShiftBase - static_cast<int>(exponent);
  if (exponent == 0 || shift > kHalfSubnormalUnitExponent) return sign;
  const uint32_t mantissa =
      (magnitude & kFloatMantissaMask) | kFloatImplicitBit;
  uint32_t half = mantissa >> shift;
  const uint32_t dropped = mantissa & ((1u << shift) - 1);
  const uint32_t halfway = 1u << (shift - 1);
  if (dropped > halfway || (dropped == halfway && (half & 1))) ++half;
  return static_cast<uint16_t>(sign | half);
}

template <typename T>
T Load(const uint8_t* bytes) {
  T value;
  std::memcpy(&value, bytes, sizeof(T));
  return value;
}

// Element `index` as a double.
double Element(const NpyArray& array, int64_t index) {
  const uint8_t* bytes = array.data.data() + index * array.item_size;
  switch (array.kind) {
    case 'f':
      if (array.item_size == kSizeOf<uint16_t>) {
        return HalfToFloat(Load<uint16_t>(bytes));
      }
      if (array.item_size == kSizeOf<float>) return Load<float>(bytes);
      if (array.item_size == kSizeOf<double>) return Load<double>(bytes);
      break;
    case 'i':
      if (array.item_size == kSizeOf<int8_t>) return Load<int8_t>(bytes);
      if (array.item_size == kSizeOf<int16_t>) return Load<int16_t>(bytes);
      if (array.item_size == kSizeOf<int32_t>) return Load<int32_t>(bytes);
      if (array.item_size == kSizeOf<int64_t>) {
        return static_cast<double>(Load<int64_t>(bytes));
      }
      break;
    case 'u':
    case 'b':
      if (array.item_size == kSizeOf<uint8_t>) return Load<uint8_t>(bytes);
      if (array.item_size == kSizeOf<uint16_t>) return Load<uint16_t>(bytes);
      if (array.item_size == kSizeOf<uint32_t>) return Load<uint32_t>(bytes);
      if (array.item_size == kSizeOf<uint64_t>) {
        return static_cast<double>(Load<uint64_t>(bytes));
      }
      break;
  }
  Fail(std::string("unsupported dtype ") + array.kind +
       std::to_string(array.item_size));
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
  // A negative window size: raw deflate data, as zip members hold.
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
  if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS,
                   kDeflateMemoryLevel, Z_DEFAULT_STRATEGY) != Z_OK) {
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

// An array of `kind` with the item size of T, holding `values`.
template <typename T>
NpyArray FromValues(char kind, const std::vector<T>& values,
                    std::vector<int64_t> shape) {
  NpyArray array;
  array.kind = kind;
  array.item_size = static_cast<int>(sizeof(T));
  array.shape = std::move(shape);
  if (array.size() != static_cast<int64_t>(values.size())) {
    Fail("shape mismatch");
  }
  array.data.resize(values.size() * sizeof(T));
  std::memcpy(array.data.data(), values.data(), array.data.size());
  return array;
}

}  // namespace

int64_t NpyArray::size() const {
  int64_t count = 1;
  for (int64_t dimension : shape) count *= dimension;
  return count;
}

int64_t NpyArray::columns() const {
  int64_t count = 1;
  for (size_t k = 1; k < shape.size(); ++k) count *= shape[k];
  return count;
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
  // 64-bit integers are copied: a double cannot hold them all.
  const bool wide = item_size == kSizeOf<int64_t>;
  for (int64_t k = 0; k < size(); ++k) {
    const uint8_t* bytes = data.data() + k * item_size;
    out[k] =
        wide ? Load<int64_t>(bytes) : static_cast<int64_t>(Element(*this, k));
  }
  return out;
}

NpyArray NpyArray::FromDoubles(const std::vector<double>& values,
                               std::vector<int64_t> shape) {
  return FromValues('f', values, std::move(shape));
}

NpyArray NpyArray::FromFloats(const std::vector<float>& values,
                              std::vector<int64_t> shape) {
  return FromValues('f', values, std::move(shape));
}

NpyArray NpyArray::FromInts(const std::vector<int32_t>& values,
                            std::vector<int64_t> shape) {
  return FromValues('i', values, std::move(shape));
}

NpyArray NpyArray::FromHalfs(const std::vector<float>& values,
                             std::vector<int64_t> shape) {
  std::vector<uint16_t> bits(values.size());
  for (size_t k = 0; k < values.size(); ++k) bits[k] = FloatToHalf(values[k]);
  NpyArray array = Of(bits, std::move(shape));
  array.kind = 'f';
  return array;
}

NpyArray NpyArray::FromBools(const std::vector<uint8_t>& values,
                             std::vector<int64_t> shape) {
  std::vector<uint8_t> bytes(values.size());
  for (size_t k = 0; k < values.size(); ++k) bytes[k] = values[k] != 0;
  NpyArray array = Of(bytes, std::move(shape));
  array.kind = 'b';
  return array;
}

NpyArray ParseNpy(const std::vector<uint8_t>& bytes) {
  if (bytes.size() < kNpyMinSize ||
      std::memcmp(bytes.data(), kNpyMagic, kNpyMagicSize) != 0) {
    Fail("not a .npy file");
  }
  const int major = bytes[kNpyMajorVersionAt];
  const size_t length_bytes =
      major == 1 ? kNpyV1LengthBytes : kNpyV2LengthBytes;
  if (bytes.size() < kNpyHeaderLengthAt + length_bytes) {
    Fail("truncated .npy header");
  }
  const size_t header_length = static_cast<size_t>(
      LoadLittleEndian(bytes.data() + kNpyHeaderLengthAt, length_bytes));
  const size_t header_start = kNpyHeaderLengthAt + length_bytes;
  const size_t data_start = header_start + header_length;
  if (bytes.size() < data_start) Fail("truncated .npy header");
  const std::string header(bytes.begin() + header_start,
                           bytes.begin() + data_start);
  static const std::regex kDescr("'descr':\\s*'([<>|=])([a-zA-Z])(\\d+)'");
  static const std::regex kFortran("'fortran_order':\\s*(True|False)");
  static const std::regex kShape("'shape':\\s*\\(([^)]*)\\)");
  // The capture groups of kDescr.
  constexpr int kByteOrderGroup = 1;
  constexpr int kKindGroup = 2;
  constexpr int kItemSizeGroup = 3;
  std::smatch descr;
  std::smatch fortran;
  std::smatch shape;
  if (!std::regex_search(header, descr, kDescr) ||
      !std::regex_search(header, fortran, kFortran) ||
      !std::regex_search(header, shape, kShape)) {
    Fail("unsupported .npy header " + header);
  }
  if (descr[kByteOrderGroup] == ">") {
    Fail("big-endian .npy arrays are not supported");
  }
  NpyArray array;
  array.kind = descr[kKindGroup].str()[0];
  array.item_size = std::stoi(descr[kItemSizeGroup]);
  if (array.kind != 'f' && array.kind != 'i' && array.kind != 'u' &&
      array.kind != 'b') {
    Fail("unsupported dtype " + descr[kKindGroup].str() +
         descr[kItemSizeGroup].str());
  }
  std::istringstream dimensions(shape[1].str());
  std::string dimension;
  while (std::getline(dimensions, dimension, ',')) {
    if (dimension.find_first_not_of(" ") != std::string::npos) {
      array.shape.push_back(std::stoll(dimension));
    }
  }
  const size_t data_size = static_cast<size_t>(array.size()) * array.item_size;
  if (bytes.size() < data_start + data_size) Fail("truncated .npy data");
  array.data.assign(bytes.begin() + data_start,
                    bytes.begin() + data_start + data_size);
  if (fortran[1] == "True" && array.shape.size() > 1) {
    // Column-major (numpy saves transposed views so): reorder to C order.
    const std::vector<uint8_t> column_major = std::move(array.data);
    array.data.assign(data_size, 0);
    const size_t rank = array.shape.size();
    std::vector<int64_t> index(rank, 0);
    for (int64_t c_offset = 0; c_offset < array.size(); ++c_offset) {
      int64_t fortran_offset = 0;  // Of the C-order element `c_offset`.
      for (size_t axis = rank; axis-- > 0;) {
        fortran_offset = fortran_offset * array.shape[axis] + index[axis];
      }
      std::memcpy(array.data.data() + c_offset * array.item_size,
                  column_major.data() + fortran_offset * array.item_size,
                  array.item_size);
      for (size_t axis = rank; axis-- > 0;) {  // Next C-order index.
        if (++index[axis] < array.shape[axis]) break;
        index[axis] = 0;
      }
    }
  }
  return array;
}

std::vector<uint8_t> SerializeNpy(const NpyArray& array) {
  std::string header =
      std::string("{'descr': '") + (array.item_size == 1 ? '|' : '<') +
      array.kind + std::to_string(array.item_size) +
      "', 'fortran_order': False, 'shape': " + ShapeText(array.shape) + ", }";
  // The preamble, the header and its newline: padded to the alignment.
  const size_t total = kNpyMinSize + header.size() + 1;
  header.append(
      (kNpyHeaderAlignment - total % kNpyHeaderAlignment) % kNpyHeaderAlignment,
      ' ');
  header.push_back('\n');
  std::vector<uint8_t> out(kNpyMagic, kNpyMagic + kNpyMagicSize);
  out.push_back(kNpyWrittenMajor);
  out.push_back(kNpyWrittenMinor);
  Store16(header.size(), &out);
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), array.data.begin(), array.data.end());
  return out;
}

NpyArray ReadNpy(const std::string& path) {
  try {
    return ParseNpy(ReadBytes(path));
  } catch (const std::runtime_error& error) {
    throw std::runtime_error(path + ": " + error.what());
  }
}

void WriteNpy(const std::string& path, const NpyArray& array) {
  WriteBytes(path, SerializeNpy(array));
}

NpzArchive ReadNpz(const std::string& path) {
  const std::vector<uint8_t> archive = ReadBytes(path);
  const uint8_t* bytes = archive.data();
  const auto fail = [&](const std::string& message) {
    Fail(path + ": " + message);
  };
  // The end record, searched backwards over the comment.
  if (archive.size() < zip::kEndRecordSize) fail("not a zip archive");
  size_t end_record = archive.size() - zip::kEndRecordSize;
  while (Load32(bytes + end_record) != zip::kEndRecordSignature) {
    if (end_record == 0 || archive.size() - end_record >
                               zip::kEndRecordSize + zip::kMaxCommentSize) {
      fail("not a zip archive");
    }
    --end_record;
  }
  uint64_t entries = Load16(bytes + end_record + zip::kEndEntriesAt);
  uint64_t directory = Load32(bytes + end_record + zip::kEndDirectoryAt);
  if ((entries == zip::kSaturated16 || directory == zip::kSaturated32) &&
      end_record >= zip::kZip64LocatorSize &&
      Load32(bytes + end_record - zip::kZip64LocatorSize) ==
          zip::kZip64LocatorSignature) {
    const uint64_t record = Load64(bytes + end_record - zip::kZip64LocatorSize +
                                   zip::kZip64LocatorRecordAt);
    if (record + zip::kZip64RecordSize > archive.size() ||
        Load32(bytes + record) != zip::kZip64EndRecordSignature) {
      fail("bad ZIP64 end record");
    }
    entries = Load64(bytes + record + zip::kZip64EntriesAt);
    directory = Load64(bytes + record + zip::kZip64DirectoryAt);
  }
  NpzArchive out;
  size_t header = static_cast<size_t>(directory);
  for (uint64_t k = 0; k < entries; ++k) {
    if (header + zip::kCentralHeaderSize > archive.size() ||
        Load32(bytes + header) != zip::kCentralHeaderSignature) {
      fail("bad central directory");
    }
    const int method =
        static_cast<int>(Load16(bytes + header + zip::kCentralMethodAt));
    const uint32_t crc =
        static_cast<uint32_t>(Load32(bytes + header + zip::kCentralCrcAt));
    uint64_t packed = Load32(bytes + header + zip::kCentralPackedSizeAt);
    uint64_t size = Load32(bytes + header + zip::kCentralSizeAt);
    const size_t name_length =
        Load16(bytes + header + zip::kCentralNameLengthAt);
    const size_t extra_length =
        Load16(bytes + header + zip::kCentralExtraLengthAt);
    const size_t comment_length =
        Load16(bytes + header + zip::kCentralCommentLengthAt);
    uint64_t local = Load32(bytes + header + zip::kCentralLocalHeaderAt);
    const std::string name(
        reinterpret_cast<const char*>(bytes + header + zip::kCentralHeaderSize),
        name_length);
    // ZIP64 extra field: the 8-byte values of the saturated fields.
    size_t extra = header + zip::kCentralHeaderSize + name_length;
    const size_t extra_end = extra + extra_length;
    while (extra + zip::kExtraHeaderSize <= extra_end) {
      const uint64_t id = Load16(bytes + extra);
      const size_t length = Load16(bytes + extra + zip::kExtraLengthAt);
      if (id == zip::kZip64ExtraId) {
        size_t field = extra + zip::kExtraHeaderSize;
        if (size == zip::kSaturated32) {
          size = Load64(bytes + field);
          field += sizeof(uint64_t);
        }
        if (packed == zip::kSaturated32) {
          packed = Load64(bytes + field);
          field += sizeof(uint64_t);
        }
        if (local == zip::kSaturated32) local = Load64(bytes + field);
      }
      extra += zip::kExtraHeaderSize + length;
    }
    header = extra_end + comment_length;
    if (local + zip::kLocalHeaderSize > archive.size() ||
        Load32(bytes + local) != zip::kLocalHeaderSignature) {
      fail("bad local header of " + name);
    }
    const size_t data = static_cast<size_t>(local) + zip::kLocalHeaderSize +
                        Load16(bytes + local + zip::kLocalNameLengthAt) +
                        Load16(bytes + local + zip::kLocalExtraLengthAt);
    if (data + packed > archive.size()) fail("truncated member " + name);
    std::vector<uint8_t> member;
    if (method == zip::kStored) {
      member.assign(bytes + data, bytes + data + packed);
    } else if (method == zip::kDeflated) {
      member = Inflate(bytes + data, static_cast<size_t>(packed),
                       static_cast<size_t>(size));
    } else {
      fail("unsupported compression of " + name);
    }
    if (crc32(0L, member.data(), static_cast<uInt>(member.size())) != crc) {
      fail("CRC mismatch in " + name);
    }
    static const std::string kNpySuffix = ".npy";
    std::string key = name;
    if (key.size() > kNpySuffix.size() &&
        key.compare(key.size() - kNpySuffix.size(), kNpySuffix.size(),
                    kNpySuffix) == 0) {
      key.resize(key.size() - kNpySuffix.size());
    }
    try {
      out[key] = ParseNpy(member);
    } catch (const std::runtime_error& error) {
      fail(name + ": " + error.what());
    }
  }
  return out;
}

void WriteNpz(const std::string& path, const NpzArchive& arrays,
              bool compress) {
  std::vector<uint8_t> archive;
  std::vector<uint8_t> directory;
  const uint16_t method = compress ? zip::kDeflated : zip::kStored;
  for (const auto& [key, array] : arrays) {
    const std::string name = key + ".npy";
    const std::vector<uint8_t> raw = SerializeNpy(array);
    const std::vector<uint8_t> packed = compress ? Deflate(raw) : raw;
    const uint32_t crc = static_cast<uint32_t>(
        crc32(0L, raw.data(), static_cast<uInt>(raw.size())));
    if (raw.size() >= zip::kSaturated32 ||
        archive.size() >= zip::kSaturated32) {
      Fail(path + ": members of 4 GiB or more are not supported");
    }
    const size_t local = archive.size();
    Store32(zip::kLocalHeaderSignature, &archive);
    Store16(zip::kVersion, &archive);  // Needed to extract.
    Store16(0, &archive);              // Flags.
    Store16(method, &archive);
    Store16(0, &archive);  // Time.
    Store16(zip::kDosDate1980, &archive);
    Store32(crc, &archive);
    Store32(packed.size(), &archive);
    Store32(raw.size(), &archive);
    Store16(name.size(), &archive);
    Store16(0, &archive);  // Extra.
    archive.insert(archive.end(), name.begin(), name.end());
    archive.insert(archive.end(), packed.begin(), packed.end());
    Store32(zip::kCentralHeaderSignature, &directory);
    Store16(zip::kVersion, &directory);  // Made by.
    Store16(zip::kVersion, &directory);  // Needed to extract.
    Store16(0, &directory);              // Flags.
    Store16(method, &directory);
    Store16(0, &directory);  // Time.
    Store16(zip::kDosDate1980, &directory);
    Store32(crc, &directory);
    Store32(packed.size(), &directory);
    Store32(raw.size(), &directory);
    Store16(name.size(), &directory);
    Store16(0, &directory);  // Extra.
    Store16(0, &directory);  // Comment.
    Store16(0, &directory);  // Disk.
    Store16(0, &directory);  // Internal attributes.
    Store32(0, &directory);  // External attributes.
    Store32(local, &directory);
    directory.insert(directory.end(), name.begin(), name.end());
  }
  const size_t directory_start = archive.size();
  archive.insert(archive.end(), directory.begin(), directory.end());
  Store32(zip::kEndRecordSignature, &archive);
  Store16(0, &archive);              // This disk.
  Store16(0, &archive);              // The disk of the directory.
  Store16(arrays.size(), &archive);  // Entries on this disk.
  Store16(arrays.size(), &archive);  // Entries.
  Store32(directory.size(), &archive);
  Store32(directory_start, &archive);
  Store16(0, &archive);  // Comment.
  WriteBytes(path, archive);
}

}  // namespace gr
