// NumPy arrays on disk: .npy files and .npz archives (zip members
// <name>.npy, stored or deflate-compressed, ZIP64 records included, as
// numpy.savez / savez_compressed write them).

#ifndef GARAGE_RECON_GR_IO_NPY_H_
#define GARAGE_RECON_GR_IO_NPY_H_

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gr {

struct NpyArray {
  char kind = 'f';  // 'f' float, 'i' signed, 'u' unsigned, 'b' bool.
  int item_size = sizeof(double);  // Bytes per element.
  std::vector<int64_t> shape;      // Empty: a scalar.
  std::vector<uint8_t> data;       // Little-endian, C order.

  int64_t size() const;  // Number of elements.
  int64_t rows() const { return shape.empty() ? 1 : shape[0]; }
  // Elements per row (product of the trailing dimensions).
  int64_t columns() const;

  // Elements converted (float16 included); throws for an unknown type.
  std::vector<double> ToDoubles() const;
  std::vector<float> ToFloats() const;
  std::vector<int64_t> ToInts() const;

  static NpyArray FromDoubles(const std::vector<double>& values,
                              std::vector<int64_t> shape);
  static NpyArray FromFloats(const std::vector<float>& values,
                             std::vector<int64_t> shape);
  static NpyArray FromInts(const std::vector<int32_t>& values,
                           std::vector<int64_t> shape);
  // Any arithmetic element type: <f8, <i2, <u1, ... as its C++ type.
  template <typename T>
  static NpyArray Of(const std::vector<T>& values, std::vector<int64_t> shape);
  // float16, rounded to nearest even (numpy's astype(np.float16)).
  static NpyArray FromHalfs(const std::vector<float>& values,
                            std::vector<int64_t> shape);
  // numpy bool (|b1) from 0 / non-zero bytes.
  static NpyArray FromBools(const std::vector<uint8_t>& values,
                            std::vector<int64_t> shape);
};

template <typename T>
NpyArray NpyArray::Of(const std::vector<T>& values,
                      std::vector<int64_t> shape) {
  static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                "NpyArray::Of: arithmetic element types (not bool)");
  NpyArray array;
  array.kind = std::is_floating_point_v<T> ? 'f'
               : std::is_signed_v<T>       ? 'i'
                                           : 'u';
  array.item_size = static_cast<int>(sizeof(T));
  array.shape = std::move(shape);
  if (array.size() != static_cast<int64_t>(values.size())) {
    throw std::runtime_error("NpyArray::Of: shape mismatch");
  }
  array.data.resize(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(array.data.data(), values.data(), array.data.size());
  }
  return array;
}

// Member name (without .npy) -> array.
using NpzArchive = std::map<std::string, NpyArray>;

// In-memory .npy content (Fortran-ordered arrays are reordered to C
// order); throws std::runtime_error on a malformed or unsupported header
// (big-endian, structured or object dtypes).
NpyArray ParseNpy(const std::vector<uint8_t>& bytes);
std::vector<uint8_t> SerializeNpy(const NpyArray& array);

NpyArray ReadNpy(const std::string& path);
void WriteNpy(const std::string& path, const NpyArray& array);

NpzArchive ReadNpz(const std::string& path);
// `compress`: deflate as savez_compressed (else stored, as savez).
void WriteNpz(const std::string& path, const NpzArchive& arrays,
              bool compress = true);

}  // namespace gr

#endif  // GARAGE_RECON_GR_IO_NPY_H_
