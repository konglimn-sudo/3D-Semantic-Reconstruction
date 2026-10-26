// NumPy arrays on disk: .npy files and .npz archives (zip members
// <name>.npy, stored or deflate-compressed, ZIP64 records included, as
// numpy.savez / savez_compressed write them).

#ifndef GARAGE_RECON_GR_IO_NPY_H_
#define GARAGE_RECON_GR_IO_NPY_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gr {

struct NpyArray {
  char kind = 'f';    // 'f' float, 'i' signed, 'u' unsigned, 'b' bool.
  int item_size = 8;  // Bytes per element.
  std::vector<int64_t> shape;  // Empty: a scalar.
  std::vector<uint8_t> data;   // Little-endian, C order.

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
};

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
