// Non-owning view of an 8-bit image.

#ifndef GARAGE_RECON_GR_IMAGE_H_
#define GARAGE_RECON_GR_IMAGE_H_

#include <cstddef>
#include <cstdint>

namespace gr {

// Row-major, interleaved channels (H x W x C). A null `data` means "no image".
struct ImageView {
  const uint8_t* data = nullptr;
  int height = 0;
  int width = 0;
  int channels = 3;

  bool empty() const { return data == nullptr || height <= 0 || width <= 0; }
  const uint8_t* PixelAt(int row, int col) const {
    return data + (static_cast<size_t>(row) * static_cast<size_t>(width) +
                   static_cast<size_t>(col)) *
                      static_cast<size_t>(channels);
  }
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_IMAGE_H_
