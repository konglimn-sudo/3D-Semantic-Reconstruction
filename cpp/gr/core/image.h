// Non-owning view of an 8-bit image.

#ifndef GARAGE_RECON_GR_CORE_IMAGE_H_
#define GARAGE_RECON_GR_CORE_IMAGE_H_

#include <cstddef>
#include <cstdint>

namespace gr {

// Channels of an RGB image.
inline constexpr int kRgbChannels = 3;

// Row-major, interleaved channels (H x W x C). A null `data` means "no image".
struct ImageView {
  const uint8_t* data = nullptr;
  int height = 0;
  int width = 0;
  int channels = kRgbChannels;

  // Whether there is no image to read.
  bool empty() const { return data == nullptr || height <= 0 || width <= 0; }
  // The first channel of the pixel at (`row`, `column`).
  const uint8_t* PixelAt(int row, int column) const {
    return data + (static_cast<size_t>(row) * static_cast<size_t>(width) +
                   static_cast<size_t>(column)) *
                      static_cast<size_t>(channels);
  }
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_IMAGE_H_
