#include "gr/semantics/slot_paint.h"

#include <algorithm>
#include <cmath>

#include "gr/core/geometry.h"
#include "gr/core/image.h"

namespace gr {
namespace {

// The hue circle is cut into kHueSectors sectors of kDegreesPerHueSector,
// red starting at sector 0, green at kGreenSector and blue at kBlueSector.
constexpr double kHueSectors = 6.0;
constexpr double kDegreesPerHueSector = 60.0;
constexpr double kGreenSector = 2.0;
constexpr double kBlueSector = 4.0;
// An inset outline with fewer vertices than this has vanished.
constexpr size_t kMinPolygonVertices = 3;
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;

// Hue in degrees [0, 360) and saturation (max - min) / max of an RGB colour.
void HueSaturation(double red, double green, double blue, double* hue,
                   double* saturation) {
  const double brightest = std::max({red, green, blue});
  const double darkest = std::min({red, green, blue});
  const double chroma = brightest - darkest;
  *saturation = brightest > 0.0 ? chroma / brightest : 0.0;
  if (chroma <= 0.0) {
    *hue = 0.0;
  } else if (brightest == red) {
    *hue = kDegreesPerHueSector *
           std::fmod((green - blue) / chroma + kHueSectors, kHueSectors);
  } else if (brightest == green) {
    *hue = kDegreesPerHueSector * ((blue - red) / chroma + kGreenSector);
  } else {
    *hue = kDegreesPerHueSector * ((red - green) / chroma + kBlueSector);
  }
}

}  // namespace

std::vector<SlotPaint> MeasureSlotPaint(
    const std::vector<std::array<Vec2, kParkingSlotCorners>>& slots,
    const std::vector<uint8_t>& rgb, const std::vector<uint8_t>& observed,
    const RasterFrame& frame, const SlotPaintConfig& config) {
  std::vector<SlotPaint> out(slots.size());
  for (size_t slot = 0; slot < slots.size(); ++slot) {
    const std::vector<Vec2> inner =
        InsetConvex(std::vector<Vec2>(slots[slot].begin(), slots[slot].end()),
                    config.inset);
    if (inner.size() < kMinPolygonVertices) continue;
    Vec2 lower = inner[0];
    Vec2 upper = inner[0];
    for (const Vec2& vertex : inner) {
      lower = lower.cwiseMin(vertex);
      upper = upper.cwiseMax(vertex);
    }
    const int first_column = std::max(
        0, static_cast<int>(std::floor((lower.x() - frame.x0) / frame.res)));
    const int last_column = std::min(
        frame.width - 1,
        static_cast<int>(std::floor((upper.x() - frame.x0) / frame.res)));
    const int first_row = std::max(
        0, static_cast<int>(std::floor((lower.y() - frame.y0) / frame.res)));
    const int last_row = std::min(
        frame.height - 1,
        static_cast<int>(std::floor((upper.y() - frame.y0) / frame.res)));
    SlotPaint& paint = out[slot];
    int in_band = 0;
    for (int row = first_row; row <= last_row; ++row) {
      for (int column = first_column; column <= last_column; ++column) {
        const Vec2 center(frame.x0 + (column + kCellCentre) * frame.res,
                          frame.y0 + (row + kCellCentre) * frame.res);
        if (!PointInPolygon(inner, center)) continue;
        ++paint.cells;
        const size_t cell = static_cast<size_t>(row) * frame.width + column;
        if (!observed[cell]) continue;
        ++paint.observed;
        const size_t pixel = kRgbChannels * cell;
        const Vec3 colour(rgb[pixel], rgb[pixel + 1], rgb[pixel + 2]);
        paint.mean_rgb += colour;
        double hue = 0.0;
        double saturation = 0.0;
        HueSaturation(colour.x(), colour.y(), colour.z(), &hue, &saturation);
        in_band += saturation >= config.min_saturation &&
                   hue >= config.hue_min && hue <= config.hue_max;
      }
    }
    if (paint.observed > 0) {
      paint.mean_rgb /= paint.observed;
      paint.in_band = static_cast<double>(in_band) / paint.observed;
    }
  }
  return out;
}

}  // namespace gr
