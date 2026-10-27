#include "gr/semantics/slot_paint.h"

#include <algorithm>
#include <cmath>

#include "gr/core/geometry.h"

namespace gr {
namespace {

// Hue in degrees [0, 360) and saturation (max - min) / max of an RGB colour.
void HueSaturation(double r, double g, double b, double* hue,
                   double* saturation) {
  const double hi = std::max({r, g, b});
  const double lo = std::min({r, g, b});
  const double chroma = hi - lo;
  *saturation = hi > 0.0 ? chroma / hi : 0.0;
  if (chroma <= 0.0) {
    *hue = 0.0;
  } else if (hi == r) {
    *hue = 60.0 * std::fmod((g - b) / chroma + 6.0, 6.0);
  } else if (hi == g) {
    *hue = 60.0 * ((b - r) / chroma + 2.0);
  } else {
    *hue = 60.0 * ((r - g) / chroma + 4.0);
  }
}

}  // namespace

std::vector<SlotPaint> MeasureSlotPaint(
    const std::vector<std::array<Vec2, 4>>& slots,
    const std::vector<uint8_t>& rgb, const std::vector<uint8_t>& observed,
    const RasterFrame& frame, const SlotPaintConfig& config) {
  std::vector<SlotPaint> out(slots.size());
  for (size_t s = 0; s < slots.size(); ++s) {
    const std::vector<Vec2> inner = InsetConvex(
        std::vector<Vec2>(slots[s].begin(), slots[s].end()), config.inset);
    if (inner.size() < 3) continue;
    Vec2 lo = inner[0];
    Vec2 hi = inner[0];
    for (const Vec2& p : inner) {
      lo = lo.cwiseMin(p);
      hi = hi.cwiseMax(p);
    }
    const int c0 = std::max(0, static_cast<int>(std::floor(
                                   (lo.x() - frame.x0) / frame.res)));
    const int c1 = std::min(frame.width - 1,
                            static_cast<int>(std::floor((hi.x() - frame.x0) /
                                                        frame.res)));
    const int r0 = std::max(0, static_cast<int>(std::floor(
                                   (lo.y() - frame.y0) / frame.res)));
    const int r1 = std::min(frame.height - 1,
                            static_cast<int>(std::floor((hi.y() - frame.y0) /
                                                        frame.res)));
    SlotPaint& paint = out[s];
    int in_band = 0;
    for (int r = r0; r <= r1; ++r) {
      for (int c = c0; c <= c1; ++c) {
        const Vec2 center(frame.x0 + (c + 0.5) * frame.res,
                          frame.y0 + (r + 0.5) * frame.res);
        if (!PointInPolygon(inner, center)) continue;
        ++paint.cells;
        const size_t cell = static_cast<size_t>(r) * frame.width + c;
        if (!observed[cell]) continue;
        ++paint.observed;
        const Vec3 colour(rgb[3 * cell], rgb[3 * cell + 1], rgb[3 * cell + 2]);
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
