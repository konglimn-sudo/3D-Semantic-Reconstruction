// Floor colour inside parking slots on the orthophoto. Many garages paint
// the charging (new-energy) slots green; the share of observed cells whose
// colour is in a hue band with enough saturation tells them apart from
// grey concrete. The slot is inset first so that its painted lines do not
// count; cells the texture did not observe (hidden under a parked car) do
// not count either.

#ifndef GARAGE_RECON_GR_SEMANTICS_SLOT_PAINT_H_
#define GARAGE_RECON_GR_SEMANTICS_SLOT_PAINT_H_

#include <array>
#include <cstdint>
#include <vector>

#include "gr/core/types.h"
#include "gr/semantics/markings.h"
#include "gr/semantics/slots.h"

namespace gr {

struct SlotPaintConfig {
  double inset = 0.25;          // m inside the slot outline.
  double min_saturation = 0.25;  // (max - min) / max of RGB.
  double hue_min = 75.0;         // deg; green: 75 .. 165.
  double hue_max = 165.0;
};

struct SlotPaint {
  int cells = 0;             // Inside the inset outline.
  int observed = 0;          // ... with texture.
  double in_band = 0.0;      // Share of the observed cells in the band.
  Vec3 mean_rgb = Vec3::Zero();  // Of the observed cells.
};

// `rgb` is height x width x kRgbChannels and `observed` height x width
// (non-zero where the texture saw the floor), both row-major in `frame`.
std::vector<SlotPaint> MeasureSlotPaint(
    const std::vector<std::array<Vec2, kParkingSlotCorners>>& slots,
    const std::vector<uint8_t>& rgb, const std::vector<uint8_t>& observed,
    const RasterFrame& frame, const SlotPaintConfig& config = {});

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_SLOT_PAINT_H_
