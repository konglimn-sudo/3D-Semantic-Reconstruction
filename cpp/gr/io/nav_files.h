// Files of the navigation map (semantics/nav/ of a reconstruction) that the
// vehicle needs, and the products of localizing a trip:
//
//   localization_map.npz     xyz (N, 3) float32: the permanent structure
//   graph.json               route graph: nodes, edges (kind, level, points)
//   goals.json               slot goals (slot_id, level, corners)
//   level_<k>_layers.npz     raster layers of a level; floor_z and grid
//                            [x0, y0, res, width, height, level]
//
// Results (as nav/localize.py writes them): poses.tum (map <- vehicle of the
// tracked sweeps), sweeps.csv and localization.json.

#ifndef GARAGE_RECON_GR_IO_NAV_FILES_H_
#define GARAGE_RECON_GR_IO_NAV_FILES_H_

#include <string>
#include <vector>

#include "gr/core/types.h"
#include "gr/io/value.h"
#include "gr/nav/position.h"
#include "gr/nav/trip_localizer.h"

namespace gr {

Points ReadLocalizationMap(const std::string& nav_dir);
std::vector<NavEdge> ReadNavEdges(const std::string& graph_json);
std::vector<NavSlot> ReadNavSlots(const std::string& goals_json);
// Floor heights of every level_<k>_layers.npz, in file name order.
std::vector<FloorHeights> ReadFloorHeights(const std::string& nav_dir);

const char* StatusName(LocalizationStatus status);

// The summary and position as the keys of localization.json (numbers
// rounded as localize.py rounds them).
Value SummaryValue(const TripLocalizationSummary& summary);
Value PositionValue(const NavPosition& position);

// poses.tum, sweeps.csv and localization.json (`report`) in `directory`
// (created if needed).
void WriteLocalizationResult(const std::string& directory,
                             const std::vector<SweepLocalization>& sweeps,
                             const Value& report);

}  // namespace gr

#endif  // GARAGE_RECON_GR_IO_NAV_FILES_H_
