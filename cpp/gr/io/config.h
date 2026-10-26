// The YAML configuration (python/garage_recon/default_config.yaml deep-merged
// with user files) and its sections as C++ config structs. As
// core_config.py's `fill`: struct fields carry the YAML key names, keys a
// struct does not have and null values (the C++ default stays) are skipped.

#ifndef GARAGE_RECON_GR_IO_CONFIG_H_
#define GARAGE_RECON_GR_IO_CONFIG_H_

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "gr/io/value.h"
#include "gr/nav/trip_localizer.h"

namespace gr {

// The files deep-merged in order (later ones override).
Value LoadConfig(const std::vector<std::string>& paths);

// Field name -> member of a config struct.
template <typename T>
class FieldTable {
 public:
  template <typename M>
  FieldTable& Add(const std::string& name, M T::*member) {
    setters_.emplace_back(name, [member](const Value& v, T* out) {
      Assign(v, &(out->*member));
    });
    return *this;
  }

  // Copies the matching non-null members of `section` (a map, or null);
  // returns how many. Throws std::runtime_error naming a mistyped key.
  int Fill(const Value& section, T* out) const {
    int copied = 0;
    if (section.is_null()) return 0;
    for (const auto& [key, value] : section.members()) {
      if (value.is_null()) continue;
      for (const auto& [name, set] : setters_) {
        if (name != key) continue;
        try {
          set(value, out);
        } catch (const std::runtime_error& e) {
          throw std::runtime_error(key + ": " + e.what());
        }
        ++copied;
      }
    }
    return copied;
  }

 private:
  static void Assign(const Value& v, double* out) { *out = v.AsDouble(); }
  static void Assign(const Value& v, int* out) { *out = v.AsInt(); }
  static void Assign(const Value& v, bool* out) { *out = v.AsBool(); }
  static void Assign(const Value& v, std::vector<double>* out) {
    *out = v.AsDoubles();
  }

  std::vector<std::pair<std::string, std::function<void(const Value&, T*)>>>
      setters_;
};

// Sections of the merged configuration (core_config.odometry_config,
// keyframe_config, localize.localizer_config).
OdometryConfig OdometryConfigFrom(const Value& config);
KeyframeConfig KeyframeConfigFrom(const Value& config);
LocalizerConfig LocalizerConfigFrom(const Value& config);
TripLocalizerConfig TripLocalizerConfigFrom(const Value& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_IO_CONFIG_H_
