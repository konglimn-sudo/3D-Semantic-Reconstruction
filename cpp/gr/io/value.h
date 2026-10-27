// A parsed document (YAML configuration, rig and meta files; JSON products)
// and its readers / writer.
//
// The YAML reader covers what the project writes and configures: block
// mappings and sequences (PyYAML's `key:\n- item` at the key's indentation
// and the compact `- - 1.0` / `- key: value` forms included), single-line
// flow collections, quoted and plain scalars and comments. Anchors, tags and
// block scalars (| >) are rejected. Plain scalars are typed as PyYAML's
// safe_load types them: null / ~, booleans (true, yes, on, ...), numbers
// (.inf, .nan included), otherwise strings.

#ifndef GARAGE_RECON_GR_IO_VALUE_H_
#define GARAGE_RECON_GR_IO_VALUE_H_

#include <string>
#include <utility>
#include <vector>

namespace gr {

class Value {
 public:
  enum class Kind { kNull, kBool, kNumber, kString, kList, kMap };

  Value() = default;
  static Value Bool(bool value);
  static Value Number(double value);
  static Value Integer(long long value);  // Written without a fraction.
  static Value String(std::string value);
  static Value List();
  static Value Map();

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::kNull; }
  bool is_number() const { return kind_ == Kind::kNumber; }
  bool is_integer() const { return kind_ == Kind::kNumber && integer_; }
  bool is_string() const { return kind_ == Kind::kString; }
  bool is_list() const { return kind_ == Kind::kList; }
  bool is_map() const { return kind_ == Kind::kMap; }

  // Typed access; throws std::runtime_error on a kind mismatch (`AsInt` also
  // on a fractional number).
  bool AsBool() const;
  double AsDouble() const;
  int AsInt() const;
  const std::string& AsString() const;
  std::vector<double> AsDoubles() const;  // A list of numbers.

  // Lists and maps: number of items / members; 0 otherwise.
  size_t size() const;
  const std::vector<Value>& items() const;  // Lists.
  const std::vector<std::pair<std::string, Value>>& members() const;  // Maps.

  // Member of a map (nullptr if absent or not a map).
  const Value* Find(const std::string& key) const;
  Value* Find(const std::string& key);
  // Member of a map; a null value if absent.
  const Value& operator[](const std::string& key) const;
  const Value& operator[](size_t index) const;  // Lists; throws if out.

  // Building: `Set` replaces an existing member (keeps its position).
  Value& Set(const std::string& key, Value value);
  Value& Append(Value value);

 private:
  Kind kind_ = Kind::kNull;
  bool bool_ = false;
  double number_ = 0.0;
  bool integer_ = false;
  std::string string_;
  std::vector<Value> items_;
  std::vector<std::pair<std::string, Value>> members_;
};

// Parsers; throw std::runtime_error naming the line of a syntax error.
Value ParseYaml(const std::string& text);
Value ParseJson(const std::string& text);

// File readers (`path` in the message when the file cannot be read).
Value ReadYamlFile(const std::string& path);
Value ReadJsonFile(const std::string& path);

// JSON text: `indent` spaces per level (negative: one line). Numbers are
// written in their shortest round-trip form, integral ones without a
// fraction; non-ASCII text is kept as UTF-8.
std::string ToJson(const Value& value, int indent = 1);

// Block-style YAML that PyYAML (YAML 1.1) reads back to the same value:
// keys and strings double-quoted, lists of scalars in flow style, floats
// always with a '.' (1.0e-05, not 1e-05, which YAML 1.1 reads as text).
std::string ToYaml(const Value& value);

// `over` merged into `base`: maps member by member, anything else replaced
// (config.deep_merge).
Value DeepMerge(const Value& base, const Value& over);

}  // namespace gr

#endif  // GARAGE_RECON_GR_IO_VALUE_H_
