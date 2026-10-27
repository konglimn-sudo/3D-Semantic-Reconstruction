#include "gr/io/value.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace gr {

Value Value::Bool(bool value) {
  Value v;
  v.kind_ = Kind::kBool;
  v.bool_ = value;
  return v;
}

Value Value::Number(double value) {
  Value v;
  v.kind_ = Kind::kNumber;
  v.number_ = value;
  return v;
}

Value Value::Integer(long long value) {
  Value v = Number(static_cast<double>(value));
  v.integer_ = true;
  return v;
}

Value Value::String(std::string value) {
  Value v;
  v.kind_ = Kind::kString;
  v.string_ = std::move(value);
  return v;
}

Value Value::List() {
  Value v;
  v.kind_ = Kind::kList;
  return v;
}

Value Value::Map() {
  Value v;
  v.kind_ = Kind::kMap;
  return v;
}

namespace {

const char* KindName(Value::Kind kind) {
  switch (kind) {
    case Value::Kind::kNull:
      return "null";
    case Value::Kind::kBool:
      return "a boolean";
    case Value::Kind::kNumber:
      return "a number";
    case Value::Kind::kString:
      return "a string";
    case Value::Kind::kList:
      return "a list";
    case Value::Kind::kMap:
      return "a map";
  }
  return "?";
}

[[noreturn]] void Mismatch(Value::Kind kind, const char* expected) {
  throw std::runtime_error(std::string("expected ") + expected + ", got " +
                           KindName(kind));
}

}  // namespace

bool Value::AsBool() const {
  if (kind_ != Kind::kBool) Mismatch(kind_, "a boolean");
  return bool_;
}

double Value::AsDouble() const {
  if (kind_ != Kind::kNumber) Mismatch(kind_, "a number");
  return number_;
}

int Value::AsInt() const {
  if (kind_ != Kind::kNumber) Mismatch(kind_, "an integer");
  if (number_ != std::floor(number_) ||
      std::abs(number_) > std::numeric_limits<int>::max()) {
    throw std::runtime_error("expected an integer, got " +
                             std::to_string(number_));
  }
  return static_cast<int>(number_);
}

const std::string& Value::AsString() const {
  if (kind_ != Kind::kString) Mismatch(kind_, "a string");
  return string_;
}

std::vector<double> Value::AsDoubles() const {
  if (kind_ != Kind::kList) Mismatch(kind_, "a list");
  std::vector<double> out;
  for (const Value& item : items_) out.push_back(item.AsDouble());
  return out;
}

size_t Value::size() const {
  if (kind_ == Kind::kList) return items_.size();
  if (kind_ == Kind::kMap) return members_.size();
  return 0;
}

const std::vector<Value>& Value::items() const {
  if (kind_ != Kind::kList) Mismatch(kind_, "a list");
  return items_;
}

const std::vector<std::pair<std::string, Value>>& Value::members() const {
  if (kind_ != Kind::kMap) Mismatch(kind_, "a map");
  return members_;
}

const Value* Value::Find(const std::string& key) const {
  if (kind_ != Kind::kMap) return nullptr;
  for (const auto& [name, value] : members_) {
    if (name == key) return &value;
  }
  return nullptr;
}

Value* Value::Find(const std::string& key) {
  return const_cast<Value*>(static_cast<const Value*>(this)->Find(key));
}

const Value& Value::operator[](const std::string& key) const {
  static const Value kNull;
  const Value* found = Find(key);
  return found ? *found : kNull;
}

const Value& Value::operator[](size_t index) const {
  if (kind_ != Kind::kList) Mismatch(kind_, "a list");
  if (index >= items_.size()) {
    throw std::runtime_error("index " + std::to_string(index) +
                             " beyond a list of " +
                             std::to_string(items_.size()));
  }
  return items_[index];
}

Value& Value::Set(const std::string& key, Value value) {
  if (kind_ == Kind::kNull) kind_ = Kind::kMap;
  if (kind_ != Kind::kMap) Mismatch(kind_, "a map");
  for (auto& [name, existing] : members_) {
    if (name == key) {
      existing = std::move(value);
      return existing;
    }
  }
  members_.emplace_back(key, std::move(value));
  return members_.back().second;
}

Value& Value::Append(Value value) {
  if (kind_ == Kind::kNull) kind_ = Kind::kList;
  if (kind_ != Kind::kList) Mismatch(kind_, "a list");
  items_.push_back(std::move(value));
  return items_.back();
}

namespace {

[[noreturn]] void Fail(int line, const std::string& message) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + message);
}

std::string Trim(const std::string& s) {
  const size_t begin = s.find_first_not_of(" \t");
  if (begin == std::string::npos) return "";
  const size_t end = s.find_last_not_of(" \t");
  return s.substr(begin, end - begin + 1);
}

void AppendUtf8(unsigned code, std::string* out) {
  if (code < 0x80) {
    out->push_back(static_cast<char>(code));
  } else if (code < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (code >> 6)));
    out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else if (code < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (code >> 12)));
    out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (code >> 18)));
    out->push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
  }
}

unsigned ParseHex(const std::string& s, size_t at, int digits, int line) {
  if (at + digits > s.size()) Fail(line, "truncated escape");
  unsigned code = 0;
  for (int k = 0; k < digits; ++k) {
    const char c = s[at + k];
    code <<= 4;
    if (c >= '0' && c <= '9') {
      code |= c - '0';
    } else if (c >= 'a' && c <= 'f') {
      code |= c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
      code |= c - 'A' + 10;
    } else {
      Fail(line, "bad hex digit in an escape");
    }
  }
  return code;
}

// A double-quoted string (JSON or YAML) starting at s[i] == '"'; leaves `i`
// after the closing quote.
std::string ParseDoubleQuoted(const std::string& s, size_t& i, int line) {
  std::string out;
  ++i;
  while (true) {
    if (i >= s.size()) Fail(line, "unterminated string");
    const char c = s[i++];
    if (c == '"') return out;
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (i >= s.size()) Fail(line, "unterminated string");
    const char e = s[i++];
    switch (e) {
      case '"':
      case '\\':
      case '/':
        out.push_back(e);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case '0':
        out.push_back('\0');
        break;
      case 'x':
        AppendUtf8(ParseHex(s, i, 2, line), &out);
        i += 2;
        break;
      case 'u': {
        unsigned code = ParseHex(s, i, 4, line);
        i += 4;
        // A surrogate pair (JSON's encoding of characters beyond the BMP).
        if (code >= 0xD800 && code < 0xDC00 && i + 6 <= s.size() &&
            s[i] == '\\' && s[i + 1] == 'u') {
          const unsigned low = ParseHex(s, i + 2, 4, line);
          if (low >= 0xDC00 && low < 0xE000) {
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            i += 6;
          }
        }
        AppendUtf8(code, &out);
        break;
      }
      default:
        Fail(line, std::string("unknown escape \\") + e);
    }
  }
}

// A single-quoted YAML string ('' is a quote); leaves `i` after it.
std::string ParseSingleQuoted(const std::string& s, size_t& i, int line) {
  std::string out;
  ++i;
  while (true) {
    if (i >= s.size()) Fail(line, "unterminated string");
    const char c = s[i++];
    if (c != '\'') {
      out.push_back(c);
    } else if (i < s.size() && s[i] == '\'') {
      out.push_back('\'');
      ++i;
    } else {
      return out;
    }
  }
}

// ------------------------------------------------------------------- YAML

// A plain scalar typed as PyYAML's safe_load (YAML 1.1 resolver).
Value TypePlain(const std::string& s) {
  static const std::regex kNull("~|null|Null|NULL|");
  static const std::regex kTrue("yes|Yes|YES|true|True|TRUE|on|On|ON");
  static const std::regex kFalse("no|No|NO|false|False|FALSE|off|Off|OFF");
  static const std::regex kInt("[-+]?(0|[1-9][0-9_]*)");
  static const std::regex kFloat(
      "[-+]?([0-9][0-9_]*)?\\.[0-9_]*([eE][-+][0-9]+)?");
  static const std::regex kInf("[-+]?\\.(inf|Inf|INF)");
  static const std::regex kNan("\\.(nan|NaN|NAN)");
  if (std::regex_match(s, kNull)) return Value();
  if (std::regex_match(s, kTrue)) return Value::Bool(true);
  if (std::regex_match(s, kFalse)) return Value::Bool(false);
  std::string digits;
  for (char c : s) {
    if (c != '_') digits.push_back(c);
  }
  if (std::regex_match(s, kInt)) {
    return Value::Integer(std::strtoll(digits.c_str(), nullptr, 10));
  }
  if (std::regex_match(s, kFloat) && s != "." && s != "+." && s != "-.") {
    return Value::Number(std::strtod(digits.c_str(), nullptr));
  }
  if (std::regex_match(s, kInf)) {
    const double inf = std::numeric_limits<double>::infinity();
    return Value::Number(s[0] == '-' ? -inf : inf);
  }
  if (std::regex_match(s, kNan)) {
    return Value::Number(std::numeric_limits<double>::quiet_NaN());
  }
  return Value::String(s);
}

// Whether a quote at s[i] opens a quoted scalar: only at the start of a
// scalar (line start, after "key:", "- ", "[", "{" or ",").
bool OpensQuote(const std::string& s, size_t i) {
  size_t k = i;
  while (k > 0 && (s[k - 1] == ' ' || s[k - 1] == '\t')) --k;
  if (k == 0) return true;
  const char p = s[k - 1];
  return p == ':' || p == '-' || p == '[' || p == '{' || p == ',' ||
         p == '?';
}

// Index just after the quoted scalar starting at s[i].
size_t SkipQuoted(const std::string& s, size_t i) {
  const char quote = s[i];
  for (size_t k = i + 1; k < s.size(); ++k) {
    if (quote == '"' && s[k] == '\\') {
      ++k;
    } else if (s[k] == quote) {
      if (quote == '\'' && k + 1 < s.size() && s[k + 1] == '\'') {
        ++k;
      } else {
        return k + 1;
      }
    }
  }
  return s.size();
}

std::string StripComment(const std::string& s) {
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if ((c == '"' || c == '\'') && OpensQuote(s, i)) {
      i = SkipQuoted(s, i) - 1;
    } else if (c == '#' && (i == 0 || s[i - 1] == ' ' || s[i - 1] == '\t')) {
      return s.substr(0, i);
    }
  }
  return s;
}

// Position of the ':' that ends a mapping key (followed by a blank or the
// end), outside quotes and flow collections; npos if none.
size_t FindMappingColon(const std::string& s) {
  int depth = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if ((c == '"' || c == '\'') && OpensQuote(s, i)) {
      i = SkipQuoted(s, i) - 1;
    } else if (c == '[' || c == '{') {
      ++depth;
    } else if (c == ']' || c == '}') {
      --depth;
    } else if (c == ':' && depth == 0 &&
               (i + 1 == s.size() || s[i + 1] == ' ')) {
      return i;
    }
  }
  return std::string::npos;
}

bool IsSequenceItem(const std::string& text) {
  return text == "-" || (text.size() > 1 && text[0] == '-' && text[1] == ' ');
}

// Bracket balance of a flow collection, outside quotes.
int FlowDepth(const std::string& s) {
  int depth = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if ((c == '"' || c == '\'') && OpensQuote(s, i)) {
      i = SkipQuoted(s, i) - 1;
    } else if (c == '[' || c == '{') {
      ++depth;
    } else if (c == ']' || c == '}') {
      --depth;
    }
  }
  return depth;
}

void SkipBlanks(const std::string& s, size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
}

Value ParseFlowNode(const std::string& s, size_t& i, int line, bool key);

Value ParseFlowCollection(const std::string& s, size_t& i, int line) {
  const char open = s[i++];
  const char close = open == '[' ? ']' : '}';
  Value out = open == '[' ? Value::List() : Value::Map();
  while (true) {
    SkipBlanks(s, i);
    if (i >= s.size()) Fail(line, "unterminated flow collection");
    if (s[i] == close) {
      ++i;
      return out;
    }
    if (open == '[') {
      out.Append(ParseFlowNode(s, i, line, false));
    } else {
      const Value key = ParseFlowNode(s, i, line, true);
      SkipBlanks(s, i);
      if (i >= s.size() || s[i] != ':') Fail(line, "expected ':' in a map");
      ++i;
      SkipBlanks(s, i);
      Value value;
      if (i < s.size() && s[i] != ',' && s[i] != '}') {
        value = ParseFlowNode(s, i, line, false);
      }
      out.Set(key.is_string() ? key.AsString() : ToJson(key, -1), value);
    }
    SkipBlanks(s, i);
    if (i < s.size() && s[i] == ',') {
      ++i;
    } else if (i >= s.size() || s[i] != close) {
      Fail(line, std::string("expected ',' or '") + close + "'");
    }
  }
}

Value ParseFlowNode(const std::string& s, size_t& i, int line, bool key) {
  SkipBlanks(s, i);
  if (i >= s.size()) Fail(line, "missing value");
  const char c = s[i];
  if (c == '[' || c == '{') return ParseFlowCollection(s, i, line);
  if (c == '"') return Value::String(ParseDoubleQuoted(s, i, line));
  if (c == '\'') return Value::String(ParseSingleQuoted(s, i, line));
  const size_t begin = i;
  while (i < s.size() && s[i] != ',' && s[i] != ']' && s[i] != '}' &&
         !(key && s[i] == ':')) {
    ++i;
  }
  return TypePlain(Trim(s.substr(begin, i - begin)));
}

struct Line {
  int number = 0;
  int indent = 0;
  std::string text;  // Without indentation, comment and trailing blanks.
};

class YamlParser {
 public:
  explicit YamlParser(std::vector<Line> lines) : lines_(std::move(lines)) {}

  Value Parse() {
    if (lines_.empty()) return Value();
    Value out = ParseBlock(lines_[0].indent);
    if (pos_ < lines_.size()) {
      Fail(lines_[pos_].number, "unexpected indentation");
    }
    return out;
  }

 private:
  Value ParseBlock(int indent) {
    const Line& line = lines_[pos_];
    if (IsSequenceItem(line.text)) return ParseSequence(indent);
    if (FindMappingColon(line.text) != std::string::npos) {
      return ParseMapping(indent);
    }
    ++pos_;
    return ParseInline(line.text, line.number);
  }

  Value ParseSequence(int indent) {
    Value out = Value::List();
    while (pos_ < lines_.size() && lines_[pos_].indent == indent &&
           IsSequenceItem(lines_[pos_].text)) {
      Line& line = lines_[pos_];
      const size_t offset = line.text.find_first_not_of(' ', 1);
      if (offset == std::string::npos) {
        ++pos_;
        if (pos_ < lines_.size() && lines_[pos_].indent > indent) {
          out.Append(ParseBlock(lines_[pos_].indent));
        } else {
          out.Append(Value());
        }
        continue;
      }
      const std::string rest = line.text.substr(offset);
      if (IsSequenceItem(rest) ||
          FindMappingColon(rest) != std::string::npos) {
        // Compact nested item: continues at the column of `rest`.
        line.indent = indent + static_cast<int>(offset);
        line.text = rest;
        out.Append(ParseBlock(line.indent));
      } else {
        const int number = line.number;
        ++pos_;
        out.Append(ParseInline(rest, number));
      }
    }
    return out;
  }

  Value ParseMapping(int indent) {
    Value out = Value::Map();
    while (pos_ < lines_.size() && lines_[pos_].indent == indent) {
      const Line line = lines_[pos_];
      if (IsSequenceItem(line.text)) Fail(line.number, "item in a mapping");
      const size_t colon = FindMappingColon(line.text);
      if (colon == std::string::npos) {
        Fail(line.number, "expected 'key: value'");
      }
      const std::string key = ParseKey(Trim(line.text.substr(0, colon)),
                                       line.number);
      const std::string rest = Trim(line.text.substr(colon + 1));
      ++pos_;
      Value value;
      if (!rest.empty()) {
        value = ParseInline(rest, line.number);
      } else if (pos_ < lines_.size() && lines_[pos_].indent > indent) {
        value = ParseBlock(lines_[pos_].indent);
      } else if (pos_ < lines_.size() && lines_[pos_].indent == indent &&
                 IsSequenceItem(lines_[pos_].text)) {
        value = ParseSequence(indent);  // PyYAML's `key:\n- item`.
      }
      out.Set(key, std::move(value));
      if (pos_ < lines_.size() && lines_[pos_].indent > indent) {
        Fail(lines_[pos_].number, "unexpected indentation");
      }
    }
    return out;
  }

  static std::string ParseKey(const std::string& text, int line) {
    if (text.empty()) Fail(line, "empty key");
    size_t i = 0;
    if (text[0] == '"') return ParseDoubleQuoted(text, i, line);
    if (text[0] == '\'') return ParseSingleQuoted(text, i, line);
    return text;
  }

  // A value on the rest of a line; flow collections may continue on the
  // following (more indented) lines.
  Value ParseInline(std::string text, int number) {
    const char c = text[0];
    if (c == '&' || c == '*' || c == '!' || c == '|' || c == '>') {
      Fail(number, "anchors, tags and block scalars are not supported");
    }
    if (c == '[' || c == '{') {
      while (FlowDepth(text) > 0 && pos_ < lines_.size()) {
        text += " " + lines_[pos_++].text;
      }
    }
    if (c == '[' || c == '{' || c == '"' || c == '\'') {
      size_t i = 0;
      Value out = ParseFlowNode(text, i, number, false);
      SkipBlanks(text, i);
      if (i != text.size()) Fail(number, "text after a value");
      return out;
    }
    return TypePlain(text);
  }

  std::vector<Line> lines_;
  size_t pos_ = 0;
};

// ------------------------------------------------------------------- JSON

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : s_(text) {}

  Value Parse() {
    Value out = ParseValue();
    Skip();
    if (i_ != s_.size()) Fail(line_, "text after the document");
    return out;
  }

 private:
  void Skip() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) {
      if (s_[i_] == '\n') ++line_;
      ++i_;
    }
  }

  bool Consume(const char* word) {
    const size_t n = std::char_traits<char>::length(word);
    if (s_.compare(i_, n, word) != 0) return false;
    i_ += n;
    return true;
  }

  Value ParseValue() {
    Skip();
    if (i_ >= s_.size()) Fail(line_, "unexpected end");
    const char c = s_[i_];
    if (c == '{') {
      ++i_;
      Value out = Value::Map();
      Skip();
      if (i_ < s_.size() && s_[i_] == '}') {
        ++i_;
        return out;
      }
      while (true) {
        Skip();
        if (i_ >= s_.size() || s_[i_] != '"') Fail(line_, "expected a key");
        const std::string key = ParseDoubleQuoted(s_, i_, line_);
        Skip();
        if (i_ >= s_.size() || s_[i_] != ':') Fail(line_, "expected ':'");
        ++i_;
        out.Set(key, ParseValue());
        Skip();
        if (i_ < s_.size() && s_[i_] == ',') {
          ++i_;
          continue;
        }
        if (i_ < s_.size() && s_[i_] == '}') {
          ++i_;
          return out;
        }
        Fail(line_, "expected ',' or '}'");
      }
    }
    if (c == '[') {
      ++i_;
      Value out = Value::List();
      Skip();
      if (i_ < s_.size() && s_[i_] == ']') {
        ++i_;
        return out;
      }
      while (true) {
        out.Append(ParseValue());
        Skip();
        if (i_ < s_.size() && s_[i_] == ',') {
          ++i_;
          continue;
        }
        if (i_ < s_.size() && s_[i_] == ']') {
          ++i_;
          return out;
        }
        Fail(line_, "expected ',' or ']'");
      }
    }
    if (c == '"') return Value::String(ParseDoubleQuoted(s_, i_, line_));
    if (Consume("true")) return Value::Bool(true);
    if (Consume("false")) return Value::Bool(false);
    if (Consume("null")) return Value();
    // Python's json writes these for non-finite floats.
    if (Consume("NaN")) {
      return Value::Number(std::numeric_limits<double>::quiet_NaN());
    }
    if (Consume("Infinity")) {
      return Value::Number(std::numeric_limits<double>::infinity());
    }
    if (Consume("-Infinity")) {
      return Value::Number(-std::numeric_limits<double>::infinity());
    }
    const size_t begin = i_;
    bool integral = true;
    while (i_ < s_.size() &&
           (std::isdigit(static_cast<unsigned char>(s_[i_])) ||
            s_[i_] == '-' || s_[i_] == '+' || s_[i_] == '.' ||
            s_[i_] == 'e' || s_[i_] == 'E')) {
      if (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E') integral = false;
      ++i_;
    }
    if (i_ == begin) Fail(line_, std::string("unexpected '") + c + "'");
    const std::string number = s_.substr(begin, i_ - begin);
    char* end = nullptr;
    if (integral) {
      const long long v = std::strtoll(number.c_str(), &end, 10);
      if (*end == '\0') return Value::Integer(v);
    }
    const double v = std::strtod(number.c_str(), &end);
    if (*end != '\0') Fail(line_, "bad number " + number);
    return Value::Number(v);
  }

  const std::string& s_;
  size_t i_ = 0;
  int line_ = 1;
};

// Python's repr of a float: the shortest digits that round-trip, fixed
// notation for exponents in [-4, 16), else d.ddde+XX.
std::string FloatRepr(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
  char buffer[40];
  int precision = 1;
  for (; precision <= 17; ++precision) {
    std::snprintf(buffer, sizeof(buffer), "%.*e", precision - 1, v);
    if (std::strtod(buffer, nullptr) == v) break;
  }
  // buffer: [-]d[.ddd]e[+-]XX
  std::string text = buffer;
  const bool negative = text[0] == '-';
  if (negative) text.erase(0, 1);
  const size_t e = text.find('e');
  const int exponent = std::atoi(text.c_str() + e + 1);
  std::string digits;
  for (size_t k = 0; k < e; ++k) {
    if (text[k] != '.') digits.push_back(text[k]);
  }
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  std::string out = negative ? "-" : "";
  if (exponent >= -4 && exponent < 16) {
    if (exponent < 0) {
      out += "0." + std::string(static_cast<size_t>(-exponent - 1), '0') +
             digits;
    } else if (static_cast<int>(digits.size()) <= exponent + 1) {
      out += digits +
             std::string(static_cast<size_t>(exponent + 1) - digits.size(),
                         '0') +
             ".0";
    } else {
      out += digits.substr(0, static_cast<size_t>(exponent + 1)) + "." +
             digits.substr(static_cast<size_t>(exponent + 1));
    }
    return out;
  }
  out += digits.substr(0, 1);
  if (digits.size() > 1) out += "." + digits.substr(1);
  char exp[16];
  std::snprintf(exp, sizeof(exp), "e%c%02d", exponent < 0 ? '-' : '+',
                std::abs(exponent));
  return out + exp;
}

void WriteString(const std::string& s, std::string* out) {
  out->push_back('"');
  for (char c : s) {
    switch (c) {
      case '"':
        *out += "\\\"";
        break;
      case '\\':
        *out += "\\\\";
        break;
      case '\n':
        *out += "\\n";
        break;
      case '\r':
        *out += "\\r";
        break;
      case '\t':
        *out += "\\t";
        break;
      case '\b':
        *out += "\\b";
        break;
      case '\f':
        *out += "\\f";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
          *out += buffer;
        } else {
          out->push_back(c);
        }
    }
  }
  out->push_back('"');
}

void WriteJson(const Value& value, int indent, int depth, std::string* out) {
  const auto newline = [&](int level) {
    if (indent < 0) return;
    out->push_back('\n');
    out->append(static_cast<size_t>(indent * level), ' ');
  };
  const char* separator = indent < 0 ? ", " : ",";
  switch (value.kind()) {
    case Value::Kind::kNull:
      *out += "null";
      return;
    case Value::Kind::kBool:
      *out += value.AsBool() ? "true" : "false";
      return;
    case Value::Kind::kNumber:
      if (value.is_integer()) {
        *out += std::to_string(static_cast<long long>(value.AsDouble()));
      } else {
        *out += FloatRepr(value.AsDouble());
      }
      return;
    case Value::Kind::kString:
      WriteString(value.AsString(), out);
      return;
    case Value::Kind::kList: {
      if (value.size() == 0) {
        *out += "[]";
        return;
      }
      out->push_back('[');
      bool first = true;
      for (const Value& item : value.items()) {
        if (!first) *out += separator;
        first = false;
        newline(depth + 1);
        WriteJson(item, indent, depth + 1, out);
      }
      newline(depth);
      out->push_back(']');
      return;
    }
    case Value::Kind::kMap: {
      if (value.size() == 0) {
        *out += "{}";
        return;
      }
      out->push_back('{');
      bool first = true;
      for (const auto& [key, item] : value.members()) {
        if (!first) *out += separator;
        first = false;
        newline(depth + 1);
        WriteString(key, out);
        *out += ": ";
        WriteJson(item, indent, depth + 1, out);
      }
      newline(depth);
      out->push_back('}');
      return;
    }
  }
}

std::string ReadText(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

}  // namespace

Value ParseYaml(const std::string& text) {
  std::vector<Line> lines;
  std::istringstream in(text);
  std::string raw;
  int number = 0;
  while (std::getline(in, raw)) {
    ++number;
    if (!raw.empty() && raw.back() == '\r') raw.pop_back();
    const size_t indent = raw.find_first_not_of(' ');
    if (indent == std::string::npos) continue;
    if (raw[indent] == '\t') Fail(number, "tab in the indentation");
    std::string content = raw.substr(indent);
    if (content.rfind("%", 0) == 0) continue;  // Directives.
    content = StripComment(content);
    const size_t end = content.find_last_not_of(" \t");
    if (end == std::string::npos) continue;
    content.resize(end + 1);
    if (indent == 0 && (content == "---" || content == "...")) continue;
    lines.push_back({number, static_cast<int>(indent), content});
  }
  return YamlParser(std::move(lines)).Parse();
}

Value ParseJson(const std::string& text) { return JsonParser(text).Parse(); }

Value ReadYamlFile(const std::string& path) {
  try {
    return ParseYaml(ReadText(path));
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

Value ReadJsonFile(const std::string& path) {
  try {
    return ParseJson(ReadText(path));
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

std::string ToJson(const Value& value, int indent) {
  std::string out;
  WriteJson(value, indent, 0, &out);
  return out;
}

namespace {

// A scalar as PyYAML (YAML 1.1) reads it back: strings double-quoted (JSON
// escapes are YAML escapes), floats with a '.' before any exponent.
std::string YamlScalar(const Value& value) {
  if (value.is_number() && !value.is_integer()) {
    std::string text = ToJson(value, -1);
    const size_t e = text.find('e');
    if (e != std::string::npos && text.find('.') == std::string::npos) {
      text.insert(e, ".0");
    }
    if (text == "NaN") return ".nan";
    if (text == "Infinity") return ".inf";
    if (text == "-Infinity") return "-.inf";
    return text;
  }
  return ToJson(value, -1);
}

bool IsScalar(const Value& value) {
  return !value.is_list() && !value.is_map();
}

void WriteYaml(const Value& value, int depth, std::string* out) {
  const std::string pad(static_cast<size_t>(2 * depth), ' ');
  if (value.is_map()) {
    for (const auto& [key, item] : value.members()) {
      *out += pad + ToJson(Value::String(key), -1) + ":";
      if (IsScalar(item) || item.size() == 0) {
        *out += " " + (item.is_list() ? std::string("[]")
                       : item.is_map() ? std::string("{}")
                                       : YamlScalar(item)) +
                "\n";
      } else if (item.is_list() &&
                 std::all_of(item.items().begin(), item.items().end(),
                             IsScalar)) {
        std::string flow = " [";
        for (size_t k = 0; k < item.size(); ++k) {
          if (k) flow += ", ";
          flow += YamlScalar(item[k]);
        }
        *out += flow + "]\n";
      } else {
        *out += "\n";
        WriteYaml(item, depth + 1, out);
      }
    }
    return;
  }
  if (value.is_list()) {
    for (const Value& item : value.items()) {
      if (IsScalar(item) || item.size() == 0) {
        *out += pad + "- " +
                (item.is_list() ? std::string("[]")
                 : item.is_map() ? std::string("{}")
                                 : YamlScalar(item)) +
                "\n";
      } else {
        // The item's block one level deeper, its first line after "- ".
        std::string block;
        WriteYaml(item, depth + 1, &block);
        block.replace(0, static_cast<size_t>(2 * (depth + 1)), pad + "- ");
        *out += block;
      }
    }
    return;
  }
  *out += pad + YamlScalar(value) + "\n";
}

}  // namespace

std::string ToYaml(const Value& value) {
  std::string out;
  if (IsScalar(value) || value.size() == 0) {
    return (value.is_list() ? std::string("[]")
            : value.is_map() ? std::string("{}")
                             : YamlScalar(value)) +
           "\n";
  }
  WriteYaml(value, 0, &out);
  return out;
}

Value DeepMerge(const Value& base, const Value& over) {
  if (!base.is_map() || !over.is_map()) return over;
  Value out = base;
  for (const auto& [key, value] : over.members()) {
    const Value* existing = out.Find(key);
    if (existing != nullptr && existing->is_map() && value.is_map()) {
      out.Set(key, DeepMerge(*existing, value));
    } else {
      out.Set(key, value);
    }
  }
  return out;
}

}  // namespace gr
