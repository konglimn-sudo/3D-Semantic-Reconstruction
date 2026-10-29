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
  Value out;
  out.kind_ = Kind::kBool;
  out.bool_ = value;
  return out;
}

Value Value::Number(double value) {
  Value out;
  out.kind_ = Kind::kNumber;
  out.number_ = value;
  return out;
}

Value Value::Integer(int64_t value) {
  Value out = Number(static_cast<double>(value));
  out.integer_ = true;
  return out;
}

Value Value::String(std::string value) {
  Value out;
  out.kind_ = Kind::kString;
  out.string_ = std::move(value);
  return out;
}

Value Value::List() {
  Value out;
  out.kind_ = Kind::kList;
  return out;
}

Value Value::Map() {
  Value out;
  out.kind_ = Kind::kMap;
  return out;
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

// UTF-8 (RFC 3629): code points below these limits take one, two or three
// bytes, the others four. The lead byte of a multi-byte sequence marks its
// length; the continuation bytes carry 6 bits each.
constexpr unsigned kOneByteLimit = 0x80;
constexpr unsigned kTwoByteLimit = 0x800;
constexpr unsigned kThreeByteLimit = 0x10000;
constexpr unsigned kTwoByteLead = 0xC0;
constexpr unsigned kThreeByteLead = 0xE0;
constexpr unsigned kFourByteLead = 0xF0;
constexpr unsigned kContinuationMarker = 0x80;
constexpr unsigned kContinuationMask = 0x3F;
constexpr int kContinuationBits = 6;

// Escapes: "\xXX" and "\uXXXX".
constexpr int kHexDigitBits = 4;
constexpr unsigned kFirstHexLetterValue = 10;  // 'a' and 'A'.
constexpr int kByteEscapeDigits = 2;
constexpr int kUnicodeEscapeDigits = 4;
constexpr size_t kEscapePrefixLength = 2;  // A backslash and the letter.
constexpr size_t kUnicodeEscapeLength =
    kEscapePrefixLength + kUnicodeEscapeDigits;
// UTF-16 surrogate pairs (JSON's encoding of characters beyond the BMP):
// the high surrogate carries the upper 10 bits of the offset from
// kFirstSupplementary, the low one the lower 10.
constexpr unsigned kHighSurrogateFirst = 0xD800;
constexpr unsigned kLowSurrogateFirst = 0xDC00;
constexpr unsigned kLowSurrogateEnd = 0xE000;
constexpr unsigned kFirstSupplementary = 0x10000;
constexpr int kSurrogateBits = 10;
// JSON escapes the control characters below this one.
constexpr unsigned kFirstUnescapedCharacter = 0x20;

constexpr int kDecimalBase = 10;

[[noreturn]] void Fail(int line, const std::string& message) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + message);
}

std::string Trim(const std::string& text) {
  const size_t begin = text.find_first_not_of(" \t");
  if (begin == std::string::npos) return "";
  const size_t end = text.find_last_not_of(" \t");
  return text.substr(begin, end - begin + 1);
}

// The continuation byte with bits [6 * position, 6 * position + 6) of `code`.
char ContinuationByte(unsigned code, int position) {
  return static_cast<char>(
      kContinuationMarker |
      ((code >> (kContinuationBits * position)) & kContinuationMask));
}

// The lead byte of a sequence with `continuations` continuation bytes.
char LeadByte(unsigned code, unsigned lead, int continuations) {
  return static_cast<char>(lead |
                           (code >> (kContinuationBits * continuations)));
}

void AppendUtf8(unsigned code, std::string* out) {
  if (code < kOneByteLimit) {
    out->push_back(static_cast<char>(code));
  } else if (code < kTwoByteLimit) {
    out->push_back(LeadByte(code, kTwoByteLead, 1));
    out->push_back(ContinuationByte(code, 0));
  } else if (code < kThreeByteLimit) {
    out->push_back(LeadByte(code, kThreeByteLead, 2));
    out->push_back(ContinuationByte(code, 1));
    out->push_back(ContinuationByte(code, 0));
  } else {
    out->push_back(LeadByte(code, kFourByteLead, 3));
    out->push_back(ContinuationByte(code, 2));
    out->push_back(ContinuationByte(code, 1));
    out->push_back(ContinuationByte(code, 0));
  }
}

unsigned ParseHex(const std::string& text, size_t at, int digits, int line) {
  if (at + digits > text.size()) Fail(line, "truncated escape");
  unsigned code = 0;
  for (int k = 0; k < digits; ++k) {
    const char digit = text[at + k];
    code <<= kHexDigitBits;
    if (digit >= '0' && digit <= '9') {
      code |= digit - '0';
    } else if (digit >= 'a' && digit <= 'f') {
      code |= digit - 'a' + kFirstHexLetterValue;
    } else if (digit >= 'A' && digit <= 'F') {
      code |= digit - 'A' + kFirstHexLetterValue;
    } else {
      Fail(line, "bad hex digit in an escape");
    }
  }
  return code;
}

// A double-quoted string (JSON or YAML) starting at text[position] == '"';
// leaves `position` after the closing quote.
std::string ParseDoubleQuoted(const std::string& text, size_t& position,
                              int line) {
  std::string out;
  ++position;
  while (true) {
    if (position >= text.size()) Fail(line, "unterminated string");
    const char character = text[position++];
    if (character == '"') return out;
    if (character != '\\') {
      out.push_back(character);
      continue;
    }
    if (position >= text.size()) Fail(line, "unterminated string");
    const char escape = text[position++];
    switch (escape) {
      case '"':
      case '\\':
      case '/':
        out.push_back(escape);
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
        AppendUtf8(ParseHex(text, position, kByteEscapeDigits, line), &out);
        position += kByteEscapeDigits;
        break;
      case 'u': {
        unsigned code = ParseHex(text, position, kUnicodeEscapeDigits, line);
        position += kUnicodeEscapeDigits;
        // A surrogate pair: this high surrogate and the low one after it.
        if (code >= kHighSurrogateFirst && code < kLowSurrogateFirst &&
            position + kUnicodeEscapeLength <= text.size() &&
            text[position] == '\\' && text[position + 1] == 'u') {
          const unsigned low = ParseHex(text, position + kEscapePrefixLength,
                                        kUnicodeEscapeDigits, line);
          if (low >= kLowSurrogateFirst && low < kLowSurrogateEnd) {
            code = kFirstSupplementary +
                   ((code - kHighSurrogateFirst) << kSurrogateBits) +
                   (low - kLowSurrogateFirst);
            position += kUnicodeEscapeLength;
          }
        }
        AppendUtf8(code, &out);
        break;
      }
      default:
        Fail(line, std::string("unknown escape \\") + escape);
    }
  }
}

// A single-quoted YAML string ('' is a quote); leaves `position` after it.
std::string ParseSingleQuoted(const std::string& text, size_t& position,
                              int line) {
  std::string out;
  ++position;
  while (true) {
    if (position >= text.size()) Fail(line, "unterminated string");
    const char character = text[position++];
    if (character != '\'') {
      out.push_back(character);
    } else if (position < text.size() && text[position] == '\'') {
      out.push_back('\'');
      ++position;
    } else {
      return out;
    }
  }
}

// ------------------------------------------------------------------- YAML

// A plain scalar typed as PyYAML's safe_load (YAML 1.1 resolver).
Value TypePlain(const std::string& text) {
  static const std::regex kNull("~|null|Null|NULL|");
  static const std::regex kTrue("yes|Yes|YES|true|True|TRUE|on|On|ON");
  static const std::regex kFalse("no|No|NO|false|False|FALSE|off|Off|OFF");
  static const std::regex kInt("[-+]?(0|[1-9][0-9_]*)");
  static const std::regex kFloat(
      "[-+]?([0-9][0-9_]*)?\\.[0-9_]*([eE][-+][0-9]+)?");
  static const std::regex kInf("[-+]?\\.(inf|Inf|INF)");
  static const std::regex kNan("\\.(nan|NaN|NAN)");
  if (std::regex_match(text, kNull)) return Value();
  if (std::regex_match(text, kTrue)) return Value::Bool(true);
  if (std::regex_match(text, kFalse)) return Value::Bool(false);
  std::string digits;
  for (char character : text) {
    if (character != '_') digits.push_back(character);
  }
  if (std::regex_match(text, kInt)) {
    return Value::Integer(std::strtoll(digits.c_str(), nullptr, kDecimalBase));
  }
  if (std::regex_match(text, kFloat) && text != "." && text != "+." &&
      text != "-.") {
    return Value::Number(std::strtod(digits.c_str(), nullptr));
  }
  if (std::regex_match(text, kInf)) {
    const double inf = std::numeric_limits<double>::infinity();
    return Value::Number(text[0] == '-' ? -inf : inf);
  }
  if (std::regex_match(text, kNan)) {
    return Value::Number(std::numeric_limits<double>::quiet_NaN());
  }
  return Value::String(text);
}

// Whether a quote at text[position] opens a quoted scalar: only at the start of
// a scalar (line start, after "key:", "- ", "[", "{" or ",").
bool OpensQuote(const std::string& text, size_t position) {
  size_t start = position;
  while (start > 0 && (text[start - 1] == ' ' || text[start - 1] == '\t')) {
    --start;
  }
  if (start == 0) return true;
  const char previous = text[start - 1];
  return previous == ':' || previous == '-' || previous == '[' ||
         previous == '{' || previous == ',' || previous == '?';
}

// Index just after the quoted scalar starting at text[position].
size_t SkipQuoted(const std::string& text, size_t position) {
  const char quote = text[position];
  for (size_t scan = position + 1; scan < text.size(); ++scan) {
    if (quote == '"' && text[scan] == '\\') {
      ++scan;
    } else if (text[scan] == quote) {
      if (quote == '\'' && scan + 1 < text.size() && text[scan + 1] == '\'') {
        ++scan;
      } else {
        return scan + 1;
      }
    }
  }
  return text.size();
}

std::string StripComment(const std::string& text) {
  for (size_t position = 0; position < text.size(); ++position) {
    const char character = text[position];
    if ((character == '"' || character == '\'') && OpensQuote(text, position)) {
      position = SkipQuoted(text, position) - 1;
    } else if (character == '#' &&
               (position == 0 || text[position - 1] == ' ' ||
                text[position - 1] == '\t')) {
      return text.substr(0, position);
    }
  }
  return text;
}

// Position of the ':' that ends a mapping key (followed by a blank or the
// end), outside quotes and flow collections; npos if none.
size_t FindMappingColon(const std::string& text) {
  int depth = 0;
  for (size_t position = 0; position < text.size(); ++position) {
    const char character = text[position];
    if ((character == '"' || character == '\'') && OpensQuote(text, position)) {
      position = SkipQuoted(text, position) - 1;
    } else if (character == '[' || character == '{') {
      ++depth;
    } else if (character == ']' || character == '}') {
      --depth;
    } else if (character == ':' && depth == 0 &&
               (position + 1 == text.size() || text[position + 1] == ' ')) {
      return position;
    }
  }
  return std::string::npos;
}

bool IsSequenceItem(const std::string& text) {
  return text == "-" || (text.size() > 1 && text[0] == '-' && text[1] == ' ');
}

// Bracket balance of a flow collection, outside quotes.
int FlowDepth(const std::string& text) {
  int depth = 0;
  for (size_t position = 0; position < text.size(); ++position) {
    const char character = text[position];
    if ((character == '"' || character == '\'') && OpensQuote(text, position)) {
      position = SkipQuoted(text, position) - 1;
    } else if (character == '[' || character == '{') {
      ++depth;
    } else if (character == ']' || character == '}') {
      --depth;
    }
  }
  return depth;
}

void SkipBlanks(const std::string& text, size_t& position) {
  while (position < text.size() &&
         (text[position] == ' ' || text[position] == '\t')) {
    ++position;
  }
}

Value ParseFlowNode(const std::string& text, size_t& position, int line,
                    bool key);

Value ParseFlowCollection(const std::string& text, size_t& position, int line) {
  const char open = text[position++];
  const char close = open == '[' ? ']' : '}';
  Value out = open == '[' ? Value::List() : Value::Map();
  while (true) {
    SkipBlanks(text, position);
    if (position >= text.size()) Fail(line, "unterminated flow collection");
    if (text[position] == close) {
      ++position;
      return out;
    }
    if (open == '[') {
      out.Append(ParseFlowNode(text, position, line, false));
    } else {
      const Value key = ParseFlowNode(text, position, line, true);
      SkipBlanks(text, position);
      if (position >= text.size() || text[position] != ':') {
        Fail(line, "expected ':' in a map");
      }
      ++position;
      SkipBlanks(text, position);
      Value value;
      if (position < text.size() && text[position] != ',' &&
          text[position] != '}') {
        value = ParseFlowNode(text, position, line, false);
      }
      out.Set(key.is_string() ? key.AsString() : ToJson(key, kJsonOneLine),
              value);
    }
    SkipBlanks(text, position);
    if (position < text.size() && text[position] == ',') {
      ++position;
    } else if (position >= text.size() || text[position] != close) {
      Fail(line, std::string("expected ',' or '") + close + "'");
    }
  }
}

Value ParseFlowNode(const std::string& text, size_t& position, int line,
                    bool key) {
  SkipBlanks(text, position);
  if (position >= text.size()) Fail(line, "missing value");
  const char character = text[position];
  if (character == '[' || character == '{') {
    return ParseFlowCollection(text, position, line);
  }
  if (character == '"') {
    return Value::String(ParseDoubleQuoted(text, position, line));
  }
  if (character == '\'') {
    return Value::String(ParseSingleQuoted(text, position, line));
  }
  const size_t begin = position;
  while (position < text.size() && text[position] != ',' &&
         text[position] != ']' && text[position] != '}' &&
         !(key && text[position] == ':')) {
    ++position;
  }
  return TypePlain(Trim(text.substr(begin, position - begin)));
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
      if (IsSequenceItem(rest) || FindMappingColon(rest) != std::string::npos) {
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
      const std::string key =
          ParseKey(Trim(line.text.substr(0, colon)), line.number);
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
    size_t position = 0;
    if (text[0] == '"') return ParseDoubleQuoted(text, position, line);
    if (text[0] == '\'') return ParseSingleQuoted(text, position, line);
    return text;
  }

  // A value on the rest of a line; flow collections may continue on the
  // following (more indented) lines.
  Value ParseInline(std::string text, int number) {
    const char first = text[0];
    if (first == '&' || first == '*' || first == '!' || first == '|' ||
        first == '>') {
      Fail(number, "anchors, tags and block scalars are not supported");
    }
    if (first == '[' || first == '{') {
      while (FlowDepth(text) > 0 && pos_ < lines_.size()) {
        text += " " + lines_[pos_++].text;
      }
    }
    if (first == '[' || first == '{' || first == '"' || first == '\'') {
      size_t position = 0;
      Value out = ParseFlowNode(text, position, number, false);
      SkipBlanks(text, position);
      if (position != text.size()) Fail(number, "text after a value");
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
  explicit JsonParser(const std::string& text) : text_(text) {}

  Value Parse() {
    Value out = ParseValue();
    Skip();
    if (position_ != text_.size()) Fail(line_, "text after the document");
    return out;
  }

 private:
  void Skip() {
    while (position_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[position_]))) {
      if (text_[position_] == '\n') ++line_;
      ++position_;
    }
  }

  bool Consume(const char* word) {
    const size_t length = std::char_traits<char>::length(word);
    if (text_.compare(position_, length, word) != 0) return false;
    position_ += length;
    return true;
  }

  Value ParseValue() {
    Skip();
    if (position_ >= text_.size()) Fail(line_, "unexpected end");
    const char character = text_[position_];
    if (character == '{') {
      ++position_;
      Value out = Value::Map();
      Skip();
      if (position_ < text_.size() && text_[position_] == '}') {
        ++position_;
        return out;
      }
      while (true) {
        Skip();
        if (position_ >= text_.size() || text_[position_] != '"') {
          Fail(line_, "expected a key");
        }
        const std::string key = ParseDoubleQuoted(text_, position_, line_);
        Skip();
        if (position_ >= text_.size() || text_[position_] != ':') {
          Fail(line_, "expected ':'");
        }
        ++position_;
        out.Set(key, ParseValue());
        Skip();
        if (position_ < text_.size() && text_[position_] == ',') {
          ++position_;
          continue;
        }
        if (position_ < text_.size() && text_[position_] == '}') {
          ++position_;
          return out;
        }
        Fail(line_, "expected ',' or '}'");
      }
    }
    if (character == '[') {
      ++position_;
      Value out = Value::List();
      Skip();
      if (position_ < text_.size() && text_[position_] == ']') {
        ++position_;
        return out;
      }
      while (true) {
        out.Append(ParseValue());
        Skip();
        if (position_ < text_.size() && text_[position_] == ',') {
          ++position_;
          continue;
        }
        if (position_ < text_.size() && text_[position_] == ']') {
          ++position_;
          return out;
        }
        Fail(line_, "expected ',' or ']'");
      }
    }
    if (character == '"') {
      return Value::String(ParseDoubleQuoted(text_, position_, line_));
    }
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
    const size_t begin = position_;
    bool integral = true;
    while (position_ < text_.size() &&
           (std::isdigit(static_cast<unsigned char>(text_[position_])) ||
            text_[position_] == '-' || text_[position_] == '+' ||
            text_[position_] == '.' || text_[position_] == 'e' ||
            text_[position_] == 'E')) {
      if (text_[position_] == '.' || text_[position_] == 'e' ||
          text_[position_] == 'E') {
        integral = false;
      }
      ++position_;
    }
    if (position_ == begin) {
      Fail(line_, std::string("unexpected '") + character + "'");
    }
    const std::string number = text_.substr(begin, position_ - begin);
    char* end = nullptr;
    if (integral) {
      const int64_t parsed = std::strtoll(number.c_str(), &end, kDecimalBase);
      if (*end == '\0') return Value::Integer(parsed);
    }
    const double parsed = std::strtod(number.c_str(), &end);
    if (*end != '\0') Fail(line_, "bad number " + number);
    return Value::Number(parsed);
  }

  const std::string& text_;
  size_t position_ = 0;
  int line_ = 1;
};

// Python's repr of a float: the shortest digits that round-trip, fixed
// notation for exponents in [kMinFixedExponent, kMaxFixedExponent), else
// d.ddde+XX.
constexpr int kMinFixedExponent = -4;
constexpr int kMaxFixedExponent = 16;
// Room for "-d.<16 digits>e-XXX" and for "e-XXX".
constexpr size_t kFloatTextBuffer = 40;
constexpr size_t kExponentTextBuffer = 16;

std::string FloatRepr(double value) {
  if (std::isnan(value)) return "NaN";
  if (std::isinf(value)) return value > 0 ? "Infinity" : "-Infinity";
  if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";
  char buffer[kFloatTextBuffer];
  int precision = 1;
  for (; precision <= std::numeric_limits<double>::max_digits10; ++precision) {
    std::snprintf(buffer, sizeof(buffer), "%.*e", precision - 1, value);
    if (std::strtod(buffer, nullptr) == value) break;
  }
  // buffer: [-]d[.ddd]e[+-]XX
  std::string text = buffer;
  const bool negative = text[0] == '-';
  if (negative) text.erase(0, 1);
  const size_t exponent_mark = text.find('e');
  const int exponent = std::atoi(text.c_str() + exponent_mark + 1);
  std::string digits;
  for (size_t k = 0; k < exponent_mark; ++k) {
    if (text[k] != '.') digits.push_back(text[k]);
  }
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  std::string out = negative ? "-" : "";
  if (exponent >= kMinFixedExponent && exponent < kMaxFixedExponent) {
    if (exponent < 0) {
      out +=
          "0." + std::string(static_cast<size_t>(-exponent - 1), '0') + digits;
    } else if (static_cast<int>(digits.size()) <= exponent + 1) {
      out +=
          digits +
          std::string(static_cast<size_t>(exponent + 1) - digits.size(), '0') +
          ".0";
    } else {
      out += digits.substr(0, static_cast<size_t>(exponent + 1)) + "." +
             digits.substr(static_cast<size_t>(exponent + 1));
    }
    return out;
  }
  out += digits.substr(0, 1);
  if (digits.size() > 1) out += "." + digits.substr(1);
  char exponent_text[kExponentTextBuffer];
  std::snprintf(exponent_text, sizeof(exponent_text), "e%c%02d",
                exponent < 0 ? '-' : '+', std::abs(exponent));
  return out + exponent_text;
}

void WriteString(const std::string& text, std::string* out) {
  out->push_back('"');
  for (char character : text) {
    switch (character) {
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
        if (static_cast<unsigned char>(character) < kFirstUnescapedCharacter) {
          char buffer[kUnicodeEscapeLength + 1];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
          *out += buffer;
        } else {
          out->push_back(character);
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
        *out += std::to_string(static_cast<int64_t>(value.AsDouble()));
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
  } catch (const std::runtime_error& error) {
    throw std::runtime_error(path + ": " + error.what());
  }
}

Value ReadJsonFile(const std::string& path) {
  try {
    return ParseJson(ReadText(path));
  } catch (const std::runtime_error& error) {
    throw std::runtime_error(path + ": " + error.what());
  }
}

std::string ToJson(const Value& value, int indent) {
  std::string out;
  WriteJson(value, indent, 0, &out);
  return out;
}

namespace {

// Spaces per level of ToYaml's block style.
constexpr int kYamlIndent = 2;

// A scalar as PyYAML (YAML 1.1) reads it back: strings double-quoted (JSON
// escapes are YAML escapes), floats with a '.' before any exponent.
std::string YamlScalar(const Value& value) {
  if (value.is_number() && !value.is_integer()) {
    std::string text = ToJson(value, kJsonOneLine);
    const size_t exponent_mark = text.find('e');
    if (exponent_mark != std::string::npos &&
        text.find('.') == std::string::npos) {
      text.insert(exponent_mark, ".0");
    }
    if (text == "NaN") return ".nan";
    if (text == "Infinity") return ".inf";
    if (text == "-Infinity") return "-.inf";
    return text;
  }
  return ToJson(value, kJsonOneLine);
}

bool IsScalar(const Value& value) {
  return !value.is_list() && !value.is_map();
}

void WriteYaml(const Value& value, int depth, std::string* out) {
  const std::string pad(static_cast<size_t>(kYamlIndent * depth), ' ');
  if (value.is_map()) {
    for (const auto& [key, item] : value.members()) {
      *out += pad + ToJson(Value::String(key), kJsonOneLine) + ":";
      if (IsScalar(item) || item.size() == 0) {
        *out += " " +
                (item.is_list()  ? std::string("[]")
                 : item.is_map() ? std::string("{}")
                                 : YamlScalar(item)) +
                "\n";
      } else if (item.is_list() && std::all_of(item.items().begin(),
                                               item.items().end(), IsScalar)) {
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
                (item.is_list()  ? std::string("[]")
                 : item.is_map() ? std::string("{}")
                                 : YamlScalar(item)) +
                "\n";
      } else {
        // The item's block one level deeper, its first line after "- ".
        std::string block;
        WriteYaml(item, depth + 1, &block);
        block.replace(0, static_cast<size_t>(kYamlIndent * (depth + 1)),
                      pad + "- ");
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
    return (value.is_list()  ? std::string("[]")
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
