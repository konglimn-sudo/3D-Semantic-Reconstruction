#include "gr/testing/test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gr::testing {
namespace {

struct TestCase {
  std::string name;
  void (*body)();
};

struct Failure {
  std::string file;
  int line = 0;
  std::string message;
};

struct RunState {
  std::vector<TestCase> tests;
  std::vector<Failure> failures;  // Of the running test.
};

// Never destroyed: no non-trivial static destructors.
RunState& State() {
  static RunState* const state = new RunState();
  return *state;
}

// Escapes the data of a GitHub Actions workflow command.
std::string EscapeCommandData(const std::string& text) {
  std::string out;
  for (const char c : text) {
    if (c == '%') {
      out += "%25";
    } else if (c == '\r') {
      out += "%0D";
    } else if (c == '\n') {
      out += "%0A";
    } else {
      out += c;
    }
  }
  return out;
}

// Under GitHub Actions each failure also becomes an error annotation, which
// the checks API shows without access to the job log. The file is made
// relative to the checkout.
void AnnotateFailure(const std::string& test, const Failure& failure) {
  const char* actions = std::getenv("GITHUB_ACTIONS");
  if (actions == nullptr || std::strcmp(actions, "true") != 0) return;
  std::string file = failure.file;
  const char* workspace = std::getenv("GITHUB_WORKSPACE");
  if (workspace != nullptr) {
    const std::string prefix = std::string(workspace) + "/";
    if (file.compare(0, prefix.size(), prefix) == 0) {
      file = file.substr(prefix.size());
    }
  }
  std::printf("::error file=%s,line=%d,title=%s::%s\n", file.c_str(),
              failure.line, test.c_str(),
              EscapeCommandData(failure.message).c_str());
}

}  // namespace

bool RegisterTest(const char* suite, const char* name, void (*body)()) {
  State().tests.push_back({std::string(suite) + "." + name, body});
  return true;
}

void ReportFailure(const char* file, int line, const std::string& message) {
  State().failures.push_back({file, line, message});
}

int RunAllTests() {
  RunState& state = State();
  int failed = 0;
  for (const TestCase& test : state.tests) {
    state.failures.clear();
    test.body();
    if (state.failures.empty()) {
      std::printf("[ OK ] %s\n", test.name.c_str());
      continue;
    }
    ++failed;
    std::printf("[FAIL] %s\n", test.name.c_str());
    for (const Failure& failure : state.failures) {
      std::printf("       %s:%d: %s\n", failure.file.c_str(), failure.line,
                  failure.message.c_str());
      AnnotateFailure(test.name, failure);
    }
  }
  std::printf("%zu/%zu tests passed\n", state.tests.size() - failed,
              state.tests.size());
  return failed == 0 ? 0 : 1;
}

}  // namespace gr::testing
