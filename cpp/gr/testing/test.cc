#include "gr/testing/test.h"

#include <cstdio>
#include <vector>

namespace gr::testing {
namespace {

struct TestCase {
  std::string name;
  void (*body)();
};

struct RunState {
  std::vector<TestCase> tests;
  std::vector<std::string> failures;  // Of the running test.
};

// Never destroyed: no non-trivial static destructors.
RunState& State() {
  static RunState* const state = new RunState();
  return *state;
}

}  // namespace

bool RegisterTest(const char* suite, const char* name, void (*body)()) {
  State().tests.push_back({std::string(suite) + "." + name, body});
  return true;
}

void ReportFailure(const char* file, int line, const std::string& message) {
  State().failures.push_back(std::string(file) + ":" + std::to_string(line) +
                             ": " + message);
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
    for (const std::string& failure : state.failures) {
      std::printf("       %s\n", failure.c_str());
    }
  }
  std::printf("%zu/%zu tests passed\n", state.tests.size() - failed,
              state.tests.size());
  return failed == 0 ? 0 : 1;
}

}  // namespace gr::testing
