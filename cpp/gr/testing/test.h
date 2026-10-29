// Minimal dependency-free test harness with a GoogleTest-like API.
//
//   GR_TEST(SuiteName, TestName) { ... }
//
// GR_EXPECT_* records a failure and continues; GR_ASSERT_* records a failure
// and returns from the test. The harness does not use exceptions.

#ifndef GARAGE_RECON_GR_TESTING_TEST_H_
#define GARAGE_RECON_GR_TESTING_TEST_H_

#include <cmath>
#include <sstream>
#include <string>

namespace gr::testing {

// Registers a test; used by GR_TEST. Always returns true.
bool RegisterTest(const char* suite, const char* name, void (*body)());
// Records a failed expectation of the running test.
void ReportFailure(const char* file, int line, const std::string& message);
// Runs every registered test and returns the process exit code.
int RunAllTests();

template <typename Left, typename Right>
std::string CompareMessage(const char* expression, const Left& left,
                           const Right& right) {
  std::ostringstream message;
  message << "expected " << expression << ", got " << left << " vs " << right;
  return message.str();
}

}  // namespace gr::testing

#define GR_TEST(suite, name)                                                \
  static void GrTest_##suite##_##name();                                    \
  [[maybe_unused]] static const bool gr_registered_##suite##_##name =       \
      ::gr::testing::RegisterTest(#suite, #name, &GrTest_##suite##_##name); \
  static void GrTest_##suite##_##name()

#define GR_EXPECT_TRUE(condition)                                 \
  do {                                                            \
    if (!(condition)) {                                           \
      ::gr::testing::ReportFailure(__FILE__, __LINE__,            \
                                   "expected true: " #condition); \
    }                                                             \
  } while (false)

#define GR_EXPECT_FALSE(condition)                                 \
  do {                                                             \
    if (condition) {                                               \
      ::gr::testing::ReportFailure(__FILE__, __LINE__,             \
                                   "expected false: " #condition); \
    }                                                              \
  } while (false)

#define GR_ASSERT_TRUE(condition)                                    \
  do {                                                               \
    if (!(condition)) {                                              \
      ::gr::testing::ReportFailure(__FILE__, __LINE__,               \
                                   "assertion failed: " #condition); \
      return;                                                        \
    }                                                                \
  } while (false)

#define GR_INTERNAL_EXPECT_OP(a, b, op)                                  \
  do {                                                                   \
    const auto gr_a = (a); /* Copies: (a) may refer into a temporary. */ \
    const auto gr_b = (b);                                               \
    if (!(gr_a op gr_b)) {                                               \
      ::gr::testing::ReportFailure(                                      \
          __FILE__, __LINE__,                                            \
          ::gr::testing::CompareMessage(#a " " #op " " #b, gr_a, gr_b)); \
    }                                                                    \
  } while (false)

#define GR_EXPECT_EQ(a, b) GR_INTERNAL_EXPECT_OP(a, b, ==)
#define GR_EXPECT_LT(a, b) GR_INTERNAL_EXPECT_OP(a, b, <)
#define GR_EXPECT_LE(a, b) GR_INTERNAL_EXPECT_OP(a, b, <=)
#define GR_EXPECT_GT(a, b) GR_INTERNAL_EXPECT_OP(a, b, >)
#define GR_EXPECT_GE(a, b) GR_INTERNAL_EXPECT_OP(a, b, >=)

#define GR_EXPECT_NEAR(a, b, tolerance)                                     \
  do {                                                                      \
    const double gr_a = (a);                                                \
    const double gr_b = (b);                                                \
    if (!(std::abs(gr_a - gr_b) <= (tolerance))) {                          \
      ::gr::testing::ReportFailure(                                         \
          __FILE__, __LINE__,                                               \
          ::gr::testing::CompareMessage("|" #a " - " #b "| <= " #tolerance, \
                                        gr_a, gr_b));                       \
    }                                                                       \
  } while (false)

#endif  // GARAGE_RECON_GR_TESTING_TEST_H_
