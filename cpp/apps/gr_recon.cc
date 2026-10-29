// gr_recon: the geometric stages of the offline reconstruction without
// Python (`garage_recon recon --stages odometry graph trajectory map`).
//
//   gr_recon --trips <trip> [<trip> ...] --work <dir>
//            [--config a.yaml [--config b.yaml ...]]
//            [--stages odometry graph trajectory map]
//            [--loop-pairs pairs.npy]
//
// The configuration files are deep-merged in order; pass the package's
// default_config.yaml first. Without --config the work directory's
// config.used.yaml is used (resuming a run). --loop-pairs: (N, 2) keyframe
// pairs proposed by image retrieval or text (Python's appearance / text
// loops), verified in the graph stage. The outputs are those of
// recon/pipeline.py (gr/io/recon_files.h), so the Python stages that need
// cameras or learned models (calibration, bev, texture, gs, semantics)
// continue from them.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gr/io/config.h"
#include "gr/io/npy.h"
#include "gr/pipeline/recon_pipeline.h"

namespace {

namespace fs = std::filesystem;

// Exit status of a command-line error (argparse's).
constexpr int kUsageExitCode = 2;
// --loop-pairs holds (N, kPairSize) keyframe indices.
constexpr int64_t kPairSize = 2;

struct Args {
  std::vector<std::string> trips;
  std::string work;
  std::vector<std::string> configs;
  std::vector<std::string> stages;
  std::string loop_pairs;
};

[[noreturn]] void Usage(const std::string& error) {
  std::fprintf(stderr,
               "%s\nusage: gr_recon --trips DIR [DIR ...] --work DIR "
               "[--config FILE ...] [--stages odometry graph trajectory "
               "map] [--loop-pairs NPY]\n",
               error.c_str());
  std::exit(kUsageExitCode);
}

Args Parse(int argc, char** argv) {
  Args args;
  std::vector<std::string>* list = nullptr;  // Flag taking several values.
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--trips" || flag == "--stages") {
      list = flag == "--trips" ? &args.trips : &args.stages;
      continue;
    }
    if (flag.rfind("--", 0) != 0) {
      if (list == nullptr) Usage("unexpected argument " + flag);
      list->push_back(flag);
      continue;
    }
    list = nullptr;
    if (index + 1 >= argc) Usage("missing value of " + flag);
    const std::string value = argv[++index];
    if (flag == "--work") {
      args.work = value;
    } else if (flag == "--config") {
      args.configs.push_back(value);
    } else if (flag == "--loop-pairs") {
      args.loop_pairs = value;
    } else {
      Usage("unknown flag " + flag);
    }
  }
  if (args.trips.empty() || args.work.empty()) {
    Usage("--trips and --work are required");
  }
  if (args.stages.empty()) args.stages = gr::ReconPipeline::StageNames();
  if (args.configs.empty()) {
    const fs::path used = fs::path(args.work) / "config.used.yaml";
    if (!fs::exists(used)) Usage("no --config and no " + used.string());
    args.configs.push_back(used.string());
  }
  return args;
}

void Log(const std::string& line) {
  std::printf("%s\n", line.c_str());
  std::fflush(stdout);
}

int Run(const Args& args) {
  using namespace gr;  // NOLINT(build/namespaces)
  const auto start = std::chrono::steady_clock::now();
  ReconPipeline pipeline(args.trips, args.work, LoadConfig(args.configs), Log);
  if (!args.loop_pairs.empty()) {
    const NpyArray array = ReadNpy(args.loop_pairs);
    if (array.size() != 0 &&
        (array.shape.size() != 2 || array.shape[1] != kPairSize)) {
      throw std::runtime_error(args.loop_pairs + ": (N, 2) pairs expected");
    }
    const std::vector<int64_t> values = array.ToInts();
    std::vector<std::pair<int, int>> pairs;
    for (size_t first = 0; first + 1 < values.size();
         first += static_cast<size_t>(kPairSize)) {
      pairs.emplace_back(static_cast<int>(values[first]),
                         static_cast<int>(values[first + 1]));
    }
    pipeline.set_loop_pairs(std::move(pairs));
  }
  pipeline.Run(args.stages);
  Log(StringPrintf("done in %.1fs",
                   std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count()));
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = Parse(argc, argv);
  try {
    return Run(args);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "gr_recon: %s\n", error.what());
    return EXIT_FAILURE;
  }
}
