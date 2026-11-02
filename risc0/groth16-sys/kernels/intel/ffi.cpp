// C entry points of the Intel Groth16 prover, mirroring the CUDA prover's (../cuda/ffi.cu):
// preload the proving key onto the GPU, prove, and release the cached prover.
#include "groth16_intel.hpp"

#include <cstring>
#include <memory>
#include <mutex>

struct SetupParams {
  const char* pcoeffs_path;
  const char* fres_path;
  const char* srs_path;
};

struct ProveParams {
  const char* public_path;
  const char* proof_path;
  const uint8_t* witness; // canonical little-endian field elements, one per witness variable
};

namespace {

// Guards everything below. Held for a whole preload, prove or release, so release can never
// free the prover under a proof in flight.
std::mutex g_mutex;
std::unique_ptr<sycl::queue> g_queue;
std::unique_ptr<r0g16::Prover> g_prover;
std::string g_async_error; // first asynchronous SYCL error since the last check

sycl::queue& queue() {
  if (!g_queue) {
    // Without a handler, an asynchronous error (e.g. a kernel fault) terminates the process.
    auto handler = [](sycl::exception_list errors) {
      for (const std::exception_ptr& e : errors) {
        try {
          std::rethrow_exception(e);
        } catch (const std::exception& ex) {
          if (g_async_error.empty())
            g_async_error = ex.what();
        }
      }
    };
    g_queue = std::make_unique<sycl::queue>(
        sycl::gpu_selector_v, handler, sycl::property::queue::in_order());
  }
  return *g_queue;
}

// Report (and clear) any asynchronous error the queue has raised.
void check_async() {
  if (g_queue)
    g_queue->wait_and_throw();
  if (!g_async_error.empty()) {
    std::string e = std::move(g_async_error);
    g_async_error.clear();
    throw std::runtime_error("groth16-intel: " + e);
  }
}

r0g16::Prover& prover(const SetupParams* p) {
  if (!g_prover) {
    sycl::queue& q = queue();
    g_prover = std::make_unique<r0g16::Prover>(q, p->srs_path, p->pcoeffs_path, p->fres_path);
    check_async();
  }
  return *g_prover;
}

const char* error_string(const char* what) {
  return strdup(what ? what : "unknown error");
}

} // namespace

extern "C" const char* risc0_groth16_intel_preload(const SetupParams* setup) {
  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    prover(setup);
  } catch (const std::exception& e) {
    return error_string(e.what());
  } catch (...) {
    return error_string("risc0_groth16_intel_preload: unknown exception");
  }
  return nullptr;
}

extern "C" const char* risc0_groth16_intel_prove(const SetupParams* setup,
                                                 const ProveParams* params) {
  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    r0g16::Prover& pr = prover(setup);
    r0g16::Proof proof = pr.prove(reinterpret_cast<const r0g16::Fr*>(params->witness));
    check_async();
    r0g16::write_proof_file(params->proof_path, proof);
  } catch (const std::exception& e) {
    return error_string(e.what());
  } catch (...) {
    return error_string("risc0_groth16_intel_prove: unknown exception");
  }
  return nullptr;
}

// Free the cached prover (proving key, coefficients and scratch: several GB of GPU memory).
// The next prove rebuilds it.
extern "C" const char* risc0_groth16_intel_release() {
  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_prover.reset();
    if (g_queue)
      g_queue->wait();
  } catch (const std::exception& e) {
    return error_string(e.what());
  } catch (...) {
    return error_string("risc0_groth16_intel_release: unknown exception");
  }
  return nullptr;
}
