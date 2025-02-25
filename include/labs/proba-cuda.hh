#ifndef PROBA_CUDA_HH
#define PROBA_CUDA_HH

#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

#include "lab.hh"

namespace utcn::ip {
class ProbaCUDA : public Lab {
 private:
  static inline std::map<int, std::string> MENU = {
      {1, "CUDA Sample from website"}};

  static void testCudaSample(); 

 public:
  void runLab() override;
};
}  // namespace utcn::ip

#endif  // PROBA_CUDA_HH