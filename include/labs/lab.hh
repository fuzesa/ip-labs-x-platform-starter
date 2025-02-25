#ifndef LAB_HH
#define LAB_HH

#include <map>
#include <opencv2/opencv.hpp>

#include "file-util.hh"
#include "image-util.hh"

using fileutil = utcn::ip::FileUtil;
using imageutil = utcn::ip::ImageUtil;

namespace utcn::ip {
class Lab {
 private:
#if defined(WIN32) || defined(_WIN32) || \
    defined(__WIN32) && !defined(__CYGWIN__)
  static inline const char* CLEAR_TERMINAL = "cls";
#else
  static inline const char* CLEAR_TERMINAL = "clear";
#endif

 protected:
  virtual void runLab() = 0;
  static void printMenu(const std::map<int, std::string>& entries);

 public:
  virtual ~Lab() = default;
};
}  // namespace utcn::ip

#endif  // LAB_HH
