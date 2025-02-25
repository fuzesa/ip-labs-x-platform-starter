#ifndef LAB2_HH
#define LAB2_HH

#include <chrono>

#include "lab.hh"

namespace utcn::ip {
class Lab2 : public Lab {
  static inline std::map<int, std::string> LAB2_MENU = {
      {1, "Display R, G, B separately"},
      {2, "Display R, G, B separately Fast"},
      {3, "RGB -> Grayscale"},
      {4, "Grayscale -> Binary with threshold from stdin"},
      {5, "RGB -> HSV"},
      {6, "isInside"}};

  static void testDisplayRGBSeparately();
  static void testDisplayRGBSeparatelyFast();
  static void testRGB2Gray();
  static void testGray2Binary();
  static void testRGB2HSV();
  static bool isInside(const cv::Mat &img, int i, int j);
  static void testIsInside();

 public:
  void runLab() override;
};
}  // namespace utcn::ip

#endif  // LAB2_HH