#ifndef LAB3_HH
#define LAB3_HH

#include <vector>

#include "lab.hh"

namespace utcn::ip {
class Lab3 : public Lab {
  static inline std::map<int, std::string> LAB3_MENU = {
      {1,
       "Compute the histogram for a given grayscale image (in an array of "
       "integers having dimension 256)"},
      {2, "Compute the PDF (in an array of floats of dimension 256)"},
      {3, "Display the computed histogram using the provided function"},
      {4, "Compute the histogram for a given number of bins m ≤ 256"},
      {5, "Implement the multilevel thresholding algorithm from section 3.3"},
      {6,
       "Enhance the multilevel thresholding algorithm using the "
       "Floyd-Steinberg dithering from section 3.4"},
      {7,
       "Perform multilevel thresholding on a color image by applying the "
       "procedure from section 3.3 on the Hue channel from the HSV color-space "
       "representation of the image. Modify only the Hue values, keeping the S "
       "and V channels unchanged or setting them to their maximum possible "
       "value. Transform the result back to RGB color-space for viewing"}};

  static std::vector<int> getHistorgram(const cv::Mat_<uchar> &image);
  static void testHistogram();
  static std::vector<float> getPDF(const std::vector<int> &histogram,
                                   const uint32_t M);
  static void testPDF();
  static void displayHistogram(const std::string &name,
                               const std::vector<int> hist, const int hist_cols,
                               const int hist_height);
  static void testDisplayHistogram();
  static cv::Mat_<uchar> applyMultiLevelThreshold(const cv::Mat_<uchar> &src,
                                                  const int wh,
                                                  const float th);
  static void testMultiLevelThreshold();                                  

 public:
  void runLab() override;
};
}  // namespace utcn::ip

#endif  // LAB3_HH