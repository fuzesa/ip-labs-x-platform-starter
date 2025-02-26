#ifndef IMAGE_UTIL_HH
#define IMAGE_UTIL_HH

#include <opencv4/opencv2/core.hpp>

class ImageUtil {
 private:
  ImageUtil() = default;
  ~ImageUtil() = default;

 public:
  static void resizeImg(const cv::Mat& src, cv::Mat& dst, int maxSize,
                        bool interpolate);
};

#endif  // IMAGE_UTIL_HH
