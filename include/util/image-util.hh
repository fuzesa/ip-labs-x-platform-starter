#ifndef IMAGE_UTIL_HH
#define IMAGE_UTIL_HH

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ImageUtil {
 private:
  ImageUtil() = default;
  ~ImageUtil() = default;

 public:
  static void resizeImg(const cv::Mat& src, cv::Mat& dst, int maxSize,
                        bool interpolate);

  static void waitKey(int delay = 0);
};

#endif  // IMAGE_UTIL_HH
