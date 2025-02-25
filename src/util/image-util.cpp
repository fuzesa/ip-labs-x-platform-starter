#include "image-util.hh"

#include <opencv2/imgproc.hpp>

void utcn::ip::ImageUtil::resizeImg(const cv::Mat& src, cv::Mat& dst,
                                    int max_size, bool interpolate) {
  double ratio = 1;
  const double w = src.cols;
  const double h = src.rows;
  if (w > h)
    ratio = w / static_cast<double>(max_size);
  else
    ratio = h / static_cast<double>(max_size);
  const int nw = static_cast<int>(w / ratio);
  const int nh = static_cast<int>(h / ratio);
  const cv::Size sz(nw, nh);
  if (interpolate)
    resize(src, dst, sz);
  else
    resize(src, dst, sz, 0, 0, cv::INTER_NEAREST);
}