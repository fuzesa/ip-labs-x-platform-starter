#include "lab3.hh"

void utcn::ip::Lab3::runLab() {
  int op;
  do {
    printMenu(LAB3_MENU);
    std::cin >> op;
    switch (op) {
      case 0:
        break;
      case 1:
        testHistogram();
        break;
      case 2:
        testPDF();
        break;
      case 3:
        testDisplayHistogram();
        break;
      case 4:
        testMultiLevelThreshold();
        break;
      case 5:

        break;
      case 6:

        break;
      default:
        std::cout << "Invalid selection" << std::endl;
    }
  } while (op != 0);
}

std::vector<int> utcn::ip::Lab3::getHistorgram(const cv::Mat_<uchar> &src) {
  std::vector<int> histogram(256, 0);
  for (int i = 0; i < src.rows; i++) {
    const uchar *col_ptr = src.ptr(i);
    for (int j = 0; j < src.cols; j++) {
      histogram[col_ptr[j]]++;
    }
  }
  return histogram;
}

void utcn::ip::Lab3::testHistogram() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat_<uchar> src =
        cv::imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    std::vector<int> histogram = getHistorgram(src);
    for (int i = 0; i < histogram.size(); i++) {
      std::cout << (int)i << " " << (int)histogram[i] << std::endl;
    }
  }
  std::cin.get();
  std::cin.get();
}

std::vector<float> utcn::ip::Lab3::getPDF(const std::vector<int> &histogram,
                                          const uint32_t M) {
  std::vector<float> pdf(256, 0);
  for (int i = 0; i < histogram.size(); i++) {
    pdf[i] = (float)histogram[i] / M;
  }
  return pdf;
}

void utcn::ip::Lab3::testPDF() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat_<uchar> src =
        cv::imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    std::vector<int> histogram = getHistorgram(src);
    std::vector<float> pdf = getPDF(histogram, src.rows * src.cols);
    for (int i = 0; i < pdf.size(); i++) {
      std::cout << (int)i << " " << pdf[i] << std::endl;
    }
  }
  std::cin.get();
  std::cin.get();
}

void utcn::ip::Lab3::displayHistogram(const std::string &name,
                                      const std::vector<int> hist,
                                      const int hist_cols,
                                      const int hist_height) {
  cv::Mat imgHist(hist_height, hist_cols, CV_8UC3,
                  CV_RGB(255, 255, 255));  // constructs a white image

  // computes histogram maximum
  int max_hist = 0;
  for (int i = 0; i < hist_cols; i++)
    if (hist[i] > max_hist) max_hist = hist[i];
  double scale = 1.0;
  scale = (double)hist_height / max_hist;
  int baseline = hist_height - 1;
  for (int x = 0; x < hist_cols; x++) {
    cv::Point p1 = cv::Point(x, baseline);
    cv::Point p2 = cv::Point(x, baseline - cvRound(hist[x] * scale));
    line(imgHist, p1, p2, CV_RGB(255, 0, 255));  // histogram bins
                                                 // colored in magenta
  }
  imshow(name, imgHist);
  cv::waitKey(0);
}

int *getHistogram2(const cv::Mat_<uchar> &src) {
  int *histogram = new int[256];
  for (int i = 0; i < 256; i++) {
    histogram[i] = 0;
  }
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      histogram[src(i, j)]++;
    }
  }
  return histogram;
}

void utcn::ip::Lab3::testDisplayHistogram() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat_<uchar> src =
        cv::imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    std::vector<int> histogram = getHistorgram(src);
    // int *histogram = getHistogram2(src);
    displayHistogram("Histogram", histogram, 256, 256);
  }
}

cv::Mat_<uchar> utcn::ip::Lab3::applyMultiLevelThreshold(
    const cv::Mat_<uchar> &src, const int wh = 5, const float th = 0.0003) {
  cv::Mat_<uchar> dst = src.clone();
  const std::vector<int> histogram = getHistorgram(src);
  const std::vector<float> pdf = getPDF(histogram, src.rows * src.cols);
  const int window_width = 2 * wh + 1;
  
  return dst;
}

void utcn::ip::Lab3::testMultiLevelThreshold() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat_<uchar> src =
        cv::imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat_<uchar> dst = applyMultiLevelThreshold(src, 0, 128);
    cv::imshow("Thresholded", dst);
    cv::waitKey(0);
  }
}