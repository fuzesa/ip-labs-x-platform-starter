#include "lab2.hh"

void utcn::ip::Lab2::runLab() {
  int op;
  do {
    printMenu(LAB2_MENU);
    std::cin >> op;
    switch (op) {
      case 0:
        break;
      case 1:
        testDisplayRGBSeparately();
        break;
      case 2:
        testDisplayRGBSeparatelyFast();
        break;
      case 3:
        testRGB2Gray();
        break;
      case 4:
        testGray2Binary();
        break;
      case 5:
        testRGB2HSV();
        break;
      case 6:
        testIsInside();
        break;
      default:
        std::cout << "Invalid selection" << std::endl;
    }
  } while (op != 0);
}

void utcn::ip::Lab2::testDisplayRGBSeparately() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat_<cv::Vec3b> src =
        cv::imread(abs_image_path, cv::IMREAD_COLOR);

    const auto t1 = std::chrono::high_resolution_clock::now();

    const int height = src.rows;
    const int width = src.cols;

    cv::Mat_<uchar> red(height, width);
    cv::Mat_<uchar> green(height, width);
    cv::Mat_<uchar> blue(height, width);

    for (int i = 0; i < src.rows; i++) {
      for (int j = 0; j < src.cols; j++) {
        red(i, j) = src(i, j)[2];
        green(i, j) = src(i, j)[1];
        blue(i, j) = src(i, j)[0];
      }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    std::cout << "It took "
              << std::chrono::duration<double, std::milli>(t2 - t1)
              << std::endl;

    imshow("source", src);
    imshow("red", red);
    imshow("green", green);
    imshow("blue", blue);
    cv::waitKey();
  }
}

void utcn::ip::Lab2::testDisplayRGBSeparatelyFast() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat_<cv::Vec3b> src =
        cv::imread(abs_image_path, cv::IMREAD_COLOR);

    const auto t1 = std::chrono::high_resolution_clock::now();

    const int height = src.rows;
    const int width = src.cols;

    cv::Mat_<uchar> red(height, width);
    cv::Mat_<uchar> green(height, width);
    cv::Mat_<uchar> blue(height, width);

    for (int i = 0; i < height; i++) {
      const uchar *col_ptr = src.ptr(i);
      for (int j = 0; j < width; j++) {
        const uchar *pixel = col_ptr;
        red(i, j) = pixel[2];
        green(i, j) = pixel[1];
        blue(i, j) = pixel[0];
        col_ptr += 3;
      }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    std::cout << "It took "
              << std::chrono::duration<double, std::milli>(t2 - t1)
              << std::endl;

    imshow("source", src);
    imshow("red", red);
    imshow("green", green);
    imshow("blue", blue);
    cv::waitKey();
  }
}

void utcn::ip::Lab2::testRGB2Gray() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat_<cv::Vec3b> src =
        cv::imread(abs_image_path, cv::IMREAD_COLOR);

    const auto t1 = std::chrono::high_resolution_clock::now();

    const int height = src.rows;
    const int width = src.cols;

    cv::Mat_<uchar> gray(height, width);

    for (int i = 0; i < height; i++) {
      const uchar *col_ptr = src.ptr(i);
      for (int j = 0; j < width; j++) {
        const uchar *pixel = col_ptr;
        gray(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
        col_ptr += 3;
      }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    std::cout << "It took "
              << std::chrono::duration<double, std::milli>(t2 - t1)
              << std::endl;

    imshow("source", src);
    imshow("gray", gray);
    cv::waitKey();
  }
}

void utcn::ip::Lab2::testGray2Binary() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    cv::Mat_<uchar> src = cv::imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    int threshold;
    std::cout << "Please enter the value for the threshold: ";
    std::cin >> threshold;
    cv::Mat_<uchar> dst(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
      for (int j = 0; j < src.cols; j++) {
        dst(i, j) = src(i, j) > threshold ? 255 : 0;
      }
    }
    cv::imshow("Source", src);
    cv::imshow("B & W", dst);
    cv::waitKey(0);
  }
}

std::vector<float> getNormalizedRGB(const uchar *pixel) {
  std::vector<float> rgb(3);
  rgb[0] = (float)pixel[0] / 255.0;
  rgb[1] = (float)pixel[1] / 255.0;
  rgb[2] = (float)pixel[2] / 255.0;
  return rgb;
}

void utcn::ip::Lab2::testRGB2HSV() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat_<cv::Vec3b> src =
        cv::imread(abs_image_path, cv::IMREAD_COLOR);

    const auto t1 = std::chrono::high_resolution_clock::now();

    const int height = src.rows;
    const int width = src.cols;

    cv::Mat_<uchar> H_norm(height, width);
    cv::Mat_<uchar> S_norm(height, width);
    cv::Mat_<uchar> V_norm(height, width);

    for (int i = 0; i < height; i++) {
      const uchar *col_ptr = src.ptr(i);
      for (int j = 0; j < width; j++) {
        const uchar *pixel = col_ptr;

        // r, g, b
        const std::vector<float> rgb = getNormalizedRGB(pixel);

        // M, m, C
        const float M = *std::max_element(rgb.begin(), rgb.end());
        const float m = *std::min_element(rgb.begin(), rgb.end());
        const float C = M - m;

        // H, S, V

        const float V = M;

        const float S = V == 0 ? 0 : C / V;

        const float H = C == 0        ? 0
                        : M == rgb[2] ? 60 * (rgb[1] - rgb[0]) / C
                        : M == rgb[1] ? 120 + 60 * (rgb[0] - rgb[2]) / C
                        : M == rgb[0] ? 240 + 60 * (rgb[2] - rgb[1]) / C
                                      : 0;
        H_norm(i, j) = (uchar)(H * 255 / 360);
        S_norm(i, j) = (uchar)(S * 255);
        V_norm(i, j) = (uchar)(V * 255);
        col_ptr += 3;
      }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    std::cout << "It took "
              << std::chrono::duration<double, std::milli>(t2 - t1)
              << std::endl;

    cv::imshow("source", src);
    cv::imshow("H", H_norm);
    cv::imshow("S", S_norm);
    cv::imshow("V", V_norm);
    cv::waitKey();
  }
}

bool utcn::ip::Lab2::isInside(const cv::Mat &img, int i, int j) {
  /* cv::Point p(i, j);
  if (p.inside(cv::Rect(0, 0, img.cols, img.rows))) {
    std::cout << "Point is inside" << std::endl;
  } else {
    std::cout << "Point is outside" << std::endl;
  } */
  return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

void utcn::ip::Lab2::testIsInside() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat_<cv::Vec3b> src =
        cv::imread(abs_image_path, cv::IMREAD_COLOR);

    int x, y;
    std::cout << "Please enter the X coordinate of the point: ";
    std::cin >> x;
    std::cout << "Please enter the Y coordinate of the point: ";
    std::cin >> y;

    if (isInside(src, x, y)) {
      std::cout << "Point is inside" << std::endl;
    } else {
      std::cout << "Point is NOT inside" << std::endl;
    }

    std::cin.get();
    std::cin.get();
  }
}