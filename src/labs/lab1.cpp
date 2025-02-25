#include "lab1.hh"

void utcn::ip::Lab1::runLab() {
  int op;
  do {
    printMenu(LAB1_MENU);
    std::cin >> op;
    switch (op) {
      case 0:
        break;
      case 1:
        testOpenImage();
        break;
      case 2:
        testOpenImagesFld();
        break;
      case 3:
        testNegativeImage();
        break;
      case 4:
        testNegativeImageFast();
        break;
      case 5:
        testNegativeImageParallel();
        break;
      case 6:
        testNegativeImageCUDA();
        break;
      case 7:
        testColor2Gray();
        break;
      case 8:
        testImageOpenAndSave();
        break;
      case 9:
        testBGR2HSV();
        break;
      case 10:
        testResize();
        break;
      case 11:
        testCanny();
        break;
      case 12:
        testVideoSequence();
        break;
      case 13:
        testSnap();
        break;
      case 14:
        testMouseClick();
        break;
      case 15:
        testChangeGrayLevelsAdditive();
        break;
      case 16:
        testChangeGrayLevelsMultiplicative();
        break;
      case 17:
        testDrawFourSquare();
        break;
      case 18:
        testPrintInverseOfMatrix();
        break;
      default:
        std::cout << "Invalid selection" << std::endl;
    }
  } while (op != 0);
}

void utcn::ip::Lab1::testOpenImage() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat src = cv::imread(abs_image_path);
    imshow("image", src);
    cv::waitKey();
  }
}

// Not recursive
void utcn::ip::Lab1::testOpenImagesFld() {
  const auto abs_file_paths = fileutil::getAllFilesInDirectory();
  for (const auto &abs_file_path : abs_file_paths) {
    const cv::Mat src = cv::imread(abs_file_path);
    const std::filesystem::path path = abs_file_path;
    imshow(path.filename().string(), src);
  }
  cv::waitKey();
}

void utcn::ip::Lab1::testNegativeImage() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat src = imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    const int height = src.rows;
    const int width = src.cols;
    auto dst = cv::Mat(height, width, CV_8UC1);

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int num = 0; num < 20000; num++) {
      // Accessing individual pixels in an 8 bits/pixel image
      // Inefficient way -> slow
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          const uchar val = src.at<uchar>(i, j);
          const uchar neg = 255 - val;
          dst.at<uchar>(i, j) = neg;
        }
      }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    std::cout << "It took "
              << std::chrono::duration<double, std::milli>(t2 - t1)
              << std::endl;

    imshow("input image", src);
    imshow("negative image", dst);
    cv::waitKey();
  }
}

// https://longstryder.com/2014/07/which-way-of-accessing-pixels-in-opencv-is-the-fastest/

void utcn::ip::Lab1::testNegativeImageFast() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat src = imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    const int height = src.rows;
    const int width = src.cols;
    const auto dst = cv::Mat(height, width, CV_8UC1);

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int num = 0; num < 20000; num++) {
      // The fastest approach of accessing the pixels -> using pointers
      const uchar *lpSrc = src.data;
      uchar *lpDst = dst.data;
      const int w = (int)src.step;  // no dword alignment is done !!!
      for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
          const uchar val = lpSrc[i * w + j];
          lpDst[i * w + j] = 255 - val;
        }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    std::cout << "It took "
              << std::chrono::duration<double, std::milli>(t2 - t1)
              << std::endl;

    imshow("input image", src);
    imshow("negative image", dst);
    cv::waitKey();
  }
}

void utcn::ip::Lab1::testNegativeImageParallel() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat src = imread(abs_image_path, cv::IMREAD_GRAYSCALE);

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int num = 0; num < 20000; num++) {
      src.forEach<uchar>(
          [](uchar &curr, const int *position) -> void { curr = 255 - curr; });
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    std::cout << "It took "
              << std::chrono::duration<double, std::milli>(t2 - t1)
              << std::endl;

    imshow("input image", src);
    cv::waitKey();
  }
}

void utcn::ip::Lab1::testNegativeImageCUDA() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat src_cpu = imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat dst_cpu;
    cv::cuda::GpuMat src_gpu, dst_gpu;

    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    std::cout << "It took "
              << std::chrono::duration<double, std::milli>(t2 - t1)
              << std::endl;

    imshow("input image", dst_cpu);
    cv::waitKey();
  }
}

void utcn::ip::Lab1::testColor2Gray() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    cv::Mat src = cv::imread(abs_image_path);

    const int height = src.rows;
    const int width = src.cols;

    auto dst = cv::Mat(height, width, CV_8UC1);

    // Accessing individual pixels in a RGB 24 bits/pixel image
    // Inefficient way --> slow
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        auto v3 = src.at<cv::Vec3b>(i, j);
        const uchar b = v3[0];
        const uchar g = v3[1];
        const uchar r = v3[2];
        dst.at<uchar>(i, j) = (r + g + b) / 3;
      }
    }

    imshow("input image", src);
    imshow("gray image", dst);
    cv::waitKey();
  }
}

void utcn::ip::Lab1::testImageOpenAndSave() {
  cv::Mat dst;
  const std::string path_to_src = ASSETS_DIR "Images/Lena_24bits.bmp";
  const cv::Mat src = imread(path_to_src, cv::IMREAD_COLOR);  // Read the image

  if (!src.data) {
    std::cout << "Could not open or find the image" << std::endl;
    return;
  }

  // Get the image resolution
  const auto src_size = cv::Size(src.cols, src.rows);

  // Display window
  const auto WIN_SRC = "Src";  // window for the source image
  namedWindow(WIN_SRC, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(WIN_SRC, 0, 0);

  const auto *WIN_DST = "Dst";  // window for the destination (processed) image
  namedWindow(WIN_DST, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(WIN_DST, src_size.width + 10, 0);

  cvtColor(src, dst,
           cv::COLOR_BGR2GRAY);  // converts the source image to a grayscale one

  const std::string path_to_dst = ASSETS_DIR "Images/Lena_24bits_gray.bmp";
  imwrite(path_to_dst, dst);  // writes the destination to
                              // file

  imshow(WIN_SRC, src);
  imshow(WIN_DST, dst);

  cv::waitKey(0);
}

void utcn::ip::Lab1::testBGR2HSV() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    cv::Mat src = cv::imread(abs_image_path);
    int height = src.rows;
    int width = src.cols;

    // HSV components
    auto H = cv::Mat(height, width, CV_8UC1);
    auto S = cv::Mat(height, width, CV_8UC1);
    auto V = cv::Mat(height, width, CV_8UC1);

    // Defining pointers to each matrix (8 bits/pixels) of the individual
    // components H, S, V
    uchar *lpH = H.data;
    uchar *lpS = S.data;
    uchar *lpV = V.data;

    cv::Mat hsvImg;
    cvtColor(src, hsvImg, cv::COLOR_BGR2HSV);

    // Defining the pointer to the HSV image matrix (24 bits/pixel)
    uchar *hsvDataPtr = hsvImg.data;

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        int hi = i * width * 3 + j * 3;
        int gi = i * width + j;

        lpH[gi] = hsvDataPtr[hi] * 510 / 360;  // lpH = 0 .. 255
        lpS[gi] = hsvDataPtr[hi + 1];          // lpS = 0 .. 255
        lpV[gi] = hsvDataPtr[hi + 2];          // lpV = 0 .. 255
      }
    }

    imshow("input image", src);
    imshow("H", H);
    imshow("S", S);
    imshow("V", V);

    cv::waitKey();
  }
}

void utcn::ip::Lab1::testResize() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat src = cv::imread(abs_image_path);
    cv::Mat dst1, dst2;
    // without interpolation
    imageutil::resizeImg(src, dst1, 320, false);
    // with interpolation
    imageutil::resizeImg(src, dst2, 320, true);
    imshow("input image", src);
    imshow("resized image (without interpolation)", dst1);
    imshow("resized image (with interpolation)", dst2);
    cv::waitKey();
  }
}

void utcn::ip::Lab1::testCanny() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    cv::Mat dst, gauss;
    const cv::Mat src = imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    constexpr double k = 0.4;
    constexpr int pH = 50;
    constexpr int pL = static_cast<int>(k) * pH;
    GaussianBlur(src, gauss, cv::Size(5, 5), 0.8, 0.8);
    Canny(gauss, dst, pL, pH, 3);
    imshow("input image", src);
    imshow("canny", dst);
    cv::waitKey();
  }
};

void utcn::ip::Lab1::testVideoSequence() {
  /* *** WARNING *** */
  /* UNCOMMENTING THE CONTENTS WITHIN THIS METHOD */
  /* COULD LEAD TO THE APPLICATION NOT WORKING */
  const std::string path_to_vid = ASSETS_DIR "Videos/rubic.avi";
  cv::VideoCapture cap(path_to_vid);  // off-line video from file
  // VideoCapture cap(0);	// live video from webcam
  if (!cap.isOpened()) {
    std::cout << "Cannot open video capture device" << std::endl;
    cv::waitKey(0);
    return;
  }

  cv::Mat edges;
  cv::Mat frame;

  while (cap.read(frame)) {
    cv::Mat grayFrame;
    cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    Canny(grayFrame, edges, 40, 100, 3);
    imshow("source", frame);
    imshow("gray", grayFrame);
    imshow("edges", edges);
    const uchar c =
        cv::waitKey(100);  // waits 100ms and advances to the next frame
    if (c == 27) {
      // press ESC to exit
      std::cout << "ESC pressed - capture finished" << std::endl;
      break;  // ESC pressed
    };
  }
}

void utcn::ip::Lab1::testSnap() {
  /* *** WARNING *** */
  /* UNCOMMENTING THE CONTENTS WITHIN THIS METHOD */
  /* COULD LEAD TO THE APPLICATION NOT WORKING */
  cv::VideoCapture cap(
      0);                 // open the deafult camera (i.e. the built in web cam)
  if (!cap.isOpened()) {  // openenig the video device failed
    std::cout << "Cannot open video capture device" << std::endl;
    return;
  }

  cv::Mat frame;
  char fileName[256];

  // video resolution
  const auto capS =
      cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
               static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

  // Display window
  const auto WIN_SRC = "Src";  // window for the source frame
  namedWindow(WIN_SRC, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(WIN_SRC, 0, 0);

  const auto WIN_DST = "Snapped";  // window for showing the snapped frame
  namedWindow(WIN_DST, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(WIN_DST, capS.width + 10, 0);

  int frameNum = -1;
  int frameCount = 0;

  for (;;) {
    cap >> frame;  // get a new frame from camera
    if (frame.empty()) {
      std::cout << "End of the video file" << std::endl;
      break;
    }

    ++frameNum;

    imshow(WIN_SRC, frame);

    const uchar c =
        cv::waitKey(10);  // waits a key press to advance to the next frame
    if (c == 27) {
      // press ESC to exit
      std::cout << "ESC pressed - capture finished" << std::endl;
      break;  // ESC pressed
    }
    if (c == 115) {
      char numberStr[256];
      //'s' pressed - snap the image to a file
      frameCount++;
      fileName[0] = '\0';
      sprintf(numberStr, "%d", frameCount);
      strcat(fileName, ASSETS_DIR "Images/A");
      strcat(fileName, numberStr);
      strcat(fileName, ".bmp");
      const bool bSuccess = imwrite(fileName, frame);
      if (!bSuccess) {
        std::cout << "Error writing the snapped image" << std::endl;
      } else
        imshow(WIN_DST, frame);
    }
  }
}

void utcn::ip::Lab1::myCallBackFunc(int event, int x, int y, int flags,
                                    void *param) {
  // More examples:
  // http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
  auto *src = static_cast<cv::Mat *>(param);
  if (event == cv::EVENT_LBUTTONDOWN) {
    // C style casting
    // Doesn't check at compile time
    // (int)(*src).at<cv::Vec3b>(y, x)[2],
    // (int)(*src).at<cv::Vec3b>(y, x)[1],
    // (int)(*src).at<cv::Vec3b>(y, x)[0]);

    // Using C++ static_cast, this checks at compile time
    std::cout << "Pos(x,y): " << x << "," << y << " Color(RGB): "
              << static_cast<int>(src->at<cv::Vec3b>(y, x)[2]) << ","
              << static_cast<int>(src->at<cv::Vec3b>(y, x)[1]) << ","
              << static_cast<int>(src->at<cv::Vec3b>(y, x)[0]) << std::endl;
  }
}

void utcn::ip::Lab1::testMouseClick() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    cv::Mat src = cv::imread(abs_image_path);
    // Create a window
    cv::namedWindow("My Window", 1);

    // set the callback function for any mouse event
    cv::setMouseCallback("My Window", myCallBackFunc, &src);

    // show the image
    imshow("My Window", src);

    // Wait until user press some key
    cv::waitKey(0);
  }
}

cv::Mat utcn::ip::Lab1::changeByFactor(const cv::Mat &orig_pic,
                                       const bool isAdditive,
                                       const uchar factor) {
  const int height = orig_pic.rows;
  const int width = orig_pic.cols;
  cv::Mat dst = cv::Mat(height, width, CV_8UC1);

  const uchar *lpSrc = orig_pic.data;
  uchar *lpDst = dst.data;

  const int w = (int)orig_pic.step;  // no dword alignment is done !!!
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      const uchar val = lpSrc[i * w + j];
      //  lpDst[i * w + j] = val + factor;
      if (isAdditive) {
        lpDst[i * w + j] = val + factor > 255 ? 255 : val + factor;
      } else {
        lpDst[i * w + j] = val * factor > 255 ? 255 : val * factor;
      }
    }
  return dst;
}

void utcn::ip::Lab1::testChangeGrayLevelsAdditive() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat src = imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat dst = changeByFactor(src, true, ADDITIVE_FACTOR);
    imshow("Original", src);
    imshow("Modified", dst);
    cv::waitKey();
  }
}

void utcn::ip::Lab1::testChangeGrayLevelsMultiplicative() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat src = imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat dst = changeByFactor(src, false, MULTIPLICATIVE_FACTOR);
    imshow("Original", src);
    imshow("Modified", dst);
    const std::string path_to_dst = ASSETS_DIR "Images/grayscale_multi.bmp";
    imwrite(path_to_dst, dst);
    cv::waitKey();
  }
}

void utcn::ip::Lab1::testDrawFourSquare() {
  cv::Mat square(256, 256, CV_8UC3);

  const int height = square.rows;
  const int width = square.cols;
  //  cv::Vec3b *lpSrc = square.data;

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      if (i > 128 && j > 128) {
        //        lpSrc[i * w + j] = YELLOW;
        square.at<cv::Vec3b>(i, j) = YELLOW;
      } else if (i > 128) {
        square.at<cv::Vec3b>(i, j) = RED;
      } else if (j > 128) {
        square.at<cv::Vec3b>(i, j) = GREEN;
      } else {
        square.at<cv::Vec3b>(i, j) = WHITE;
      }
    }
  imshow("Multi-color Square", square);
  cv::waitKey();
}

void utcn::ip::Lab1::testPrintInverseOfMatrix() {
  std::cout << "Original matrix: " << std::endl
            << MATRIX3X3 << std::endl
            << std::endl;
  const cv::Mat inverted = MATRIX3X3.inv();
  std::cout << "Inverse: " << std::endl << inverted << std::endl;
  std::cin.get();
  std::cin.get();
}