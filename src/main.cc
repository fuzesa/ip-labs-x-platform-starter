#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "file-util.hh"
#include "image-util.hh"
#include "terminal-util.hh"

/**
 * IMPORTANT!!!
 * In professional environments it is usually frowned upon to use
 * namespace bla-bla anything. This is because it can lead to naming
 * conflicts in case some libraries have the same function names.
 * Since this is university, this will make some of the syntex a bit
 * easier to read, so that's why we're using it here.
 */
using namespace cv;
using namespace std;

/**
 * CONSTANTS FOR SOME OF THE LAB ASSIGNMENTS
 */

// Lab 1 Constants
static inline uchar ADDITIVE_FACTOR = 54;
static inline uchar MULTIPLICATIVE_FACTOR = 3;
static inline Vec3b WHITE{255, 255, 255};
static inline Vec3b RED{0, 0, 255};
static inline Vec3b GREEN{0, 255, 0};
static inline Vec3b YELLOW{0, 255, 255};
static inline float MATRIX_VALS[9] = {2, 3, 1, 3, 4, 1, 3, 7, 2};
static inline Mat MATRIX3X3{3, 3, CV_32FC1, MATRIX_VALS};
// End of Lab 1 Constants

/**
 * LAB 1
 */
void testOpenImage() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat src = imread(abs_image_path);
    imshow("image", src);
    ImageUtil::waitKey();
  }
}

// Not recursive
void testOpenImagesFld() {
  const auto abs_file_paths = FileUtil::getAllFilesInDirectory();
  for (const auto& abs_file_path : abs_file_paths) {
    const Mat src = imread(abs_file_path);
    const filesystem::path path = abs_file_path;
    imshow(path.filename().string(), src);
  }
  ImageUtil::waitKey();
}

void testNegativeImage() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);
    const int height = src.rows;
    const int width = src.cols;
    auto dst = Mat(height, width, CV_8UC1);

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int num = 0; num < 1000; num++) {
      // Accessing individual pixels in an 8 bits/pixel image
      // Inefficient way -> slow
      cout << "num: " << num << endl;
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
    cout << "It took " << std::chrono::duration<double, std::milli>(t2 - t1)
         << endl;

    imshow("input image", src);
    imshow("negative image", dst);
    ImageUtil::waitKey();
  }
}

// https://longstryder.com/2014/07/which-way-of-accessing-pixels-in-opencv-is-the-fastest/

void testNegativeImageFast() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);
    const int height = src.rows;
    const int width = src.cols;
    const auto dst = Mat(height, width, CV_8UC1);

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int num = 0; num < 1000; num++) {
      // The fastest approach of accessing the pixels -> using pointers
      cout << "num: " << num << endl;
      const uchar* lpSrc = src.data;
      uchar* lpDst = dst.data;
      const int w = (int)src.step;  // no dword alignment is done !!!
      for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
          const uchar val = lpSrc[i * w + j];
          lpDst[i * w + j] = 255 - val;
        }
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    cout << "It took " << std::chrono::duration<double, std::milli>(t2 - t1)
         << endl;

    imshow("input image", src);
    imshow("negative image", dst);
    ImageUtil::waitKey();
  }
}

void testNegativeImageParallel() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);
    Mat dst = src.clone();

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int num = 0; num < 1000; num++) {
      cout << "num: " << num << endl;
      // OpenCV forEach
      src.forEach<uchar>([&dst](uchar& curr, const int* position) -> void {
        dst.at<uchar>(position) = 255 - curr;
      });
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    cout << "It took " << std::chrono::duration<double, std::milli>(t2 - t1)
         << endl;

    imshow("input image", src);
    imshow("negative image", dst);
    ImageUtil::waitKey();
  }
}

void testNegativeImageUnifiedMat() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    // UMat is a class to store data in GPU memory
    // CANNOT BE const!!!
    UMat src, dst;
    imread(abs_image_path, IMREAD_GRAYSCALE).copyTo(src);

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int num = 0; num < 1000; num++) {
      cout << "num: " << num << endl;
      bitwise_not(src, dst);
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    cout << "It took " << std::chrono::duration<double, std::milli>(t2 - t1)
         << endl;

    imshow("input image", src);
    imshow("negative image", dst);
    ImageUtil::waitKey();
  }
}

void testColor2Gray() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    Mat src = imread(abs_image_path);

    const int height = src.rows;
    const int width = src.cols;

    auto dst = Mat(height, width, CV_8UC1);

    // Accessing individual pixels in a RGB 24 bits/pixel image
    // Inefficient way --> slow
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        auto v3 = src.at<Vec3b>(i, j);
        const uchar b = v3[0];
        const uchar g = v3[1];
        const uchar r = v3[2];
        dst.at<uchar>(i, j) = (r + g + b) / 3;
      }
    }

    imshow("input image", src);
    imshow("gray image", dst);
    ImageUtil::waitKey();
  }
}

void testImageOpenAndSave() {
  Mat dst;
  const string path_to_src = ASSETS_DIR "Images/starry_night.bmp";
  const Mat src = imread(path_to_src, IMREAD_COLOR);  // Read the image

  if (!src.data) {
    cout << "Could not open or find the image" << endl;
    return;
  }

  // Get the image resolution
  const auto src_size = Size(src.cols, src.rows);

  // Display window
  const auto WIN_SRC = "Src";  // window for the source image
  namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
  moveWindow(WIN_SRC, 0, 0);

  const auto* WIN_DST = "Dst";  // window for the destination (processed) image
  namedWindow(WIN_DST, WINDOW_AUTOSIZE);
  moveWindow(WIN_DST, src_size.width + 10, 0);

  cvtColor(src, dst,
           COLOR_BGR2GRAY);  // converts the source image to a grayscale one

  const string path_to_dst = ASSETS_DIR "Images/starry_night_gray.bmp";
  imwrite(path_to_dst, dst);  // writes the destination to
                              // file

  imshow(WIN_SRC, src);
  imshow(WIN_DST, dst);

  ImageUtil::waitKey();
}

void testBGR2HSV() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    Mat src = imread(abs_image_path);
    int height = src.rows;
    int width = src.cols;

    // HSV components
    auto H = Mat(height, width, CV_8UC1);
    auto S = Mat(height, width, CV_8UC1);
    auto V = Mat(height, width, CV_8UC1);

    // Defining pointers to each matrix (8 bits/pixels) of the individual
    // components H, S, V
    uchar* lpH = H.data;
    uchar* lpS = S.data;
    uchar* lpV = V.data;

    Mat hsvImg;
    cvtColor(src, hsvImg, COLOR_BGR2HSV);

    // Defining the pointer to the HSV image matrix (24 bits/pixel)
    uchar* hsvDataPtr = hsvImg.data;

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

    ImageUtil::waitKey();
  }
}

void testResize() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat src = imread(abs_image_path);
    Mat dst1, dst2;
    // without interpolation
    ImageUtil::resizeImg(src, dst1, 320, false);
    // with interpolation
    ImageUtil::resizeImg(src, dst2, 320, true);
    imshow("input image", src);
    imshow("resized image (without interpolation)", dst1);
    imshow("resized image (with interpolation)", dst2);
    ImageUtil::waitKey();
  }
}

void testCanny() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    Mat dst, gauss;
    const Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);
    constexpr double k = 0.4;
    constexpr int pH = 50;
    constexpr int pL = static_cast<int>(k) * pH;
    GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
    Canny(gauss, dst, pL, pH, 3);
    imshow("input image", src);
    imshow("canny", dst);
    ImageUtil::waitKey();
  }
};

void testVideoSequence() {
  /* *** WARNING *** */
  /* UNCOMMENTING THE CONTENTS WITHIN THIS METHOD */
  /* COULD LEAD TO THE APPLICATION NOT WORKING */
  const string path_to_vid = ASSETS_DIR "Videos/rubic.avi";
  VideoCapture cap(path_to_vid);  // off-line video from file
  // VideoCapture cap(0);	// live video from webcam
  if (!cap.isOpened()) {
    cout << "Cannot open video capture device" << endl;
    waitKey(0);
    return;
  }

  Mat edges;
  Mat frame;

  while (cap.read(frame)) {
    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
    Canny(grayFrame, edges, 40, 100, 3);
    imshow("source", frame);
    imshow("gray", grayFrame);
    imshow("edges", edges);
    const uchar c = waitKey(100);  // waits 100ms and advances to the next frame
    if (c == 27) {
      // press ESC to exit
      cout << "ESC pressed - capture finished" << endl;
      break;  // ESC pressed
    };
  }
#ifdef __APPLE__
  destroyAllWindows();
  waitKey(1);
#endif
}

void testSnap() {
  /* *** WARNING *** */
  /* UNCOMMENTING THE CONTENTS WITHIN THIS METHOD */
  /* COULD LEAD TO THE APPLICATION NOT WORKING */
  VideoCapture cap(0);    // open the deafult camera (i.e. the built in web cam)
  if (!cap.isOpened()) {  // openenig the video device failed
    cout << "Cannot open video capture device" << endl;
    return;
  }

  Mat frame;
  char fileName[256];

  // video resolution
  const auto capS = Size(static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)),
                         static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)));

  // Display window
  const auto WIN_SRC = "Src";  // window for the source frame
  namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
  moveWindow(WIN_SRC, 0, 0);

  const auto WIN_DST = "Snapped";  // window for showing the snapped frame
  namedWindow(WIN_DST, WINDOW_AUTOSIZE);
  moveWindow(WIN_DST, capS.width + 10, 0);

  int frameNum = -1;
  int frameCount = 0;

  for (;;) {
    cap >> frame;  // get a new frame from camera
    if (frame.empty()) {
      cout << "End of the video file" << endl;
      break;
    }

    ++frameNum;

    imshow(WIN_SRC, frame);

    const uchar c =
        waitKey(10);  // waits a key press to advance to the next frame
    if (c == 27) {
      // press ESC to exit
      cout << "ESC pressed - capture finished" << endl;
      break;  // ESC pressed
    }
    if (c == 115) {
      char numberStr[256];
      //'s' pressed - snap the image to a file
      frameCount++;
      fileName[0] = '\0';
      snprintf(numberStr, sizeof(numberStr), "%d", frameCount);
      strcat(fileName, ASSETS_DIR "Images/A");
      strcat(fileName, numberStr);
      strcat(fileName, ".bmp");
      const bool bSuccess = imwrite(fileName, frame);
      if (!bSuccess) {
        cout << "Error writing the snapped image" << endl;
      } else
        imshow(WIN_DST, frame);
    }
  }
#ifdef __APPLE__
  destroyAllWindows();
  waitKey(1);
#endif
}

void myCallBackFunc(int event, int x, int y, int flags, void* param) {
  // More examples:
  // http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
  auto* src = static_cast<Mat*>(param);
  if (event == EVENT_LBUTTONDOWN) {
    // C style casting
    // Doesn't check at compile time
    // (int)(*src).at<Vec3b>(y, x)[2],
    // (int)(*src).at<Vec3b>(y, x)[1],
    // (int)(*src).at<Vec3b>(y, x)[0]);

    // Using C++ static_cast, this checks at compile time
    cout << "Pos(x,y): " << x << "," << y
         << " Color(RGB): " << static_cast<int>(src->at<Vec3b>(y, x)[2]) << ","
         << static_cast<int>(src->at<Vec3b>(y, x)[1]) << ","
         << static_cast<int>(src->at<Vec3b>(y, x)[0]) << endl;
  }
}

void testMouseClick() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    Mat src = imread(abs_image_path);
    // Create a window
    namedWindow("My Window", 1);

    // set the callback function for any mouse event
    setMouseCallback("My Window", myCallBackFunc, &src);

    // show the image
    imshow("My Window", src);

    // Wait until user press some key
    ImageUtil::waitKey();
  }
}

Mat changeByFactor(const Mat& orig_pic, const bool isAdditive,
                   const uchar factor) {
  const int height = orig_pic.rows;
  const int width = orig_pic.cols;
  Mat dst = Mat(height, width, CV_8UC1);

  const uchar* lpSrc = orig_pic.data;
  uchar* lpDst = dst.data;

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

void testChangeGrayLevelsAdditive() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);
    const Mat dst = changeByFactor(src, true, ADDITIVE_FACTOR);
    imshow("Original", src);
    imshow("Modified", dst);
    ImageUtil::waitKey();
  }
}

void testChangeGrayLevelsMultiplicative() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);
    const Mat dst = changeByFactor(src, false, MULTIPLICATIVE_FACTOR);
    imshow("Original", src);
    imshow("Modified", dst);
    const string path_to_dst = ASSETS_DIR "Images/grayscale_multi.bmp";
    imwrite(path_to_dst, dst);
    ImageUtil::waitKey();
  }
}

void testDrawFourSquare() {
  Mat square(256, 256, CV_8UC3);

  const int height = square.rows;
  const int width = square.cols;
  //  Vec3b *lpSrc = square.data;

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      if (i > 128 && j > 128) {
        //        lpSrc[i * w + j] = YELLOW;
        square.at<Vec3b>(i, j) = YELLOW;
      } else if (i > 128) {
        square.at<Vec3b>(i, j) = RED;
      } else if (j > 128) {
        square.at<Vec3b>(i, j) = GREEN;
      } else {
        square.at<Vec3b>(i, j) = WHITE;
      }
    }
  imshow("Multi-color Square", square);
  ImageUtil::waitKey();
}

void testPrintInverseOfMatrix() {
  cout << "Original matrix: " << endl << MATRIX3X3 << endl << endl;
  const Mat inverted = MATRIX3X3.inv();
  cout << "Inverse: " << endl << inverted << endl;
  TerminalUtil::waitForUserInput();
}
// End of Lab 1

/**
 * LAB 2
 */
void testDisplayRGBSeparately() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat_<Vec3b> src = imread(abs_image_path, IMREAD_COLOR);

    const auto t1 = std::chrono::high_resolution_clock::now();

    const int height = src.rows;
    const int width = src.cols;

    Mat_<uchar> red(height, width);
    Mat_<uchar> green(height, width);
    Mat_<uchar> blue(height, width);

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

    ImageUtil::waitKey();
  }
}

void testDisplayRGBSeparatelyFast() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat_<Vec3b> src = imread(abs_image_path, IMREAD_COLOR);

    const auto t1 = std::chrono::high_resolution_clock::now();

    const int height = src.rows;
    const int width = src.cols;

    Mat_<uchar> red(height, width);
    Mat_<uchar> green(height, width);
    Mat_<uchar> blue(height, width);

    for (int i = 0; i < height; i++) {
      const uchar* col_ptr = src.ptr(i);
      for (int j = 0; j < width; j++) {
        const uchar* pixel = col_ptr;
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
    ImageUtil::waitKey();
  }
}

void testRGB2Gray() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    Mat src = imread(abs_image_path);

    Mat dst(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; i++) {
      for (int j = 0; j < src.cols; j++) {
        Vec3b pixel = src.at<Vec3b>(i, j);
        unsigned char B = pixel[0];
        unsigned char G = pixel[1];
        unsigned char R = pixel[2];

        dst.at<uchar>(i, j) = (B + G + R) / 3;
      }
    }

    imshow("Color", src);
    imshow("Grayscale", dst);

    ImageUtil::waitKey();
  }
}

Mat getGrayFromRGB(const Mat& src) {
  Mat dst = Mat(src.rows, src.cols, CV_8UC1);
  for (int i = 0; i < src.rows; i++) {
    const uchar* col_ptr = src.ptr(i);
    for (int j = 0; j < src.cols; j++) {
      const uchar* pixel = col_ptr;
      unsigned char B = pixel[0];
      unsigned char G = pixel[1];
      unsigned char R = pixel[2];

      // dst.at<uchar>(i, j) = (B + G + R) / 3;
      // Using the luminosity method for better results
      dst.at<uchar>(i, j) = 0.299 * R + 0.587 * G + 0.114 * B;
      col_ptr += 3;
    }
  }
  return dst;
}

void testRGB2GrayFast() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    Mat src = imread(abs_image_path);

    Mat dst = getGrayFromRGB(src);

    imshow("Color", src);
    imshow("Grayscale", dst);

    ImageUtil::waitKey();
  }
}

Mat getBinaryFromGray(const Mat& src, const uchar threshold) {
  Mat dst = Mat(src.rows, src.cols, CV_8UC1);
  for (int i = 0; i < src.rows; i++) {
    const uchar* col_ptr = src.ptr(i);
    for (int j = 0; j < src.cols; j++) {
      const uchar val = col_ptr[j];
      dst.at<uchar>(i, j) = val > threshold ? 255 : 0;
    }
  }
  return dst;
}

void testGray2Binary() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);

    uchar threshold;
    cout << "Please enter the threshold value (0-255): ";
    cin >> threshold;
    while (threshold <= 0 || threshold > 255) {
      cout << "Invalid threshold value. Please enter a value between 0 and "
              "255: ";
      cin >> threshold;
    }

    Mat dst = getBinaryFromGray(src, threshold);

    imshow("Grayscale", src);
    imshow("Binary", dst);

    ImageUtil::waitKey();
  }
}

std::vector<float> getNormalizedRGB(const uchar* pixel) {
  std::vector<float> rgb(3);
  rgb[0] = (float)pixel[0] / 255.0;
  rgb[1] = (float)pixel[1] / 255.0;
  rgb[2] = (float)pixel[2] / 255.0;
  return rgb;
}

void testRGB2HSV() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat_<Vec3b> src = imread(abs_image_path, IMREAD_COLOR);

    const auto t1 = std::chrono::high_resolution_clock::now();

    const int height = src.rows;
    const int width = src.cols;

    Mat_<uchar> H_norm(height, width);
    Mat_<uchar> S_norm(height, width);
    Mat_<uchar> V_norm(height, width);

    for (int i = 0; i < height; i++) {
      const uchar* col_ptr = src.ptr(i);
      for (int j = 0; j < width; j++) {
        const uchar* pixel = col_ptr;

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

    imshow("source", src);
    imshow("H", H_norm);
    imshow("S", S_norm);
    imshow("V", V_norm);
    ImageUtil::waitKey();
  }
}

bool isInside(const Mat& img, int i, int j) {
  /* Point p(i, j);
  if (p.inside(Rect(0, 0, img.cols, img.rows))) {
    std::cout << "Point is inside" << std::endl;
  } else {
    std::cout << "Point is outside" << std::endl;
  } */
  return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

void testIsInside() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat_<Vec3b> src = imread(abs_image_path, IMREAD_COLOR);

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

    TerminalUtil::waitForUserInput();
  }
}
// End of Lab 2

/**
 * LAB 3
 */

// Return the histogram of the image as an array of 256 elements
int* getHistogramOld(const Mat& img) {
  const int height = img.rows;
  const int width = img.cols;

  int* histogram = new int[256]();

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      const uchar val = img.at<uchar>(i, j);
      histogram[val]++;
    }
  }
  return histogram;
}

int* getHistogramOldFast(const Mat& img) {
  const int height = img.rows;
  const int width = img.cols;

  int* histogram = new int[256]();

  const uchar* lpSrc = img.data;
  const int w = (int)img.step;  // no dword alignment is done !!!
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      const uchar val = lpSrc[i * w + j];
      histogram[val]++;
    }
  return histogram;
}

// Using std::array instead of raw pointer, this way we don't have to worry
// about memory management
std::array<int, 256> getHistogram(const Mat& img) {
  const int height = img.rows;
  const int width = img.cols;

  std::array<int, 256> histogram{};
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      const int val = img.at<uchar>(i, j);
      histogram[val]++;
    }
  }
  return histogram;
}

std::array<int, 256> getHistogramFast(const Mat& img) {
  const int height = img.rows;
  const int width = img.cols;

  std::array<int, 256> histogram{};
  const uchar* lpSrc = img.data;
  const int w = (int)img.step;  // no dword alignment is done !!!
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      const int val = lpSrc[i * w + j];
      histogram[val]++;
    }
  return histogram;
}

float* getPDFOld(const int* histogram, int M) {
  float* pdf = new float[256]();
  for (int i = 0; i < 256; i++) {
    pdf[i] = (float)histogram[i] / M;
  }
  return pdf;
}

std::array<float, 256> getPDF(const std::array<int, 256>& histogram, int M) {
  std::array<float, 256> pdf{};
  for (int i = 0; i < 256; i++) {
    pdf[i] = (float)histogram[i] / M;
  }
  return pdf;
}

void showHistogram(const string& name, int* hist, const int hist_cols,
                   const int hist_height) {
  Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
  // constructs a white image

  // computes histogram maximum
  int max_hist = 0;
  for (int i = 0; i < hist_cols; i++)
    if (hist[i] > max_hist) max_hist = hist[i];

  double scale = 1.0;
  scale = (double)hist_height / max_hist;
  int baseline = hist_height - 1;
  for (int x = 0; x < hist_cols; x++) {
    Point p1 = Point(x, baseline);
    Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
    line(imgHist, p1, p2, CV_RGB(255, 0, 255));  // histogram bins
    // colored in magenta
  }
  imshow(name, imgHist);
}

void showHistogram(const string& name, const std::array<int, 256>& hist,
                   const int hist_cols, const int hist_height) {
  Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
  // constructs a white image

  // computes histogram maximum
  int max_hist = *std::max_element(hist.begin(), hist.end());

  double scale = 1.0;
  scale = (double)hist_height / max_hist;
  int baseline = hist_height - 1;
  for (int x = 0; x < hist_cols; x++) {
    Point p1 = Point(x, baseline);
    Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
    line(imgHist, p1, p2, CV_RGB(255, 0, 255));  // histogram bins
    // colored in magenta
  }
  imshow(name, imgHist);
}

void testCalcHist() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);

    auto t1 = std::chrono::high_resolution_clock::now();
    int* histogramOld = getHistogramOld(src);
    auto t2 = std::chrono::high_resolution_clock::now();
    // Compute the time difference [ms]
    cout << "(OLD) It took "
         << std::chrono::duration<double, std::milli>(t2 - t1) << endl;

    t1 = std::chrono::high_resolution_clock::now();
    int* histogramOldFast = getHistogramOldFast(src);
    t2 = std::chrono::high_resolution_clock::now();
    // Compute the time difference [ms]
    cout << "(OLD FAST) It took "
         << std::chrono::duration<double, std::milli>(t2 - t1) << endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::array<int, 256> histogramNew = getHistogram(src);
    t2 = std::chrono::high_resolution_clock::now();
    // Compute the time difference [ms]
    cout << "(NEW) It took "
         << std::chrono::duration<double, std::milli>(t2 - t1) << endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::array<int, 256> histogramNewFast = getHistogramFast(src);
    t2 = std::chrono::high_resolution_clock::now();
    // Compute the time difference [ms]
    cout << "(NEW FAST) It took "
         << std::chrono::duration<double, std::milli>(t2 - t1) << endl;

    showHistogram("Histogram Old", histogramOld, 256, 200);
    showHistogram("Histogram Old Fast", histogramOldFast, 256, 200);
    showHistogram("Histogram New", histogramNew, 256, 200);
    showHistogram("Histogram New Fast", histogramNewFast, 256, 200);

    delete[] histogramOld;
    delete[] histogramOldFast;

    ImageUtil::waitKey();
  }
}

std::vector<int> getHistogramWithBins(const cv::Mat& img, int m) {
  const int height = img.rows;
  const int width = img.cols;

  std::vector<int> histogram(m, 0);

  const uchar* lpSrc = img.data;
  const int w = (int)img.step;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      const int val = lpSrc[i * w + j];
      const int bin = (val * m) / 256;
      histogram[bin]++;
    }
  }

  return histogram;
}

std::vector<int> getLocalHistogramMaxima(const std::array<float, 256>& pdf,
                                         const int windowHalfWidth = 5,
                                         const float threshold = 0.0003f) {
  std::vector<int> maxima;

  for (int k = windowHalfWidth; k < 256 - windowHalfWidth; k++) {
    float sum = 0.0f;
    bool isMaximum = true;

    for (int offset = -windowHalfWidth; offset <= windowHalfWidth; offset++) {
      const float value = pdf[k + offset];
      sum += value;
      if (pdf[k] < value) {
        isMaximum = false;
      }
    }

    const float average = sum / static_cast<float>(2 * windowHalfWidth + 1);
    if (isMaximum && pdf[k] > average + threshold) {
      maxima.push_back(k);
    }
  }

  maxima.insert(maxima.begin(), 0);
  maxima.push_back(255);
  maxima.erase(std::unique(maxima.begin(), maxima.end()), maxima.end());

  return maxima;
}

std::array<uchar, 256> buildNearestMaximumLookup(
    const std::vector<int>& maxima) {
  std::array<uchar, 256> lookup{};

  for (int grayLevel = 0; grayLevel < 256; grayLevel++) {
    int nearestMaximum = maxima.front();
    int minDistance = std::abs(grayLevel - nearestMaximum);

    for (size_t index = 1; index < maxima.size(); index++) {
      const int currentMaximum = maxima[index];
      const int currentDistance = std::abs(grayLevel - currentMaximum);
      if (currentDistance < minDistance) {
        minDistance = currentDistance;
        nearestMaximum = currentMaximum;
      }
    }

    lookup[grayLevel] = static_cast<uchar>(nearestMaximum);
  }

  return lookup;
}

Mat getMultiThresholdedFromGray(const Mat& src,
                                const std::vector<int>& maxima) {
  Mat dst(src.rows, src.cols, CV_8UC1);
  const std::array<uchar, 256> lookup = buildNearestMaximumLookup(maxima);

  for (int i = 0; i < src.rows; i++) {
    const uchar* srcRow = src.ptr(i);
    uchar* dstRow = dst.ptr(i);
    for (int j = 0; j < src.cols; j++) {
      dstRow[j] = lookup[srcRow[j]];
    }
  }

  return dst;
}

void testMultiLevelThresholding() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    constexpr int windowHalfWidth = 5;
    constexpr float threshold = 0.0003f;

    const Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);
    const std::array<int, 256> histogram = getHistogramFast(src);
    const std::array<float, 256> pdf = getPDF(histogram, src.rows * src.cols);
    const std::vector<int> maxima =
        getLocalHistogramMaxima(pdf, windowHalfWidth, threshold);
    const Mat dst = getMultiThresholdedFromGray(src, maxima);
    const std::array<int, 256> dstHistogram = getHistogramFast(dst);

    cout << "Detected maxima: ";
    for (const int maximum : maxima) {
      cout << maximum << ' ';
    }
    cout << endl;

    imshow("Grayscale", src);
    imshow("Multi-Level Thresholding", dst);
    showHistogram("Histogram", histogram, 256, 200);
    showHistogram("Multi-Level Thresholding Histogram", dstHistogram, 256, 200);

    ImageUtil::waitKey();
  }
}

void diffuseError(Mat& img, const int row, const int col,
                  const float errorFactor) {
  if (isInside(img, row, col)) {
    img.at<float>(row, col) =
        std::clamp(img.at<float>(row, col) + errorFactor, 0.0f, 255.0f);
  }
}

Mat getFloydSteinbergDitheredFromGray(const Mat& src,
                                      const std::vector<int>& maxima) {
  Mat dst(src.rows, src.cols, CV_8UC1);
  Mat work;
  src.convertTo(work, CV_32FC1);

  const std::array<uchar, 256> lookup = buildNearestMaximumLookup(maxima);

  for (int i = 0; i < work.rows; i++) {
    for (int j = 0; j < work.cols; j++) {
      const float oldPixel = work.at<float>(i, j);
      const int oldPixelIndex = std::clamp(cvRound(oldPixel), 0, 255);
      const uchar newPixel = lookup[oldPixelIndex];
      const float error = oldPixel - static_cast<float>(newPixel);

      work.at<float>(i, j) = static_cast<float>(newPixel);
      dst.at<uchar>(i, j) = newPixel;

      diffuseError(work, i, j + 1, 7.0f * error / 16.0f);
      diffuseError(work, i + 1, j - 1, 3.0f * error / 16.0f);
      diffuseError(work, i + 1, j, 5.0f * error / 16.0f);
      diffuseError(work, i + 1, j + 1, 1.0f * error / 16.0f);
    }
  }

  return dst;
}

void testFloydSteinbergDithering() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    constexpr int windowHalfWidth = 5;
    constexpr float threshold = 0.0003f;

    const Mat src = imread(abs_image_path, IMREAD_GRAYSCALE);
    const std::array<int, 256> histogram = getHistogramFast(src);
    const std::array<float, 256> pdf = getPDF(histogram, src.rows * src.cols);
    const std::vector<int> maxima =
        getLocalHistogramMaxima(pdf, windowHalfWidth, threshold);
    const Mat multiLevel = getMultiThresholdedFromGray(src, maxima);
    const Mat floydStein = getFloydSteinbergDitheredFromGray(src, maxima);

    imshow("Grayscale", src);
    imshow("Multi-Level Thresholding", multiLevel);
    imshow("Floyd-Steinberg Dithering", floydStein);

    ImageUtil::waitKey();
  }
}

// End of Lab 3

/**
 * LAB 4
 */

int getAreaSlow(const Mat& img, const Vec3b& color) {
  int area = 0;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if (img.at<Vec3b>(i, j) == color) {
        area++;
      }
    }
  }
  return area;
}

int getAreaFast(const Mat& img, const Vec3b& color) {
  int area = 0;
  for (int i = 0; i < img.rows; i++) {
    const Vec3b* row_ptr = img.ptr<Vec3b>(i);
    for (int j = 0; j < img.cols; j++) {
      if (row_ptr[j] == color) {
        area++;
      }
    }
  }
  return area;
}

Point2f getCenterOfMass(const Mat& img, const Vec3b& color, const int area) {
  Point2f centerOfMass(0, 0);
  for (int i = 0; i < img.rows; i++) {
    const Vec3b* row_ptr = img.ptr<Vec3b>(i);
    for (int j = 0; j < img.cols; j++) {
      if (row_ptr[j] == color) {
        centerOfMass.x += j;
        centerOfMass.y += i;
      }
    }
  }
  centerOfMass.x /= area;
  centerOfMass.y /= area;
  return centerOfMass;
}

void myCallBackFuncGeom(int event, int x, int y, int flags, void* param) {
  // More examples:
  // http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
  // Mat& src = *((Mat*)param);
  Mat& src = *static_cast<Mat*>(param);
  if (event == EVENT_LBUTTONDOWN) {
    // C style casting
    // Doesn't check at compile time
    // (int)(*src).at<Vec3b>(y, x)[2],
    // (int)(*src).at<Vec3b>(y, x)[1],
    // (int)(*src).at<Vec3b>(y, x)[0]);

    Vec3b pixel = src.at<Vec3b>(y, x);

    auto t1 = std::chrono::high_resolution_clock::now();
    int area = getAreaSlow(src, pixel);
    auto t2 = std::chrono::high_resolution_clock::now();
    // Compute the time difference [ms]
    cout << "(SLOW) It took "
         << std::chrono::duration<double, std::milli>(t2 - t1) << " ms" << endl;

    t1 = std::chrono::high_resolution_clock::now();
    area = getAreaFast(src, pixel);
    t2 = std::chrono::high_resolution_clock::now();
    // Compute the time difference [ms]
    cout << "(FAST) It took "
         << std::chrono::duration<double, std::milli>(t2 - t1) << " ms" << endl;

    Point2f centerOfMass = getCenterOfMass(src, pixel, area);

    // Using C++ static_cast, this checks at compile time
    cout << "" << endl;
    cout << "      Pos(x,y): " << x << "," << y << endl;
    cout << "    Color(RGB): " << static_cast<int>(pixel[2]) << ","
         << static_cast<int>(pixel[1]) << "," << static_cast<int>(pixel[0])
         << endl;
    cout << "          Area: " << area << " pixels" << endl;
    cout << std::fixed << std::setprecision(2) << "Center of Mass: ("
         << centerOfMass.x << "," << centerOfMass.y << ")" << endl;
    cout << "" << endl;
  }
}

void testGeometricCalcs() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    // Important to make sure Mat is NOT const!
    Mat src = imread(abs_image_path, IMREAD_COLOR);

    std::string windowName = "Source";

    imshow(windowName, src);
    setMouseCallback(windowName, myCallBackFuncGeom, &src);

    ImageUtil::waitKey();
  }
}

int main() {
  int op;
  do {
    destroyAllWindows();
#ifdef __APPLE__
    waitKey(1);
#endif
    TerminalUtil::clearScreen();
    printf("Menu:\n");
    printf("  1 - Open image\n");
    printf("  2 - Open BMP images from folder\n");
    printf("  3 - Image negative\n");
    printf("  4 - Image negative (fast)\n");
    printf(" 17 - Image negative (parallel)\n");
    printf(" 18 - Image negative (UMat)\n");
    printf("  5 - BGR->Gray\n");
    printf("  6 - BGR->Gray (fast, save result to disk) \n");
    printf("  7 - BGR->HSV\n");
    printf("  8 - Resize image\n");
    printf("  9 - Canny edge detection\n");
    printf(" 10 - Edges in a video sequence\n");
    printf(" 11 - Snap frame from live video\n");
    printf(" 12 - Mouse callback demo\n");
    printf(" 13 - Additive\n");
    printf(" 14 - Multiplicative\n");
    printf(" 15 - Four squares\n");
    printf(" 16 - Inverse\n");
    printf(" 21 - Display R, G, B separately\n");
    printf(" 22 - Display R, G, B separately Fast\n");
    printf(" 23 - RGB -> Grayscale\n");
    printf(" 27 - RGB -> Grayscale (fast)\n");
    printf(" 24 - Grayscale -> Binary with threshold from stdin\n");
    printf(" 25 - RGB -> HSV\n");
    printf(" 26 - isInside\n");
    printf(" 31 - Show Histogram\n");
    printf(" 32 - Multi-Level Thresholding\n");
    printf(" 33 - Floyd-Steinberg Dithering\n");
    printf(" 41 - Test Geometric Calcs\n");
    printf("  0 - Exit\n\n");
    printf("Option: ");
    cin >> op;
    switch (op) {
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
      case 17:
        testNegativeImageParallel();
        break;
      case 18:
        testNegativeImageUnifiedMat();
        break;
      case 5:
        testColor2Gray();
        break;
      case 6:
        testImageOpenAndSave();
        break;
      case 7:
        testBGR2HSV();
        break;
      case 8:
        testResize();
        break;
      case 9:
        testCanny();
        break;
      case 10:
        testVideoSequence();
        break;
      case 11:
        testSnap();
        break;
      case 12:
        testMouseClick();
        break;
      case 13:
        testChangeGrayLevelsAdditive();
        break;
      case 14:
        testChangeGrayLevelsMultiplicative();
        break;
      case 15:
        testDrawFourSquare();
        break;
      case 16:
        testPrintInverseOfMatrix();
        break;
      case 21:
        testDisplayRGBSeparately();
        break;
      case 22:
        testDisplayRGBSeparatelyFast();
        break;
      case 23:
        testRGB2Gray();
        break;
      case 27:
        testRGB2GrayFast();
        break;
      case 24:
        testGray2Binary();
        break;
      case 25:
        testRGB2HSV();
        break;
      case 26:
        testIsInside();
        break;
      case 31:
        testCalcHist();
        break;
      case 32:
        testMultiLevelThresholding();
        break;
      case 33:
        testFloydSteinbergDithering();
        break;
      case 41:
        testGeometricCalcs();
        break;
    }
  } while (op != 0);
  return 0;
}