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
  for (const auto &abs_file_path : abs_file_paths) {
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

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int num = 0; num < 20000; num++) {
      src.forEach<uchar>(
          [](uchar &curr, const int *position) -> void { curr = 255 - curr; });
    }

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Compute the time difference [ms]
    cout << "It took " << std::chrono::duration<double, std::milli>(t2 - t1)
         << endl;

    imshow("input image", src);
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
  const string path_to_src = ASSETS_DIR "Images/Lena_24bits.bmp";
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

  const auto *WIN_DST = "Dst";  // window for the destination (processed) image
  namedWindow(WIN_DST, WINDOW_AUTOSIZE);
  moveWindow(WIN_DST, src_size.width + 10, 0);

  cvtColor(src, dst,
           COLOR_BGR2GRAY);  // converts the source image to a grayscale one

  const string path_to_dst = ASSETS_DIR "Images/Lena_24bits_gray.bmp";
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
    uchar *lpH = H.data;
    uchar *lpS = S.data;
    uchar *lpV = V.data;

    Mat hsvImg;
    cvtColor(src, hsvImg, COLOR_BGR2HSV);

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
      sprintf(numberStr, "%d", frameCount);
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

void myCallBackFunc(int event, int x, int y, int flags, void *param) {
  // More examples:
  // http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
  auto *src = static_cast<Mat *>(param);
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

Mat changeByFactor(const Mat &orig_pic, const bool isAdditive,
                   const uchar factor) {
  const int height = orig_pic.rows;
  const int width = orig_pic.cols;
  Mat dst = Mat(height, width, CV_8UC1);

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
    ImageUtil::waitKey();
  }
}

void testRGB2Gray() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat_<Vec3b> src = imread(abs_image_path, IMREAD_COLOR);

    const auto t1 = std::chrono::high_resolution_clock::now();

    const int height = src.rows;
    const int width = src.cols;

    Mat_<uchar> gray(height, width);

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
    ImageUtil::waitKey();
  }
}

void testGray2Binary() {
  const std::string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    Mat_<uchar> src = imread(abs_image_path, IMREAD_GRAYSCALE);
    int threshold;
    std::cout << "Please enter the value for the threshold: ";
    std::cin >> threshold;
    Mat_<uchar> dst(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
      for (int j = 0; j < src.cols; j++) {
        dst(i, j) = src(i, j) > threshold ? 255 : 0;
      }
    }
    imshow("Source", src);
    imshow("B & W", dst);
    ImageUtil::waitKey();
  }
}

std::vector<float> getNormalizedRGB(const uchar *pixel) {
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

    imshow("source", src);
    imshow("H", H_norm);
    imshow("S", S_norm);
    imshow("V", V_norm);
    ImageUtil::waitKey();
  }
}

bool isInside(const Mat &img, int i, int j) {
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
    printf(" 24 - Grayscale -> Binary with threshold from stdin\n");
    printf(" 25 - RGB -> HSV\n");
    printf(" 26 - isInside\n");
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
      case 24:
        testGray2Binary();
        break;
      case 25:
        testRGB2HSV();
        break;
      case 26:
        testIsInside();
        break;
    }
  } while (op != 0);
  return 0;
}