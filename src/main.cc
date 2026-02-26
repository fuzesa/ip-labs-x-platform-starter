#include <chrono>
#include <iostream>


#include "file-util.hh"
#include "image-util.hh"
#include "terminal-util.hh"

using namespace cv;
using namespace std;

void testOpenImage() {
  const string abs_image_path = FileUtil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const Mat src = imread(abs_image_path);
    imshow("image", src);
    ImageUtil::waitKey();
  }
}

int main() {
  testOpenImage();
  return 0;
}
