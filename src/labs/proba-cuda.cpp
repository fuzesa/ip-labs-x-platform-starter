#include "proba-cuda.hh"

void utcn::ip::ProbaCUDA::runLab() {
  int op;
  do {
    printMenu(MENU);
    std::cin >> op;
    switch (op) {
      case 0:
        break;
      case 1:
        testCudaSample();
        break;
      default:
        std::cout << "Invalid selection" << std::endl;
    }
  } while (op != 0);
}

void utcn::ip::ProbaCUDA::testCudaSample() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  if (!abs_image_path.empty()) {
    const cv::Mat src_host = cv::imread(abs_image_path, cv::IMREAD_GRAYSCALE);
    cv::imshow("Original Grayscale", src_host);

    cv::cuda::GpuMat dst, src;
    src.upload(src_host);

    cv::cuda::threshold(src, dst, 128.0, 255.0, cv::THRESH_BINARY);

    cv::Mat result_host;
    dst.download(result_host);

    cv::imshow("Result Processed in GPU", result_host);
    cv::waitKey();
  }
}