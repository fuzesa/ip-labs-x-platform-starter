#ifndef FILE_UTIL_HH
#define FILE_UTIL_HH

#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

class FileUtil {
 private:
  FileUtil() = default;
  ~FileUtil() = default;
  static inline std::string DEFAULT_IMAGES_PATH = ASSETS_DIR "Images/*";
  static std::string getFileOrDir(bool isFile = true);

 public:
  static std::string getSingleFileAbsPath();
  static std::string getDirectoryAbsPath();
  static std::vector<std::string> getAllFilesInDirectory();
};

#endif  // FILE_UTIL_HH
