// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com
#include "file-util.hh"

#include "nfd.hpp"

std::string FileUtil::getFileOrDir(const bool isFile) {
  NFD::Guard nfdGuard;
  NFD::UniquePath outPath;
  nfdresult_t result;
  if (isFile) {
    result = OpenDialog(outPath);
  } else {
    result = PickFolder(outPath);
  }
  if (result == NFD_OKAY) {
    return outPath.get();
  }
  return "";
}

std::string FileUtil::getSingleFileAbsPath() { return getFileOrDir(); }

std::string FileUtil::getDirectoryAbsPath() { return getFileOrDir(false); }

std::vector<std::string> FileUtil::getAllFilesInDirectory() {
  std::vector<std::string> filenames;
  const std::string dir_abs_path = getDirectoryAbsPath();
  if (!dir_abs_path.empty()) {
    const fs::path path_to_traverse(dir_abs_path);
    if (fs::exists(path_to_traverse) && fs::is_directory(path_to_traverse)) {
      for (const auto &entry : fs::directory_iterator(path_to_traverse)) {
        auto abs_file_path = entry.path().string();
        filenames.push_back(abs_file_path);
      }
    }
  }
  return filenames;
}