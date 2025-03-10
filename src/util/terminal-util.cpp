// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com
#include "terminal-util.hh"

void TerminalUtil::clearScreen() {
#ifdef WINDOWS
  std::system("cls");
#else
  // Assume POSIX
  std::system("clear");
#endif
}

void TerminalUtil::waitForUserInput() {
#ifdef WINDOWS
  system("pause");
#else
  std::cin.get();
  std::cin.get();
#endif
}
