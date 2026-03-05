#include "terminal-util.hh"

void TerminalUtil::clearScreen() {
#if defined(_WIN32) || defined(_WIN64)
  std::system("cls");
#else
  // POSIX
  std::system("clear");
#endif
}

void TerminalUtil::waitForUserInput() {
#if defined(_WIN32) || defined(_WIN64)
  std::system("pause");
#else
  // Print a prompt and wait for Enter. Use ignore to discard any leftover input.
  std::cout << "Press Enter to continue...";
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::cin.get();
#endif
}
