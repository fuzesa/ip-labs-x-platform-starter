#ifndef TERMINAL_UTIL_HH
#define TERMINAL_UTIL_HH

#include <cstdlib>
#include <iostream>
#include <limits>

class TerminalUtil {
 private:
  TerminalUtil() = default;
  ~TerminalUtil() = default;

 public:
  static void clearScreen();
  static void waitForUserInput();
};

#endif  // TERMINAL_UTIL_HH
