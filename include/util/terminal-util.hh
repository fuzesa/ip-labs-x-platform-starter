#ifndef TERMINAL_UTIL_HH
#define TERMINAL_UTIL_HH

#include <cstdlib>

class TerminalUtil {
 private:
  TerminalUtil() = default;
  ~TerminalUtil() = default;

 public:
  static void clearScreen();
};

#endif  // TERMINAL_UTIL_HH