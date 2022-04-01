#ifndef REPORT_H
#define REPORT_H

#include <iostream>
#include <string>

class Report {
private:
  static uint32_t failCount;
  static void out(std::string msg) {
    std::cout << msg;
  }
public:
  static void start(std::string testName) {
    failCount = 0;
    out(std::string("TEST:: ") + testName + "\n");
  }
  static int finish() {
    if (failCount == 0) {
      out("PASSED\n");
      return 0;
    } else {
      out(std::string("FAILED: ") + std::to_string(failCount) + " failure(s) detected\n");
      return 1;
    }
  }
  static void fail(std::string msg) {
    failCount++;
    out(std::string("FAIL:: ") + msg + "\n");
  }
  static void pass(std::string msg) {
    out(std::string("PASS:: ") + msg + "\n");
  }
  static void info(std::string msg) {
    out(std::string("INFO:: ") + msg + "\n");
  }
  template<typename T1, typename T2>
  static void check(std::string descr, const T1 &value, const T2 &valueExpected) {
    if (value != valueExpected) {
      fail(descr);
    }
    else {
      pass(descr);
    }
  }
};
uint32_t Report::failCount;
#endif
