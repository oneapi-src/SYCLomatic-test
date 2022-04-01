//===---helper_report.h--------- --------------------------------*- C++ -*---===//
////
//// Copyright (C) Intel Corporation. All rights reserved.
////
//// The information and source code contained herein is the exclusive
//// property of Intel Corporation and may not be disclosed, examined
//// or reproduced in whole or in part without explicit written authorization
//// from the company.
////
////===-----------------------------------------------------------------===//
#include<string>
#include<iostream>

class Report
{
private:
    static void print(const std::string& des) {
      std::cout << des;
    }
    static void printErr(const std::string& des) {
      std::cerr << des;
    }
public:
    static void info(const std::string& msg) {
      print(std::string("INFO:: ") + msg + "\n");
    }
    static void fail(const std::string& msg) {
      printErr(std::string("FAIL:: ") + msg + "\n");
    }
    static void pass(const std::string& msg) {
      print(std::string("PASS:: ") + msg + "\n");
    }
    Report(/* args */);
    ~Report();
};

Report::Report(/* args */)
{
}

Report::~Report()
{
}
