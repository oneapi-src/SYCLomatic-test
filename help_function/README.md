This folder will contain a test folder for each release, each folder contains a full helper function test sets.
   eg. 2021.4: contains helper function test set for 2021.4 release
       gold  : contains helper function test set for gold release
       2022.0: contains helper function test set for 2022.0 release
      feel  free to create new for new release.

For development test, only latest test set is required.
   eg. if devloping 2021.4, then only test in 2021.4 is required to run for development test.

For Compatibility test, just run any test set against the helper header files you want to test.
   eg. run 2021.4 test against 2022.0's helper function to come out whether 2021.4 and 2022.0 are compatible.

For better orginize the test cases:
   a. Each helper file should contain at least one test file, and have helper file name included in test file name.
   b. Each test file should contain test info in file header to show what kind of features are tested.
     eg.  #cat ${helper_file}_test.cpp
        //test_feature:featurename   (for feature name, ref to the API or msg "// DPCT_LABEL_BEGIN|atomic_fetch_add|dpct" in helper files.
        //test_feature:featurename2
        //....
        //test_feature:featurenameN
        ....
        //test code
        ....


NOTE:
Take care when you try to modify any old test available in the old test set  make sure it can pass in old test
