# Contributing

## License
The code in this repo is licensed under the terms in [LICENSE](LICENSE). By contributing to this project, you agree to the Apache License v2.0 with LLVM Exceptions and copyright terms therein and release your contribution under these terms.

## Contribution process

### Development

- Create a personal fork for this project on Github.
- Prepare your patch:
  - follow [LLVM coding standards](https://llvm.org/docs/CodingStandards.html).
  - Use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) tools to help you ensure formatting and stylistic compliance.

### Testing
See [README.md](https://github.com/oneapi-src/SYCLomatic-test/blob/SYCLomatic/README.md) for instructions.

### Review and acceptance testing
- Create a pull request for your changes following [Creating a pull request instructions](https://help.github.com/articles/creating-a-pull-request/) to create your change.
- Changes addressing comments made during code review should be added as a separate commits to the same PR.
- CI test will run checks which are prerequisites for submitting PR:
  - jenkins/c2s-ci-linux - runs all related tests on the Ubuntu* 20.04 machine on GPU device (Level_zero backend) and CPU device (OpenCL backend).
  - jenkins/c2s-ci-windows - runs all related tests on the Windows* 2019 machine on GPU device (Level_zero backend) and CPU device (OpenCL backend).

The CI checks are tested with the latest nightly build for DPC++ compiler and runtime.
When all CI testing passed and PR review is approved, the pull request is ready to merge.

### Merge
Click the "Squash and merge" to merge your pull requests. Add the PR description if needed.


## Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.


## Trademarks information
Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.<br>
\*Other names and brands may be claimed as the property of others. SYCL is a trademark of the Khronos Group Inc.
