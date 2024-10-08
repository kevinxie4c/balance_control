# Install

Install Perl packages
```
cpan ExtUtils::PkgConfig PDL GD File::Slurp JSON
```

Install MXNet and Perl binding following the instructions [here](https://mxnet.apache.org/versions/1.5.0/install/).

Install [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page), [DARTSIM](https://dartsim.github.io/), [JSON for Modern C++](https://github.com/nlohmann/json).

Build the Perl binding for the simuation environment
```
    cd CharacterEnv
    perl Makefile.PL
    make
```
If you get warnings such as `Cannot find dart. Please define DART_INSTALL_PREFIX`, set the corresponding variable to the correct path when running `perl Makefile.PL`, for example:
```
    cd CharacterEnv
    DART_INSTALL_PREFIX=/path/to/dart/install/base perl Makefile.PL
    make
```
