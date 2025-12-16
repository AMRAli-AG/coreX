
1. Old Library Versions

Raspberry Pi OS has old numpy (1.19).

Your code needs numpy ≥ 1.22 for pandas, scikit-learn, etc.

Effect: ImportError or version mismatch.

2. ARM Architecture / Missing Compilers

Packages like scipy or numpy try to compile from source.

Compilation needs gfortran, BLAS, LAPACK, which may not be installed.

Effect: Pip fails to build packages.

3. Python Version Limit

Your Pi has Python 3.9.

Latest versions of numpy, tensorflow, etc., need Python 3.10+.

Effect: Cannot install newest versions.

4. Heavy Builds Fail

Limited CPU and memory of Pi 4 can cause errors when building large packages.

Effect: Errors like AttributeError: module 'gast' has no attribute 'MatchStar'.
