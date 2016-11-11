FolderBrowser
=============

Description
-----------
This project provides a GUI for visualizing data acquired with the [matlab-qd framework](https://github.com/qdev-dk/matlab-qd) by Anders Jellinggaard.
The gui itself is a Python implementation of the [gui from matlab-qd](https://github.com/qdev-dk/matlab-qd/tree/master/%2Bqd/%2Bgui).

Installation
------------
All packages used for FolderBrowser are included in the Anaconda distribution. Get it from [https://www.continuum.io/downloads](https://www.continuum.io/downloads) with a Python version >=3.5.

If you already have Anaconda already installed (with a Python version >=3.5) but the packages are not up to date, simply run
```
conda update anaconda
````
from the terminal to update the packages.

When you have all packages installed run `example.py` in the `examples` directory.


Requirements
------------
* Python 3+ (tested with version 3.5)
* Matplotlib (tested with version 1.5.3)
* PyQt 5 (tested with version 5.6)
* Numpy (tested with version 1.11)


Optional packages
-----------------
* Pandas (improves loading times by a factor 2-10x, tested with version 0.18.1)