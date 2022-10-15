# CPnP-cpp-version
A cpp implementation of "CPnP: Consistent Pose Estimator for Perspective-n-Point Problem with Bias Elimination"

# Related Publications
CPnP:
- **CPnP: Consistent Pose Estimator for Perspective-n-Point Problem with Bias Elimination**, G. Zeng, S. Chen, B. Mu, G. Shi, and J. Wu, arXiv:2209.05824, 2022
# Requirements 
## Eigen3
```
sudo apt install libeigen3-dev
```
# Installation
Clone the repository
```
git clone https://github.com/bereze/CPnP-cpp-version.git
```
Build and install
```
cd CPnP-cpp-version
mkdir build && cd build
cmake ..
make -j4
sudo make install
```
