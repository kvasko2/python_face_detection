# Python Webcam Face Augmentation Test

Quick test project for doing face and face landmark detection (using great libraries others have built).


## Rough Setup Instructions

Start by setting up opencv. [This page](http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/) provides a helpful guide on Mac OSX.

The rough setup instructions I could piece together from my terminal history.

~~~~
brew install boost OR brew upgrade boost
brew install boost-python

brew install opencv3 --with-contrib

pip install numpy
pip install scipy OR sudo pip install --ignore-installed scipy
pip install scikit-image
pip install dlib

cd tests/dlib
python setup.py install

python face_landmark_detection.py
~~~~

## Run

```python webcam_landmark_front.py```

## Other resources

[This article](http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/) is very helpful for setting up OpenCV on Mac OS X and dealing with some of the errors that can come up.
