# Corner Detection and Stereo Camera Calibration

I compared the stereo camera calibrations between the OpenCV and the MATLAB. The two differences what I thought are the corner detection and the parameter of the optimization process. Unfortunately, I'm not sure about the last one. 

Firstly, the MATLAB uses the corner detection algorithm from "Automatic camera and range sensor calibration using a single shot". On the other hand, the calibration lib on OpenCV uses the traditional method to deal with the corner detection problems. Therefore, I decided to implement this algorithm using C++ with OpenCV to control them in order to have the same corner detection. After that, I can decide whether the optimization is the reason to decide their differences.

According to the tests, the calibration time using OpenCV reduced about 20%. Unfortunately, I think that though the optimization algorithm is the same in both two calibration tools, the parameters and the termination condition are not the same. 

The code is compiled in Ubuntu 16.04 with OpenCV 3.1.0


# Reference
Geiger A, Moosmann F, Car Ö, et al. Automatic camera and range sensor calibration using a single shot[C]//Robotics and Automation (ICRA), 2012 IEEE International Conference on. IEEE, 2012: 3936-3943.

https://github.com/onlyliucat/Multi-chessboard-Corner-extraction-detection-

https://github.com/qibao77/cornerDetect
