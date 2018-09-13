#ifndef DETECTOR_H
#define DETECTOR_H

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
using std::vector;

struct Corners
{
	std::vector<cv::Point2f> p;
	std::vector<cv::Vec2f> v1;
	std::vector<cv::Vec2f> v2;
	std::vector<float> score;
} ;

enum ConvolutionType {
	/* Return the full convolution, including border */
	CONVOLUTION_FULL,

	/* Return only the part that corresponds to the original image */
	CONVOLUTION_SAME,
	/* Return only the submatrix containing elements that were not influenced by the border */
	CONVOLUTION_VALID
};

class FindCorners
{
public:
	FindCorners();
	FindCorners(Mat img);

	~FindCorners();

public:
	void detectCorners(Mat &Src, vector<Point2f> &resultCorners, Corners& mcorners, float scoreThreshold);
	void savecorners(Corners& mcorners, char * filename);
	void readcorners(Corners& mcorners, char * filename);
	static 	Mat conv2(const cv::Mat &img, const cv::Mat& ikernel, ConvolutionType type);
private:
	//正态分布
	float normpdf(float dist, float mu, float sigma);
	//获取最小值
	void getMin(Mat src1, Mat src2, Mat &dst);
	//获取最大值
	void getMax(Mat src1, Mat src2, Mat &dst);
	//获取梯度角度和权重
	void getImageAngleAndWeight(Mat img, Mat &imgDu, Mat &imgDv, Mat &imgAngle, Mat &imgWeight);
	//estimate edge orientations
	void edgeOrientations(Mat imgAngle, Mat imgWeight,int index);
	//find modes of smoothed histogram
	void findModesMeanShift(vector<float> hist, vector<float> &hist_smoothed, vector<pair<float, int>> &modes, float sigma);
	//score corners
	void scoreCorners(Mat img, Mat imgAngle, Mat imgWeight, vector<Point2f> &corners, vector<int> radius, vector<float> &score);
	//compute corner statistics
	void cornerCorrelationScore(Mat img, Mat imgWeight, vector<Point2f> cornersEdge, float &score);
	//亚像素精度找角点
	void refineCorners(vector<Point2f> &corners, Mat imgDu, Mat imgDv, Mat imgAngle, Mat imgWeight, float radius);
	//生成核
	void createkernel(float angle1, float angle2, int kernelSize, Mat &kernelA, Mat &kernelB, Mat &kernelC, Mat &kernelD);
	//非极大值抑制
	void nonMaximumSuppression(Mat& inputCorners, vector<Point2f>& outputCorners, float threshold, int margin, int patchSize);
	float norm2d(cv::Point2f o);
	Mat structureRecovery(Corners& mcorners);
	bool findValue(const cv::Mat &mat, float value);
	Mat initChessboard(Corners& mcorners,int idx);
	int directionalNeighbor(int idx, Vec2f v, Mat chessboard, Corners& corners,int& neighbor_idx, float& min_dist);
	//float mean(vector<float> distance);
	float mean_l(std::vector<float> &resultSet);
	float distv(cv::Vec2f& a, cv::Vec2f &b);
	float stdev_l(std::vector<float> &resultSet, float &mean);
	float stdevmean(std::vector<float> &resultSet);
	//float deviation(vector<float>distance);	
	float chessboardEnergy(Mat chessboard,Corners& corners);
	Mat growChessboard(Mat chessboard, Corners& corners, int border_type);
	bool is_element_in_vector(vector<float> v,int element);
	vector<Point2f> predictCorners(std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2, 
	std::vector<cv::Point2f>& p3, std::vector<cv::Point2f> &pred);
	void assignClosestCorners(std::vector<cv::Point2f>&cand, std::vector<cv::Point2f>&pred, std::vector<int> &idx );

private:
	vector<Point2f> templateProps;
	vector<int> radius;
	vector<Point2f> cornerPoints;
	std::vector<std::vector<float> > cornersEdge1;
	std::vector<std::vector<float> > cornersEdge2;
	std::vector<cv::Point* > cornerPointsRefined;

};

#endif // DETECTOR_H
