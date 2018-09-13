

#include "DetectCorners.h"
#include <fstream>
#include <limits>
#include <numeric>

FindCorners::FindCorners()
{
}
FindCorners::~FindCorners()
{
}
FindCorners::FindCorners(Mat img)
{
	radius.push_back(4);
	radius.push_back(8);
	radius.push_back(12);

	templateProps.push_back(Point2f((float)0, (float)CV_PI / 2));
	templateProps.push_back(Point2f((float)CV_PI / 4, (float)-CV_PI / 4));
	templateProps.push_back(Point2f((float)0, (float)CV_PI / 2));
	templateProps.push_back(Point2f((float)CV_PI / 4, (float)-CV_PI / 4));
	templateProps.push_back(Point2f((float)0, (float)CV_PI / 2));
	templateProps.push_back(Point2f((float)CV_PI / 4, (float)-CV_PI / 4));
}

//正态分布
float FindCorners::normpdf(float dist, float mu, float sigma)
{
	return exp(-0.5 * (dist - mu) * (dist - mu) / (sigma * sigma)) / (std::sqrt(2 * CV_PI) * sigma);
}

//**************************生成核*****************************//
//angle代表核类型：45度核和90度核
//kernelSize代表核大小（最终生成的核的大小为kernelSize*2+1）
//kernelA...kernelD是生成的核
//*************************************************************************//
void FindCorners::createkernel(float angle1, float angle2, int kernelSize, Mat &kernelA, Mat &kernelB, Mat &kernelC, Mat &kernelD)
{

	int width = (int)kernelSize * 2 + 1;
	int height = (int)kernelSize * 2 + 1;
	kernelA = cv::Mat::zeros(height, width, CV_32F);
	kernelB = cv::Mat::zeros(height, width, CV_32F);
	kernelC = cv::Mat::zeros(height, width, CV_32F);
	kernelD = cv::Mat::zeros(height, width, CV_32F);

	for (int u = 0; u < width; ++u)
	{
		for (int v = 0; v < height; ++v)
		{
			float vec[] = {u - kernelSize, v - kernelSize};				  //相当于将坐标原点移动到核中心
			float dis = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);	 //相当于计算到中心的距离
			float side1 = vec[0] * (-sin(angle1)) + vec[1] * cos(angle1); //相当于将坐标原点移动后的核进行旋转，以此产生四种核
			float side2 = vec[0] * (-sin(angle2)) + vec[1] * cos(angle2); //X=X0*cos+Y0*sin;Y=Y0*cos-X0*sin
			if (side1 <= -0.1 && side2 <= -0.1)
			{
				kernelA.ptr<float>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1 && side2 >= 0.1)
			{
				kernelB.ptr<float>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 <= -0.1 && side2 >= 0.1)
			{
				kernelC.ptr<float>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1 && side2 <= -0.1)
			{
				kernelD.ptr<float>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
		}
	}
	//std::cout << "kernelA:" << kernelA << endl << "kernelB:" << kernelB << endl
	//	<< "kernelC:" << kernelC<< endl << "kernelD:" << kernelD << endl;
	//归一化
	kernelA = kernelA / cv::sum(kernelA)[0];
	kernelB = kernelB / cv::sum(kernelB)[0];
	kernelC = kernelC / cv::sum(kernelC)[0];
	kernelD = kernelD / cv::sum(kernelD)[0];
}
//**************************//获取最小值*****************************//
//*************************************************************************//
void FindCorners::getMin(Mat src1, Mat src2, Mat &dst)
{
	//src1和src2的大小要一样
	//if (src1.size() != src2.size())
	//{
	//	cout << "The size of matrix don't match" << endl;
	//}
	//dst = Mat::zeros(src1.size(), src1.type());
	//for (int i = 0; i < src1.rows; i++)
	//{
	//	for (int j = 0; j < src1.cols; j++)
	//	{
	//		dst.ptr<float>(i)[j] = src1.ptr<float>(i)[j] <= src2.ptr<float>(i)[j] ? src1.ptr<float>(i)[j] : src2.ptr<float>(i)[j];
	//	}
	//}
	int rowsLeft = src1.rows;
	int colsLeft = src1.cols;
	int rowsRight = src2.rows;
	int colsRight = src2.cols;
	if (rowsLeft != rowsRight || colsLeft != colsRight)
		return;

	int channels = src1.channels();

	int nr = rowsLeft;
	int nc = colsLeft;
	if (src1.isContinuous())
	{
		nc = nc * nr;
		nr = 1;
		//std::cout<<"continue"<<std::endl;
	}
	for (int i = 0; i < nr; i++)
	{
		const float *dataLeft = src1.ptr<float>(i);
		const float *dataRight = src2.ptr<float>(i);
		float *dataResult = dst.ptr<float>(i);
		for (int j = 0; j < nc * channels; ++j)
		{
			dataResult[j] = (dataLeft[j] < dataRight[j]) ? dataLeft[j] : dataRight[j];
		}
	}
}
//**************************//获取最大值*****************************//
//*************************************************************************//
void FindCorners::getMax(Mat src1, Mat src2, Mat &dst)
{
	//src1和src2的大小要一样
	//if (src1.size() != src2.size())
	//{
	//	cout << "The size of matrix don't match" << endl;
	//}
	//dst = Mat::zeros(src1.size(), src1.type());
	//for (int i = 0; i < src1.cols; i++)
	//{
	//	const float* dataLeft = src1.ptr<float>(i);
	//	const float* dataRight = src2.ptr<float>(i);
	//	float* dataResult = dst.ptr<float>(i);
	//	for (int j = 0; j < src1.rows; j++)
	//	{
	//		dataResult[j] = (dataLeft[j] >= dataRight[j]) ? dataLeft[j] : dataRight[j];
	//	}
	//}
	//(没搞明白，只是换了种写法就不行了，就只能进行一次最大值的获取了。。)
	int rowsLeft = src1.rows;
	int colsLeft = src1.cols;
	int rowsRight = src2.rows;
	int colsRight = src2.cols;
	if (rowsLeft != rowsRight || colsLeft != colsRight)
		return;

	int channels = src1.channels();

	int nr = rowsLeft;
	int nc = colsLeft;
	if (src1.isContinuous())
	{
		nc = nc * nr;
		nr = 1;
		//std::cout<<"continue"<<std::endl;
	}
	for (int i = 0; i < nr; i++)
	{
		const float *dataLeft = src1.ptr<float>(i);
		const float *dataRight = src2.ptr<float>(i);
		float *dataResult = dst.ptr<float>(i);
		for (int j = 0; j < nc * channels; ++j)
		{
			dataResult[j] = (dataLeft[j] >= dataRight[j]) ? dataLeft[j] : dataRight[j];
		}
	}
}
//获取梯度角度和权重
void FindCorners::getImageAngleAndWeight(Mat img, Mat &imgDu, Mat &imgDv, Mat &imgAngle, Mat &imgWeight)
{
	Mat sobelKernel(3, 3, CV_32F);
	Mat sobelKernelTrs(3, 3, CV_32F);
	//soble滤波器算子核
	sobelKernel.col(0).setTo(cv::Scalar(-1.0));
	sobelKernel.col(1).setTo(cv::Scalar(0.0));
	sobelKernel.col(2).setTo(cv::Scalar(1.0));

	sobelKernelTrs = sobelKernel.t();

	// 	imgDu = conv2(img, sobelKernel, CONVOLUTION_SAME);
	// 	imgDv = conv2(img, sobelKernelTrs, CONVOLUTION_SAME);

	filter2D(img, imgDu, CV_32F, sobelKernel);
	filter2D(img, imgDv, CV_32F, sobelKernelTrs);

	if (imgDu.size() != imgDv.size())
		return;

	cartToPolar(imgDu, imgDv, imgWeight, imgAngle, false);

	// for (int i = 0; i < imgDu.rows; i++)
	// {
	// 	float* dataDv = imgDv.ptr<float>(i);
	// 	float* dataDu = imgDu.ptr<float>(i);
	// 	float* dataAngle = imgAngle.ptr<float>(i);
	// 	float* dataWeight = imgWeight.ptr<float>(i);
	// 	for (int j = 0; j < imgDu.cols; j++)
	// 	{
	// 		dataAngle[j] = atan2((float)dataDv[j], (float)dataDu[j]);
	// 		if (dataAngle[j] < 0)dataAngle[j] = dataAngle[j] + CV_PI;
	// 		else if (dataAngle[j] > CV_PI)dataAngle[j] = dataAngle[j] - CV_PI;

	// 	//	dataWeight[j] = std::sqrt((float)dataDv[j] * (float)dataDv[j] + (float)dataDu[j] * (float)dataDu[j]);
	// 	}
	// }
	for (int i = 0; i < imgDu.rows; i++)
	{
		for (int j = 0; j < imgDu.cols; j++)
		{
			float *dataAngle = imgAngle.ptr<float>(i);
			if (dataAngle[j] < 0)
				dataAngle[j] = dataAngle[j] + CV_PI;
			else if (dataAngle[j] > CV_PI)
				dataAngle[j] = dataAngle[j] - CV_PI;
		}
	}
}
//**************************非极大值抑制*****************************//
//inputCorners是输入角点，outputCorners是非极大值抑制后的角点
//threshold是设定的阈值
//margin是进行非极大值抑制时检查方块与输入矩阵边界的距离，patchSize是该方块的大小
//*************************************************************************//
void FindCorners::nonMaximumSuppression(Mat &inputCorners, vector<Point2f> &outputCorners, float threshold, int margin, int patchSize)
{
	if (inputCorners.size <= 0)
	{
		cout << "The imput mat is empty!" << endl;
		return;
	}
	for (int i = margin + patchSize; i <= inputCorners.cols - (margin + patchSize + 1); i = i + patchSize + 1) //移动检查方块，每次移动一个方块的大小
	{
		for (int j = margin + patchSize; j <= inputCorners.rows - (margin + patchSize + 1); j = j + patchSize + 1)
		{
			float maxVal = inputCorners.ptr<float>(j)[i];
			int maxX = i;
			int maxY = j;
			for (int m = i; m <= i + patchSize; m++) //找出该检查方块中的局部最大值
			{
				for (int n = j; n <= j + patchSize; n++)
				{
					float temp = inputCorners.ptr<float>(n)[m];
					if (temp > maxVal)
					{
						maxVal = temp;
						maxX = m;
						maxY = n;
					}
				}
			}
			if (maxVal < threshold)
				continue; //若该局部最大值小于阈值则不满足要求
			int flag = 0;
			for (int m = maxX - patchSize; m < min(maxX + patchSize, inputCorners.cols - margin); m++) //二次检查
			{
				for (int n = maxY - patchSize; n < min(maxY + patchSize, inputCorners.rows - margin); n++)
				{
					if (inputCorners.ptr<float>(n)[m] > maxVal && (m < i || m > i + patchSize || n < j || n > j + patchSize))
					{
						flag = 1;
						break;
					}
				}
				if (flag)
					break;
			}
			if (flag)
				continue;
			outputCorners.push_back(Point(maxX, maxY));
			std::vector<float> e1(2, 0.0);
			std::vector<float> e2(2, 0.0);
			cornersEdge1.push_back(e1);
			cornersEdge2.push_back(e2);
		}
	}
}
int cmp(const pair<float, int> &a, const pair<float, int> &b)
{
	return a.first > b.first;
}

//find modes of smoothed histogram
void FindCorners::findModesMeanShift(vector<float> hist, vector<float> &hist_smoothed, vector<pair<float, int>> &modes, float sigma)
{
	//efficient mean - shift approximation by histogram smoothing
	//compute smoothed histogram
	bool allZeros = true;
	for (int i = 0; i < hist.size(); i++)
	{
		float sum = 0;
		for (int j = -(int)round(2 * sigma); j <= (int)round(2 * sigma); j++)
		{
			int idx = 0;
			// 			if ((i + j) < 0)idx = i + j + hist.size();
			// 			else if ((i + j) >= 32)idx = i + j - hist.size();
			// 			else idx = (i + j);//cornerDetect
			idx = (i + j) % hist.size();
			sum = sum + hist[idx] * normpdf(j, 0, sigma);
		}
		hist_smoothed[i] = sum;
		if (abs(hist_smoothed[i] - hist_smoothed[0]) > 0.0001)
			allZeros = false; // check if at least one entry is non - zero
							  //(otherwise mode finding may run infinitly)
	}
	if (allZeros)
		return;
	//mode finding
	for (int i = 0; i < hist.size(); i++)
	{
		int j = i;
		while (true)
		{
			float h0 = hist_smoothed[j];
			int j1 = (j + 1) % hist.size();
			int j2 = (j - 1) % hist.size();
			float h1 = hist_smoothed[j1];
			float h2 = hist_smoothed[j2];
			if (h1 >= h0 && h1 >= h2)
				j = j1;
			else if (h2 > h0 && h2 > h1)
				j = j2;
			else
				break;
		}
		bool ys = true;
		if (modes.size() == 0)
		{
			ys = true;
		}
		else
		{
			for (int k = 0; k < modes.size(); k++)
			{
				if (modes[k].second == j)
				{
					ys = false;
					break;
				}
			}
		}
		if (ys == true)
		{
			modes.push_back(std::make_pair(hist_smoothed[j], j));
		}
	}
	//for (int i = 0; i < hist.size(); i++)
	//{
	//	int j = i;
	//	while (true)
	//	{
	//		float h0 = hist_smoothed[j];
	//		int j1 = (j - 1)<0 ? j - 1 + hist.size() : j - 1;
	//		j1 = j>hist.size() ? j - 1 - hist.size() : j - 1;
	//		int j2 = (j + 1)>hist.size() - 1 ? j + 1 - hist.size() : j + 1;
	//		j2 = (j + 1)<0 ? j + 1 + hist.size() : j + 1;
	//		float h1 = hist_smoothed[j1];
	//		float h2 = hist_smoothed[j2];
	//		if (h1 >= h0&&h1 >= h2)j = j1;
	//		else if (h2 >= h0&&h2 >= h1)j = j2;
	//		else break;
	//	}
	//	if (modes.size() == 0 || modes[i].x!=(float)j)
	//	{

	//	}
	//}
	// 	for (int i = 0; i<hist.size(); ++i){
	// 		int j = i;
	// 		int curLeft = (j - 1)<0 ? j - 1 + hist.size() : j - 1;
	// 		int curRight = (j + 1)>hist.size() - 1 ? j + 1 - hist.size() : j + 1;
	// 		if (hist_smoothed[curLeft]<hist_smoothed[i] && hist_smoothed[curRight]<hist_smoothed[i]){
	// 			modes.push_back(std::make_pair(hist_smoothed[i], i));
	// 		}
	// 	}
	std::sort(modes.begin(), modes.end(), cmp);
}
//estimate edge orientations
void FindCorners::edgeOrientations(Mat imgAngle, Mat imgWeight, int index)
{
	//number of bins (histogram parameter)
	int binNum = 32;

	//convert images to vectors
	if (imgAngle.size() != imgWeight.size())
		return;
	vector<float> vec_angle, vec_weight;
	for (int i = 0; i < imgAngle.cols; i++)
	{
		for (int j = 0; j < imgAngle.rows; j++)
		{
			// convert angles from normals to directions
			float angle = imgAngle.ptr<float>(j)[i] + CV_PI / 2;
			angle = angle > CV_PI ? (angle - CV_PI) : angle;
			vec_angle.push_back(angle);

			vec_weight.push_back(imgWeight.ptr<float>(j)[i]);
		}
	}

	//create histogram
	vector<float> angleHist(binNum, 0);
	for (int i = 0; i < vec_angle.size(); i++)
	{
		int bin = max(min((int)floor(vec_angle[i] / (CV_PI / binNum)), binNum - 1), 0);
		angleHist[bin] = angleHist[bin] + vec_weight[i];
	}

	// find modes of smoothed histogram
	vector<float> hist_smoothed(angleHist);
	vector<std::pair<float, int>> modes;
	findModesMeanShift(angleHist, hist_smoothed, modes, 1);

	// if only one or no mode = > return invalid corner
	if (modes.size() <= 1)
		return;

	float fo[2];
	fo[0] = modes[0].second * (CV_PI / binNum);
	fo[1] = modes[1].second * (CV_PI / binNum);
	float deltaAngle = 0;
	if (fo[0] > fo[1])
	{
		float t = fo[0];
		fo[0] = fo[1];
		fo[1] = t;
	}

	deltaAngle = MIN(fo[1] - fo[0], fo[0] - fo[1] + (float)CV_PI);
	// if angle too small => return invalid corner
	if (deltaAngle <= 0.3)
		return;

	//set statistics: orientations
	cornersEdge1[index][0] = cos(fo[0]);
	cornersEdge1[index][1] = sin(fo[0]);
	cornersEdge2[index][0] = cos(fo[1]);
	cornersEdge2[index][1] = sin(fo[1]);

	//extract 2 strongest modes and compute orientation at modes
	// 	std::pair<float, int> most1 = modes[modes.size() - 1];
	// 	std::pair<float, int> most2 = modes[modes.size() - 2];
	// 	float most1Angle = most1.second*CV_PI / binNum;
	// 	float most2Angle = most2.second*CV_PI / binNum;
	// 	float tmp = most1Angle;
	// 	most1Angle = (most1Angle>most2Angle) ? most1Angle : most2Angle;
	// 	most2Angle = (tmp>most2Angle) ? most2Angle : tmp;
	//
	// 	// compute angle between modes
	// 	float deltaAngle = min(most1Angle - most2Angle, most2Angle + (float)CV_PI - most1Angle);
	//
	// 	// if angle too small => return invalid corner
	// 	if (deltaAngle <= 0.3)return;
	//
	// 	//set statistics: orientations
	// 	cornersEdge1[index][0] = cos(most1Angle);
	// 	cornersEdge1[index][1] = sin(most1Angle);
	// 	cornersEdge2[index][0] = cos(most2Angle);
	// 	cornersEdge2[index][1] = sin(most2Angle);
}

float FindCorners::norm2d(cv::Point2f o)
{
	return sqrt(o.x * o.x + o.y * o.y);
}
//亚像素精度找角点
void FindCorners::refineCorners(vector<Point2f> &corners, Mat imgDu, Mat imgDv, Mat imgAngle, Mat imgWeight, float radius)
{
	// image dimensions
	int width = imgDu.cols;
	int height = imgDu.rows;
	// for all corners do
	for (int i = 0; i < corners.size(); i++)
	{
		//extract current corner location
		int cu = corners[i].x;
		int cv = corners[i].y;
		// estimate edge orientations
		int startX, startY, ROIwidth, ROIheight;
		startX = max(cu - radius, (float)0);
		startY = max(cv - radius, (float)0);
		ROIwidth = min(cu + radius + 1, (float)width - 1) - startX;
		ROIheight = min(cv + radius + 1, (float)height - 1) - startY;

		Mat roiAngle, roiWeight;
		roiAngle = imgAngle(Rect(startX, startY, ROIwidth, ROIheight));
		roiWeight = imgWeight(Rect(startX, startY, ROIwidth, ROIheight));
		edgeOrientations(roiAngle, roiWeight, i);

		// continue, if invalid edge orientations
		if (cornersEdge1[i][0] == 0 && cornersEdge1[i][1] == 0 || cornersEdge2[i][0] == 0 && cornersEdge2[i][1] == 0)
			continue;

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//% corner orientation refinement %
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		cv::Mat A1 = cv::Mat::zeros(cv::Size(2, 2), CV_32F);
		cv::Mat A2 = cv::Mat::zeros(cv::Size(2, 2), CV_32F);

		for (int u = startX; u < startX + ROIwidth; u++)
			for (int v = startY; v < startY + ROIheight; v++)
			{
				// pixel orientation vector
				cv::Point2f o(imgDu.at<float>(v, u), imgDv.at<float>(v, u));
				float no = norm2d(o);
				if (no < 0.1)
					continue;
				o = o / no;
				// robust refinement of orientation 1
				float t0 = abs(o.x * cornersEdge1[i][0] + o.y * cornersEdge1[i][1]);
				if (t0 < 0.25) // inlier ?
				{
					Mat addtion(1, 2, CV_32F);
					addtion.col(0).setTo(imgDu.at<float>(v, u));
					addtion.col(1).setTo(imgDv.at<float>(v, u));
					Mat addtionu = imgDu.at<float>(v, u) * addtion;
					Mat addtionv = imgDv.at<float>(v, u) * addtion;
					for (int j = 0; j < A1.cols; j++)
					{
						A1.at<float>(0, j) = A1.at<float>(0, j) + addtionu.at<float>(0, j);
						A1.at<float>(1, j) = A1.at<float>(1, j) + addtionv.at<float>(0, j);
					}
				}
				// robust refinement of orientation 2
				float t1 = abs(o.x * cornersEdge2[i][0] + o.y * cornersEdge2[i][1]);
				if (t1 < 0.25) // inlier ?
				{
					Mat addtion(1, 2, CV_32F);
					addtion.col(0).setTo(imgDu.at<float>(v, u));
					addtion.col(1).setTo(imgDv.at<float>(v, u));
					Mat addtionu = imgDu.at<float>(v, u) * addtion;
					Mat addtionv = imgDv.at<float>(v, u) * addtion;
					for (int j = 0; j < A2.cols; j++)
					{
						A2.at<float>(0, j) = A2.at<float>(0, j) + addtionu.at<float>(0, j);
						A2.at<float>(1, j) = A2.at<float>(1, j) + addtionv.at<float>(0, j);
					}
				}
			} //end for
		// set new corner orientation
		cv::Mat v1, foo1;
		cv::Mat v2, foo2;
		cv::eigen(A1, v1, foo1);
		cv::eigen(A2, v2, foo2);
		cornersEdge1[i][0] = -foo1.at<float>(1, 0);
		cornersEdge1[i][1] = -foo1.at<float>(1, 1);
		cornersEdge2[i][0] = -foo2.at<float>(1, 0);
		cornersEdge2[i][1] = -foo2.at<float>(1, 1);

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//%  corner location refinement  %
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		cv::Mat G = cv::Mat::zeros(cv::Size(2, 2), CV_32F);
		cv::Mat b = cv::Mat::zeros(cv::Size(1, 2), CV_32F);
		for (int u = startX; u < startX + ROIwidth; u++)
			for (int v = startY; v < startY + ROIheight; v++)
			{
				// pixel orientation vector
				cv::Point2f o(imgDu.at<float>(v, u), imgDv.at<float>(v, u));
				float no = norm2d(o);
				if (no < 0.1)
					continue;
				o = o / no;
				//robust subpixel corner estimation
				if (u != cu || v != cv) // % do not consider center pixel
				{
					//compute rel. position of pixel and distance to vectors
					cv::Point2f w(u - cu, v - cv);
					float wvv1 = w.x * cornersEdge1[i][0] + w.y * cornersEdge1[i][1];
					float wvv2 = w.x * cornersEdge2[i][0] + w.y * cornersEdge2[i][1];

					cv::Point2f wv1(wvv1 * cornersEdge1[i][0], wvv1 * cornersEdge1[i][1]);
					cv::Point2f wv2(wvv2 * cornersEdge2[i][0], wvv2 * cornersEdge2[i][1]);
					cv::Point2f vd1(w.x - wv1.x, w.y - wv1.y);
					cv::Point2f vd2(w.x - wv2.x, w.y - wv2.y);
					float d1 = norm2d(vd1), d2 = norm2d(vd2);
					//if pixel corresponds with either of the vectors / directions
					if ((d1 < 3) && abs(o.x * cornersEdge1[i][0] + o.y * cornersEdge1[i][1]) < 0.25 || (d2 < 3) && abs(o.x * cornersEdge2[i][0] + o.y * cornersEdge2[i][1]) < 0.25)
					{
						float du = imgDu.at<float>(v, u), dv = imgDv.at<float>(v, u);
						cv::Mat uvt = (Mat_<float>(2, 1) << u, v);
						cv::Mat H = (Mat_<float>(2, 2) << du * du, du * dv, dv * du, dv * dv);
						G = G + H;
						cv::Mat t = H * (uvt);
						b = b + t;
					}
				}
			} //endfor
		//set new corner location if G has full rank
		Mat s, u, v;
		SVD::compute(G, s, u, v);
		int rank = 0;
		for (int k = 0; k < s.rows; k++)
		{
			if (s.at<float>(k, 0) > 0.0001 || s.at<float>(k, 0) < -0.0001) // not equal zero
			{
				rank++;
			}
		}
		if (rank == 2)
		{
			cv::Mat mp = G.inv() * b;
			cv::Point2f corner_pos_new(mp.at<float>(0, 0), mp.at<float>(1, 0));
			//  % set corner to invalid, if position update is very large
			if (norm2d(cv::Point2f(corner_pos_new.x - cu, corner_pos_new.y - cv)) >= 4)
			{
				cornersEdge1[i][0] = 0;
				cornersEdge1[i][1] = 0;
				cornersEdge2[i][0] = 0;
				cornersEdge2[i][1] = 0;
			}
			else
			{
				corners[i].x = mp.at<float>(0, 0);
				corners[i].y = mp.at<float>(1, 0);
			}
		}
		else //otherwise: set corner to invalid
		{
			cornersEdge1[i][0] = 0;
			cornersEdge1[i][1] = 0;
			cornersEdge2[i][0] = 0;
			cornersEdge2[i][1] = 0;
		}
	}
}

//compute corner statistics
void FindCorners::cornerCorrelationScore(Mat img, Mat imgWeight, vector<Point2f> cornersEdge, float &score)
{
	//center
	int c[] = {imgWeight.cols / 2, imgWeight.cols / 2};

	//compute gradient filter kernel(bandwith = 3 px)
	Mat img_filter = Mat::ones(imgWeight.size(), imgWeight.type());
	img_filter = img_filter * -1;
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			Point2f p1 = Point2f(i - c[0], j - c[1]);
			Point2f p2 = Point2f(p1.x * cornersEdge[0].x * cornersEdge[0].x + p1.y * cornersEdge[0].x * cornersEdge[0].y,
								 p1.x * cornersEdge[0].x * cornersEdge[0].y + p1.y * cornersEdge[0].y * cornersEdge[0].y);
			Point2f p3 = Point2f(p1.x * cornersEdge[1].x * cornersEdge[1].x + p1.y * cornersEdge[1].x * cornersEdge[1].y,
								 p1.x * cornersEdge[1].x * cornersEdge[1].y + p1.y * cornersEdge[1].y * cornersEdge[1].y);
			float norm1 = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
			float norm2 = sqrt((p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y));
			if (norm1 <= 1.5 || norm2 <= 1.5)
			{
				img_filter.ptr<float>(j)[i] = 1;
			}
		}
	}

	//normalize
	Mat mean, std, mean1, std1;
	meanStdDev(imgWeight, mean, std);
	meanStdDev(img_filter, mean1, std1);
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			imgWeight.ptr<float>(j)[i] = (float)(imgWeight.ptr<float>(j)[i] - mean.ptr<double>(0)[0]) / (float)std.ptr<double>(0)[0];
			img_filter.ptr<float>(j)[i] = (float)(img_filter.ptr<float>(j)[i] - mean1.ptr<double>(0)[0]) / (float)std1.ptr<double>(0)[0];
		}
	}

	//convert into vectors
	vector<float> vec_filter, vec_weight;
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			vec_filter.push_back(img_filter.ptr<float>(j)[i]);
			vec_weight.push_back(imgWeight.ptr<float>(j)[i]);
		}
	}

	//compute gradient score
	float sum = 0;
	for (int i = 0; i < vec_weight.size(); i++)
	{
		sum += vec_weight[i] * vec_filter[i];
	}
	sum = (float)sum / (float)(vec_weight.size() - 1);
	float score_gradient = sum >= 0 ? sum : 0;

	//create intensity filter kernel
	Mat kernelA, kernelB, kernelC, kernelD;
	createkernel(atan2(cornersEdge[0].y, cornersEdge[0].x), atan2(cornersEdge[1].y, cornersEdge[1].x), c[0], kernelA, kernelB, kernelC, kernelD); //1.1 产生四种核

	//checkerboard responses
	float a1, a2, b1, b2;
	a1 = kernelA.dot(img);
	a2 = kernelB.dot(img);
	b1 = kernelC.dot(img);
	b2 = kernelD.dot(img);

	float mu = (a1 + a2 + b1 + b2) / 4;

	float score_a = (a1 - mu) >= (a2 - mu) ? (a2 - mu) : (a1 - mu);
	float score_b = (mu - b1) >= (mu - b2) ? (mu - b2) : (mu - b1);
	float score_1 = score_a >= score_b ? score_b : score_a;

	score_b = (b1 - mu) >= (b2 - mu) ? (b2 - mu) : (b1 - mu);
	score_a = (mu - a1) >= (mu - a2) ? (mu - a2) : (mu - a1);
	float score_2 = score_a >= score_b ? score_b : score_a;

	float score_intensity = score_1 >= score_2 ? score_1 : score_2;

	score_intensity = score_intensity > 0.0 ? score_intensity : 0.0;
	score = score_gradient * score_intensity;
}
//score corners
void FindCorners::scoreCorners(Mat img, Mat imgAngle, Mat imgWeight, vector<Point2f> &corners, vector<int> radius, vector<float> &score)
{
	//for all corners do
	for (int i = 0; i < corners.size(); i++)
	{
		//corner location
		int u = corners[i].x + 0.5;
		int v = corners[i].y + 0.5;
		if (i == 278)
		{
			int aaa = 0;
		}
		//compute corner statistics @ radius 1
		vector<float> scores;
		for (int j = 0; j < radius.size(); j++)
		{
			scores.push_back(0);
			int r = radius[j];
			if (u > r && u <= (img.cols - r - 1) && v > r && v <= (img.rows - r - 1))
			{
				int startX, startY, ROIwidth, ROIheight;
				startX = u - r;
				startY = v - r;
				ROIwidth = 2 * r + 1;
				ROIheight = 2 * r + 1;

				Mat sub_img = img(Rect(startX, startY, ROIwidth, ROIheight));
				Mat sub_imgWeight = imgWeight(Rect(startX, startY, ROIwidth, ROIheight));
				vector<Point2f> cornersEdge;
				cornersEdge.push_back(Point2f((float)cornersEdge1[i][0], (float)cornersEdge1[i][1]));
				cornersEdge.push_back(Point2f((float)cornersEdge2[i][0], (float)cornersEdge2[i][1]));
				cornerCorrelationScore(sub_img, sub_imgWeight, cornersEdge, scores[j]);
			}
		}
		//take highest score
		score.push_back(*max_element(begin(scores), end(scores)));
		//score.swap(scores)
		// 		for(int i = 0;i < score.size();i++){
		// 		  cout << score[i] << endl;
		// 		}
	}
}

Mat FindCorners::conv2(const Mat &img, const Mat &ikernel, ConvolutionType type)
{
	Mat dest;
	Mat kernel;
	flip(ikernel, kernel, -1);
	Mat source = img;
	if (CONVOLUTION_FULL == type)
	{
		source = Mat();
		const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
		copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
	}
	Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = BORDER_CONSTANT;
	filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);

	if (CONVOLUTION_VALID == type)
	{
		dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2).rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
	}
	return dest;
}
//scoreThreshold(tau) in Paper is 0.02
void FindCorners::detectCorners(Mat &Src, vector<Point2f> &resultCorners, Corners &mcorners, float scoreThreshold)
{
	Mat gray, imageNorm, colorMat;
	gray = Mat(Src.size(), CV_8U);
	colorMat = Mat(Src.size(), CV_8UC3);

	if (Src.channels() == 3)
	{
		cvtColor(Src, gray, COLOR_BGR2GRAY); //变为灰度图
		Src.copyTo(colorMat);
	}
	else
	{
		gray = Src.clone();
		cvtColor(Src, colorMat, COLOR_GRAY2BGR);
	}

	cv::GaussianBlur(gray, gray, cv::Size(9, 9), 1.5);

	normalize(gray, imageNorm, 0, 1, cv::NORM_MINMAX, CV_32F); //对灰度图进行归一化

	Mat imgCorners = Mat::zeros(imageNorm.size(), CV_32F); //卷积核得出的点

	std::cout << "begin filtering !" << std::endl;
	double t = (double)getTickCount();

	Mat imgCornerA1(imageNorm.size(), CV_32F);
	Mat imgCornerB1(imageNorm.size(), CV_32F);
	Mat imgCornerC1(imageNorm.size(), CV_32F);
	Mat imgCornerD1(imageNorm.size(), CV_32F);

	Mat imgCornerMean(imageNorm.size(), CV_32F);

	Mat imgCornerA(imageNorm.size(), CV_32F);
	Mat imgCornerB(imageNorm.size(), CV_32F);
	Mat imgCorner1(imageNorm.size(), CV_32F);
	Mat imgCorner2(imageNorm.size(), CV_32F);

	for (int i = 0; i < 6; i++)
	{
		//按照论文步骤，第一步：用卷积核进行卷积的方式找出可能是棋盘格角点的点
		Mat kernelA1, kernelB1, kernelC1, kernelD1;
		createkernel(templateProps[i].x, templateProps[i].y, radius[i / 2], kernelA1, kernelB1, kernelC1, kernelD1); //1.1 产生四种核
																													 // 		std::cout << "kernelA:" << kernelA1 << endl << "kernelB:" << kernelB1 << endl
																													 // 			<< "kernelC:" << kernelC1 << endl << "kernelD:" << kernelD1 << endl;
																													 // 		imgCornerA1 = conv2(imageNorm, kernelA1, CONVOLUTION_SAME);
																													 // 		imgCornerB1 = conv2(imageNorm, kernelB1, CONVOLUTION_SAME);
																													 // 		imgCornerC1 = conv2(imageNorm, kernelC1, CONVOLUTION_SAME);
																													 // 		imgCornerD1 = conv2(imageNorm, kernelD1, CONVOLUTION_SAME);

		filter2D(imageNorm, imgCornerA1, CV_32F, kernelA1); //1.2 用所产生的核对图像做卷积
		filter2D(imageNorm, imgCornerB1, CV_32F, kernelB1);
		filter2D(imageNorm, imgCornerC1, CV_32F, kernelC1);
		filter2D(imageNorm, imgCornerD1, CV_32F, kernelD1);

		imgCornerMean = (imgCornerA1 + imgCornerB1 + imgCornerC1 + imgCornerD1) / 4.0; //1.3 按照公式进行计算

		getMin(imgCornerA1 - imgCornerMean, imgCornerB1 - imgCornerMean, imgCornerA);
		getMin(imgCornerMean - imgCornerC1, imgCornerMean - imgCornerD1, imgCornerB);
		getMin(imgCornerA, imgCornerB, imgCorner1);

		getMin(imgCornerMean - imgCornerA1, imgCornerMean - imgCornerB1, imgCornerA);
		getMin(imgCornerC1 - imgCornerMean, imgCornerD1 - imgCornerMean, imgCornerB);
		getMin(imgCornerA, imgCornerB, imgCorner2);

		getMax(imgCorners, imgCorner1, imgCorners);
		getMax(imgCorners, imgCorner2, imgCorners);

		//getMin(imgCornerA1, imgCornerB1, imgCornerA); getMin(imgCornerC1, imgCornerD1, imgCornerB);
		//getMin(imgCornerA - imgCornerMean, imgCornerMean - imgCornerB, imgCorner1);
		//getMin(imgCornerMean - imgCornerA, imgCornerB - imgCornerMean, imgCorner2);
		//getMax(imgCorners, imgCorner2, imgCorners);//1.4 获取每个像素点的得分
		//getMax(imgCorners, imgCorner1, imgCorners);//1.4 获取每个像素点的得分
	}

	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "filtering time cost :" << t << std::endl;
	namedWindow("ROI"); //创建窗口，显示原始图像

	imshow("ROI", imgCorners);
	waitKey(1);
	for (int i = 0; i < cornerPoints.size(); i++)
	{
		circle(colorMat, cornerPoints[i], 5, CV_RGB(255, 0, 0), 2);
	}
	//cout << "点数 = " << imgCorners.size() << endl;

	nonMaximumSuppression(imgCorners, cornerPoints, 0.025, 5, 3); //1.5 非极大值抑制算法进行过滤，获取棋盘格角点初步结果
																  // 	namedWindow("src");//创建窗口，显示原始图像
																  // 	imshow("src", cornerPoints); waitKey(0);

	// 	for(int i = 0; i< imgCorners.size();i++){
	// 	  circle(Src, imgCorners[i], 5, CV_RGB(255, 0, 0), 2);
	// 	}
	//cout << "corner数量 = " << cornerPoints.size() << endl;

	if (cornerPoints.size() > 0)
	{
		for (int i = 0; i < cornerPoints.size(); i++)
		{
			circle(Src, cornerPoints[i], 5, CV_RGB(255, 0, 0), 2);
		}
	}
	namedWindow("src"); //创建窗口，显示原始图像
	imshow("src", Src);
	waitKey(1);
	//std::cout << "Good Job_1" << std::endl;

	//算两个方向的梯度
	Mat imageDu(gray.size(), CV_32F);
	Mat imageDv(gray.size(), CV_32F);

	Mat img_angle(gray.size(), CV_32F);
	Mat img_weight(gray.size(), CV_32F);
	//获取梯度角度和权重
	getImageAngleAndWeight(gray, imageDu, imageDv, img_angle, img_weight);
	//subpixel refinement
	refineCorners(cornerPoints, imageDu, imageDv, img_angle, img_weight, 10);
	if (cornerPoints.size() > 0)
	{
		for (int i = 0; i < cornerPoints.size(); i++)
		{
			if (cornersEdge1[i][0] == 0 && cornersEdge1[i][0] == 0)
			{
				cornerPoints[i].x = 0;
				cornerPoints[i].y = 0;
			}
		}
	}
	//remove corners without edges

	//score corners
	vector<float> score;

	scoreCorners(imageNorm, img_angle, img_weight, cornerPoints, radius, score);

	// remove low scoring corners

	// 	for(int i = 0;i < nlen;i++){
	// 	  circle(colorMat, cornerPoints[i], 5, CV_RGB(255, 0, 0), 2);
	// 	}
	//
	// 	namedWindow("colorMat");
	// 	imshow("colorMat", colorMat);
	// 	waitKey(0);
	//cout << cornerPoints.size() << endl;
	vector<Point2f> resultTemp;
	int nlen = cornerPoints.size();
	if (nlen > 0)
	{
		for (int i = 0; i < nlen; i++)
		{
			if (score[i] > scoreThreshold)
			{
				//resultTemp.push_back(cornerPoints[i]);
				//resultCorners.push_back(cornerPoints[i]);
				mcorners.p.push_back(cornerPoints[i]);
				mcorners.v1.push_back(cv::Vec2f(cornersEdge1[i][0], cornersEdge1[i][1]));
				mcorners.v2.push_back(cv::Vec2f(cornersEdge2[i][0], cornersEdge2[i][1]));
				mcorners.score.push_back(score[i]);
				//cout << "Mcorners.p[" << i << "] = " << mcorners.p[i] << endl;
				//cout <<"Mcorners.score[" << i << "] = " <<mcorners.score[i] << endl;
				circle(colorMat, cornerPoints[i], 5, CV_RGB(255, 0, 0), 2);
				//cout << "resultTemp[" << i << "] = " << resultTemp[i] << endl;
			}
		}

		Mat chessbd = structureRecovery(mcorners);
		cout << "rows = "<< chessbd.rows << "cols = " << chessbd.cols<<endl;
		
		if(chessbd.rows < chessbd.cols){
			for (int i = chessbd.rows-1; i >= 0; i--)
		{
			for (int j = 0; j < chessbd.cols; j++)
			{
				//cout << "(" << i << "," << j << ") "<<endl;
				//cout << "(" << i << "," << j << ") = " << mcorners.p[chessbd.at<int>(i,j)] << " \ t";
				resultCorners.push_back(mcorners.p[chessbd.at<int>(i,j)]);
				putText(colorMat, cv::format("%d", chessbd.at<int>(i,j)), mcorners.p[chessbd.at<int>(i,j)], 
				0, 0.3, CV_RGB(0,255,0));
				// putText(colorMat, cv::format("%d", chessbd.at<int>(i,j)), mcorners.p[chessbd.at<int>(i,j)], 
				// 0, 0.3, CV_RGB(0,255,0));
			}
			cout << endl;
		}
		}
		else{
		for (int j = chessbd.cols-1; j >= 0; j--)
		{
			for (int i = 0; i < chessbd.rows; i++)
			{
				//cout << "(" << i << "," << j << ") "<<endl;
				//cout << "(" << i << "," << j << ") = " << mcorners.p[chessbd.at<int>(i,j)] << " \ t";
				resultCorners.push_back(mcorners.p[chessbd.at<int>(i,j)]);
				putText(colorMat, cv::format("%d", chessbd.at<int>(i,j)), mcorners.p[chessbd.at<int>(i,j)], 
				0, 0.3, CV_RGB(0,255,0));
			}
			cout << endl;
		}
		}

		//将Mat中的点排序后转至Vector中
		// 		for( int i = 0; i < chessbd.rows;i++ ){
		// 		  for(int j = 0;j<chessbd.cols;j++){
		// 		      resultCorners.push_back(mcorners.p[chessbd.at<float>(i,j)]);
		// 		  }
		// 		}

		// 		for(int i = 0; i< resultCorners.size(); i++){
		//
		// 		  cout << "("<< resultCorners[i].x << "," << resultCorners[i].y << ")" << endl;
		// 		}
		//
		// 		for(int i = 0; i < mcorners.p.size();i++){
		// 		  int t = i * 8;
		//
		// 		  //cout << mcorners.p.size();
		// 		  if(t > mcorners.p.size()-1){
		// 		   t = t %(mcorners.p.size()-1);
		//
		// 		  }
		// 		    if( i == (mcorners.p.size()-1)){
		// 		    t = i;
		// 		  }
		// 		 // cout << "t = "<< t << endl;
		// 		  resultCorners.push_back(resultTemp[t]);
		// 		  //cout << "ResultCorners[" << i << "] = " << resultCorners[i] << endl;
		// 		}

		// 		for(int i = 0; i < mcorners.p.size();i++){
		// 		 sort() resultTemp[i].x
		//
		// 		}

		//
	}
	namedWindow("colorMat");
	imshow("colorMat", colorMat);
	waitKey(1);
	//circle(Src, mcorners.p, 5, CV_RGB(255, 0, 0), 2);
	//cout << "ResultCorners.size() = " << resultCorners.size() << endl;

	std::vector<cv::Vec2f> corners_n1(mcorners.p.size());

	for (int i = 0; i < corners_n1.size(); i++)
	{
		if (mcorners.v1[i][0] + mcorners.v1[i][1] < 0.0)
		{
			mcorners.v1[i] = -mcorners.v1[i];
		}
		corners_n1[i] = mcorners.v1[i];
		float flipflag = corners_n1[i][0] * mcorners.v2[i][0] + corners_n1[0][1] * mcorners.v2[i][1];
		if (flipflag > 0)
			flipflag = -1;
		else
			flipflag = 1;
		mcorners.v2[i] = flipflag * mcorners.v2[i];
	}
}

// 	cvtColor(Src,Src,COLOR_GRAY2BGR);
// 	if (cornerPoints.size()>0)
// 	{
// 		for (int i = 0; i < cornerPoints.size(); i++)
// 		{
// 			if (score[i]>scoreThreshold)
// 			{
// 				circle(Src, cornerPoints[i], 5, CV_RGB(255, 0, 0), 2);
// 			}
//
// 		}
// 	}
// 	namedWindow("src");//创建窗口，显示原始图像
// 	imshow("src", Src); waitKey(0);
//
// 	Point maxLoc;
// 	FileStorage fs2("test.xml", FileStorage::WRITE);//写XML文件
// 	fs2 << "img_corners_a1" << cornerPoints;
// }
void FindCorners::savecorners(Corners &mcorners, char *filename)
{
	FILE *fq;
	fq = fopen(filename, "wt+");
	if (fq == NULL)
	{
		printf("no file : %d\n", filename);
		return;
	}
	int n = mcorners.p.size();
	fprintf(fq, "%d\n", n);

	for (int i = 0; i < n; i++)
	{
		fprintf(fq, "%d \t %f \t %f \t %f \t %f \t %f \t %f \t %f  \n", i, mcorners.p[i].x, mcorners.p[i].y,
				mcorners.v1[i][0], mcorners.v1[i][1], mcorners.v2[i][0], mcorners.v2[i][1], mcorners.score[i]);
	}
	fclose(fq);
}

void FindCorners::readcorners(Corners &mcorners, char *filename)
{
	FILE *fq;
	fq = fopen(filename, "rt+");
	if (fq == NULL)
	{
		printf("no file : %d\n", filename);
		return;
	}
	int n;
	fscanf(fq, "%d\n", &n);

	mcorners.p.resize(n);
	mcorners.v1.resize(n);
	mcorners.v2.resize(n);
	mcorners.score.resize(n);

	for (int i = 0; i < n; i++)
	{
		int ii;
		fscanf(fq, "%d \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n", &ii, &mcorners.p[i].x, &mcorners.p[i].y,
			   &mcorners.v1[i][0], &mcorners.v1[i][1], &mcorners.v2[i][0], &mcorners.v2[i][1], &mcorners.score[i]);

		//printf("%d \t %f \t %f \t %f \t %f \t %f \t %f \t %f  \n", i, mcorners.p[i].x, mcorners.p[i].y,
		//mcorners.v1[i][0], mcorners.v1[i][1], mcorners.v2[i][0], mcorners.v2[i][1], mcorners.score[i]);
	}
	fclose(fq);
}

inline float FindCorners::distv(cv::Vec2f &a, cv::Vec2f &b)
{
	return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]));
}

inline float FindCorners::mean_l(std::vector<float> &resultSet)
{
	double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
	double mean = sum / resultSet.size(); //��ֵ
	return mean;
}

inline float FindCorners::stdev_l(std::vector<float> &resultSet, float &mean)
{
	double accum = 0.0;
	mean = mean_l(resultSet);
	std::for_each(std::begin(resultSet), std::end(resultSet), [&](const double d) {
		accum += (d - mean) * (d - mean);
	});
	double stdev = sqrt(accum / (resultSet.size() - 1)); //����
	return stdev;
}

inline float FindCorners::stdevmean(std::vector<float> &resultSet)
{
	float stdvalue, meanvalue;

	stdvalue = stdev_l(resultSet, meanvalue);

	return stdvalue / meanvalue;
}

Mat FindCorners::structureRecovery(Corners &mcorners)
{

	//Mat chessboards = Mat::zeros(3,3,CV_32F);
	Mat chessboard;
	Mat coutChessboard;
	//vector<Mat> chessboards;
	for (int i = 0; i < mcorners.p.size(); i++)
	{

		if (i % 100 == 0)
		{
			cout << i << "/" << mcorners.p.size() << endl;
		}

		
		cout << "i = " << i << endl;

		chessboard = initChessboard(mcorners, i);
		// cout << "chessboard" << chessboard << endl;
		// cout << "chessboards.empty = " << chessboard.empty() << endl;

		if (chessboard.empty() == true)
		{
			continue;
		}
		//cout << "Test" << endl;
		float E = chessboardEnergy(chessboard, mcorners);
		if (E > 0)
		{
			continue;
		}
		cout << " E = " << E << endl;

		//Mat chessboard;
		// compute current energy
		// float energy = chessboardEnergy(chessboard, mcorners);

		// Mat proposal = growChessboard(chessboard, mcorners);
		//float = chessboardEnergy(proposal, mcorners);

		int s = 0;

		// for (int i = 0; i < 1; i++)
		// {
		while(true)
		{
			s++;
			// compute current energy
			float energy = chessboardEnergy(chessboard, mcorners);
			std::vector<cv::Mat> proposal(4);
			std::vector<float> p_energy(4);
			//compute proposals and energies
			for (int j = 0; j < 4; j++)
			{
				proposal[j] = growChessboard(chessboard, mcorners, j);
				//cout << "proposal{" << j << "} = " << proposal[j] << endl;
				p_energy[j] = chessboardEnergy(proposal[j], mcorners);
			}
			// find best proposal
			float min_value = p_energy[0];
			int min_idx = 0;
			for (int i0 = 1; i0 < p_energy.size(); i0++)
			{
				if (min_value > p_energy[i0])
				{
					min_value = p_energy[i0];
					min_idx = i0;
				}
			}
			// accept best proposal, if energy is reduced
			cv::Mat chessboardt;
			if (p_energy[min_idx] < energy)
			{
				chessboardt = proposal[min_idx];
				chessboard = chessboardt.clone();
				//
				//cout << "size = " << chessboard.size().area()<< endl;
				if(chessboard.size().area() == 88)
				{
					coutChessboard = chessboard.clone();
				}
				// cout << chessboard << endl;
				// cout << endl;
				// cout << "Cout" << coutChessboard << endl;
			}
			else
			{
				break;
			}

		}
		// if(chessboard.rows == 11 && chessboard.cols == 8)
		// 	{
		// 		break;
		// 	}//end while
		//return coutChessboard;
		
		// if(chessboard.size().area() == 88){
		// 	break;
		// }
		/*if (chessboardEnergy(chessboard, mcorners) < -10)
		{
			//check if new chessboard proposal overlaps with existing chessboards
			cv::Mat overlap = cv::Mat::zeros(cv::Size(2, chessboards.size()), CV_32FC1);
			for (int j = 0; j < chessboards.size(); j++)
			{
				bool isbreak = false;
				for (int k = 0; k < chessboards[j].size().area(); k++)
				{
					int refv = chessboards[j].at<int>(k / chessboards[j].cols, k % chessboards[j].cols);
					for (int l = 0; l < chessboard.size().area(); l++)
					{
						int isv = chessboard.at<int>(l / chessboard.cols, l % chessboard.cols);
						if (refv == isv)
						{
							overlap.at<float>(j, 0) = 1.0;
							float s = chessboardEnergy(chessboards[j], mcorners);
							overlap.at<float>(j, 1) = s;
							isbreak = true;
							break;
						}
					}
					//	if (isbreak == true)
					//	{
					//	break;
					//	}
				}
				//if (isbreak == true)
				//{
				//	break;
				//}
			} //endfor

			// add chessboard(and replace overlapping if neccessary)
			bool isoverlap = false;
			for (int i0 = 0; i0 < overlap.rows; i0++)
			{
				if (overlap.empty() == false)
				{
					if (abs(overlap.at<float>(i0, 0)) > 0.000001) // ==1
					{
						isoverlap = true;
						break;
					}
				}
			}
			if (isoverlap == false)
			{
				chessboards.push_back(chessboard);
			}
			else
			{
				bool flagpush = true;
				std::vector<bool> flagerase(overlap.rows);
				for (int m = 0; m < flagerase.size(); m++)
				{
					flagerase[m] = false;
				}
				float ce = chessboardEnergy(chessboard, mcorners);
				for (int i1 = 0; i1 < overlap.rows; i1++)
				{
					if (abs(overlap.at<float>(i1, 0)) > 0.0001) // ==1//���ص�
					{
						bool isb1 = overlap.at<float>(i1, 1) > ce;
						int a = int(overlap.at<float>(i1, 1) * 1000);
						int b = int(ce * 1000);
						bool isb2 = a > b;
						if (isb1 != isb2)
							printf("find bug!\n");
						if (isb2)
						{
							flagerase[i1] = true;
						}
						else
						{
							flagpush = false;
							//	break;
						} //endif
					}	 //endif
				}		  //end for
				if (flagpush == true)
				{
					for (int i1 = 0; i1 < chessboards.size();)
					{
						std::vector<cv::Mat>::iterator it = chessboards.begin() + i1;
						std::vector<bool>::iterator it1 = flagerase.begin() + i1;
						if (*it1 == true)
						{
							chessboards.erase(it);
							flagerase.erase(it1);
							i1 = 0;
						}
						i1++;
					}
					chessboards.push_back(chessboard);
				}

			} //endif

		} //endif
		*/
	} //end for
	cout << "Final chessboard = " << coutChessboard << endl;
	return coutChessboard;
	printf("end!\n");
}//end function
//     cv::Mat chessboard;
//     float energy = chessboardEnergy(chessboard, mcorners);
//     for(int j = 0; j < 4; j++){
// 	    proposal[j].push_back(growChessboard(chessboard, mcorners, j));
// 	    p_energy[j] = chessboardEnergy(proposal[j],mcorners);
// 	  }
//
//     float min_value = p_energy[0];
//     int min_idx = 0;
//     for (int i0 = 1; i0 < p_energy.size(); i0++)
//     {
//       if (min_value > p_energy[i0])
//       {
// 	min_value = p_energy[i0];
// 	min_idx = i0;
//
//       }
//     }
//     // accept best proposal, if energy is reduced
//     cv::Mat chessboardt;
//     if (p_energy[min_idx] < energy)
//     {
//       chessboardt = proposal[min_idx];
//       chessboard = chessboardt.clone();
//     }
//     else
//     {
//       break;
//     }
//   }
// }

bool FindCorners::findValue(const cv::Mat &mat, float value)
{
	for (int i = 0; i < mat.rows; i++)
	{
		const float *row = mat.ptr<float>(i);
		if (std::find(row, row + mat.cols, value) != row + mat.cols)
			return true;
	}
	return false;
}

Mat FindCorners::initChessboard(Corners &mcorners, int idx)
{
	// return if not enough corners
	//cout << " idx = " << idx << endl;
	if (mcorners.p.size() < 9)
	{
		cout << "The seed does not have the enough corners.Please retry!";
		//chessboard.release();//return empty!
		//return -1;
	}
	//extract feature index and orientation(central element)
	Mat chessboard = -1 * Mat::ones(3, 3, CV_32S);

	Vec2f v1 = mcorners.v1[idx];
	Vec2f v2 = mcorners.v2[idx];
	chessboard.at<int>(1, 1) = idx;
	vector<float> dist1(2);
	vector<float> dist2(6);
	// cout << "v1" << v1 << endl;
	// cout << "v2" << v2 << endl;
	//find left/right/top/bottom neighbor
	directionalNeighbor(idx, +1 * v1, chessboard, mcorners, chessboard.at<int>(1, 2), dist1[0]);
	directionalNeighbor(idx, -1 * v1, chessboard, mcorners, chessboard.at<int>(1, 0), dist1[1]);
	directionalNeighbor(idx, +1 * v2, chessboard, mcorners, chessboard.at<int>(2, 1), dist2[0]);
	directionalNeighbor(idx, -1 * v2, chessboard, mcorners, chessboard.at<int>(0, 1), dist2[1]);
	//cout << "Mat:Chessboard = " << endl;
	//cout << chessboard << endl;
	directionalNeighbor(chessboard.at<int>(1, 0), -1 * v2, chessboard, mcorners, chessboard.at<int>(0, 0), dist2[2]);
	directionalNeighbor(chessboard.at<int>(1, 0), +1 * v2, chessboard, mcorners, chessboard.at<int>(2, 0), dist2[3]);
	directionalNeighbor(chessboard.at<int>(1, 2), -1 * v2, chessboard, mcorners, chessboard.at<int>(0, 2), dist2[4]);
	directionalNeighbor(chessboard.at<int>(1, 2), +1 * v2, chessboard, mcorners, chessboard.at<int>(2, 2), dist2[5]);
	// cout << "Mat:Chessboard_2 = " << endl;
	// cout << chessboard << endl;
	// initialization must be homogenously distributed
	// cout << "dist1 = " << endl;
	// for(int i = 0; i < dist1.size();i++){
	// 	cout << dist1[i] << endl;
	// }

	// cout << "dist2 = " << endl;
	// for(int j = 0;j < dist2.size();j++){
	// 	cout << dist2[j] << endl;
	// }

	bool sigood = false;
	sigood = sigood || (dist1[0] < 0) || (dist1[1] < 0);
	sigood = sigood || (dist2[0] < 0) || (dist2[1] < 0) || (dist2[2] < 0) || (dist2[3] < 0) || (dist2[4] < 0) || (dist2[5] < 0);
	//cout << "Test" << endl;
	//cout << "Std(dist1) = " << stdevmean(dist1) << endl;
	//cout << "Std(dist2) = " << stdevmean(dist2) << endl;
	sigood = sigood || (stdevmean(dist1) > 0.3) || (stdevmean(dist2) > 0.3);
	if (sigood == true)
	{
		//cout << "It is not homogenously distributed, please try again." << endl;
		chessboard.release();
		return chessboard;
	}
	return chessboard;
	//cout << "Begin&&&" << endl;
}

int FindCorners::directionalNeighbor(int idx, cv::Vec2f v, cv::Mat chessboard, Corners &corners, int &neighbor_idx, float &min_dist)
{
#if 1
	// list of neighboring elements, which are currently not in use
	std::vector<int> unused(corners.p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;

	}

	for (int i = 0; i < chessboard.cols; i++)
	{
		for (int j = 0; j < chessboard.rows; j++)
		{
			int xy = chessboard.at<int>(j, i);
			if (xy >= 0)
			{
				unused[xy] = -1;
				//cout << " unused[" << i << "] =" << unused[]
			}
			//cout << xy <;
			//cout << "("<<j << "," << i <<") = " << xy << endl;
		}
	}
	for (int i = 0; i < chessboard.rows; i++)
	{
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;
			}
		}
	}
	//cout << "Test" << endl;
	//cout << "id = " << idx << endl;
	int nsize = unused.size();

	for (int i = 0; i < nsize;)
	{
		if (unused[i] < 0)
		{
			std::vector<int>::iterator iter = unused.begin() + i;
			unused.erase(iter);
			i = 0;
			nsize = unused.size();
			continue;
		}
		i++;
	}

	std::vector<float> dist_edge;
	std::vector<float> dist_point;
	//vector<float> dir;
	cv::Point2f idxp = cv::Point2f(corners.p[idx].x, corners.p[idx].y);
	// direction and distance to unused corners
	for (int i = 0; i < unused.size(); i++)
	{
		int ind = unused[i];
		cv::Point2f diri = cv::Point2f(corners.p[ind].x, corners.p[ind].y) - idxp;
		float disti = diri.x * v[0] + diri.y * v[1];

		cv::Point2f de = Point2f(diri.x - disti * v[0],diri.y - disti * v[1]);
		dist_edge.push_back(norm2d(de));
		// distances
		dist_point.push_back(disti);
	}
#else
	// list of neighboring elements, which are currently not in use
	std::vector<int> unused(corners.p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1; //flag the used idx
			}
		}

	std::vector<float> dist_edge;
	std::vector<float> dist_point;

	cv::Vec2f idxp = cv::Vec2f(corners.p[idx].x, corners.p[idx].y);
	// direction and distance to unused corners
	for (int i = 0; i < corners.p.size(); i++)
	{
		if (unused[i] == -1)
		{
			dist_point.push_back(std::numeric_limits<float>::max());
			dist_edge.push_back(0);
			continue;
		}
		cv::Vec2f diri = cv::Vec2f(corners.p[i].x, corners.p[i].y) - idxp;
		float disti = diri[0] * v[0] + diri[1] * v[1];

		cv::Vec2f de = diri - disti * v;
		dist_edge.push_back(distv(de, cv::Vec2f(0, 0)));
		// distances
		dist_point.push_back(disti);
	}

#endif

	// find best neighbor
	int min_idx = 0;
	min_dist = std::numeric_limits<float>::max();

	//min_dist = dist_point[0] + 5 * dist_edge[0];
	for (int i = 0; i < dist_point.size(); i++)
	{
		if (dist_point[i] > 0)
		{
			float m = dist_point[i] + 5 * dist_edge[i];
			if (m < min_dist)
			{
				min_dist = m;
				min_idx = i;
			}
		}
	}
	neighbor_idx = unused[min_idx];

	return 1;
}
// vector<float> unused;
//     vector<float> used;
//     vector<float>::iterator Iter;
//
//     for(int i = 0; i < chessboard.cols; i++){
//         for(int j = 0; j < chessboard.rows; j++){
// 	  cout << "123123123" << endl;
// 	  cout << chessboard.at<float>(j,i) << endl;
//             if(chessboard.at<float>(j,i) != 0){
//                 used.push_back(chessboard.at<float>(j,i));
//
//             }
//
//         }
//     }
//
//     for(int i = 0; i < used.size();i++){
// 	    cout << used[i] << endl;
//     }
//
//     for(int i = 0; i < unused.size(); i++){
//     }
//     for(Iter = unused.begin();Iter!= unused.end();){
//       for(int j = 0; j < used.size();j++){
// 	      if(*Iter == used[j]){
// 	        unused.erase(Iter);
// 	        Iter=unused.begin();
// 	      }
// 	    else{
// 	      Iter++;
//     	}
//       }
//     }
//     vector<Point2f> dir;
//     vector<float> dist;
//     vector<float>dist_edge;
//     for(int i = 0;i < unused.size();i++){
//       dir.push_back(corners.p[i] - corners.p[idx]);
//       dist.push_back(dir[i].x*v[0] + dir[i].y * v[1]);
//       Point2f dist_edgePoints = Point2f(dir[i].x - dist[i] * v[0],dir[i].y - dist[i] * v[1]);
//       dist_edge.push_back(sqrt(dist_edgePoints.x * dist_edgePoints.x + dist_edgePoints.y * dist_edgePoints.y));
//     }
//
//     vector<float> dist_point(dist);
//
//     for(int i = 0;i < unused.size();i++){
//       if(dist[i] < 0)
// 	      dist_point[i] = INT_MAX;
//     }
//
//     vector<float> temp;
//
//     for(int i = 0; i < dist_point.size();i++){
//       temp[i] = dist_point[i] + 5 * dist_edge[i];
//     }
//
//     auto smallest = std::min_element(begin(temp), end(temp));
//     int min_idx = distance(std::begin(temp), smallest);
//     cout << "min element is " << *smallest<< " at position " << min_idx << endl;
//
//     min_dist = *smallest;
//     neighbor_idx = unused[min_idx];
//
// //     for(int i = 0; i < min_idx;i++){
// //       neighbor_idx.push_back(unused[i]);
// //     }
// }

float FindCorners::chessboardEnergy(Mat chessboard, Corners &corners)
{
	float E_corners = -1 * chessboard.size().area();
	//energy: structur
	float E_structure = 0;
	//walk through rows
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols - 2; j++)
		{
			std::vector<cv::Point2f> x;
			float E_structure0 = 0;
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(i, k);
				x.push_back(corners.p[n]);
			}
			E_structure0 = norm2d(x[0] + x[2] - 2.0 * x[1]);
			float tv = norm2d(x[0] - x[2]);
			E_structure0 = E_structure0 / tv;
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	//walk through columns
	for (int i = 0; i < chessboard.cols; i++)
		for (int j = 0; j < chessboard.rows - 2; j++)
		{
			std::vector<cv::Point2f> x;
			float E_structure0 = 0;
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(k, i);
				x.push_back(corners.p[n]);
			}
			E_structure0 = norm2d(x[0] + x[2] - 2 * x[1]);
			float tv = norm2d(x[0] - x[2]);
			E_structure0 = E_structure0 / tv;
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	// final energy
	float E = E_corners + 1 * chessboard.size().area() * E_structure;

	return E;
}

Mat FindCorners::growChessboard(Mat chessboard, Corners &corners, int border_type)
{
	//if there do not exist any chessboards.
	// cout << "test123" << endl;
	if (chessboard.empty() == true)
	{
		//std::cout << "Chessboard is Empty.So sad=.=" << endl;
		return chessboard;
	}
	//extract feature locations.
	std::vector<cv::Point2f> p = corners.p;
	// list of  unused feature elements

	std::vector<int> unused(p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	for (int i = 0; i < chessboard.rows; i++)
	{
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy > 0)
			{
				unused[xy] = -1;
			}
			//cout << "unused[" << xy << "] = " << unused[xy] << endl;
		}

	}
	int nsize = unused.size();
	for (int i = 0; i < nsize;)
	{
		if (unused[i] < 0)
		{
			std::vector<int>::iterator iter = unused.begin() + i;
			unused.erase(iter);
			i = 0;
			nsize = unused.size();
			continue;
		}
		// cout << "unused[" << i << "] = " << unused[i] << endl;
		i++;
	}
	std::vector<cv::Point2f> cand;
	for (int i = 0; i < unused.size(); i++)
	{
		cand.push_back(corners.p[unused[i]]);
	}
	// switch border type 1..4
	cv::Mat chesstemp;

	switch (border_type)
	{
	case 0:
	{
		std::vector<cv::Point2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
		{
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == chessboard.cols - 3)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Point2f(p[ij]));
				}
				if (col == chessboard.cols - 2)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Point2f(p[ij]));
				}

				if (col == chessboard.cols - 1)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Point2f(p[ij]));
				}
			}
			// cout << "size1" << p1.size() << endl;
			// cout << "size2" << p2.size() << endl;
			// cout << "size3" << p3.size() << endl;

		}
		//vector<Point2f> pred;
		std::vector<int> idx;
		pred = predictCorners(p1, p2, p3,pred);
		//cout << "test" << endl;
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 0, 1, 0, 0);

		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, chesstemp.cols - 1) = unused[idx[i]]; //��
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 1:
	{
		std::vector<cv::Point2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row == chessboard.rows - 3)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Point2f(p[ij]));
				}
				if (row == chessboard.rows - 2)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Point2f(p[ij]));
				}
				if (row == chessboard.rows - 1)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Point2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 1, 0, 0, 0, 0);
		for (int i = 0; i < chesstemp.cols; i++)
		{
			chesstemp.at<int>(chesstemp.rows - 1, i) = unused[idx[i]]; //��
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 2:
	{
		std::vector<cv::Point2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == 2)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Point2f(p[ij]));
				}
				if (col == 1)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Point2f(p[ij]));
				}
				if (col == 0)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Point2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 1, 0, 0, 0); //��
		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, 0) = unused[idx[i]];
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 3:
	{
		std::vector<cv::Point2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row == 2)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Point2f(p[ij]));
				}
				if (row == 1)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Point2f(p[ij]));
				}
				if (row == 0)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Point2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}
		cv::copyMakeBorder(chessboard, chesstemp, 1, 0, 0, 0, 0, 0); //��
		for (int i = 0; i < chesstemp.cols; i++)
		{
			chesstemp.at<int>(0, i) = unused[idx[i]];
		}
		chessboard = chesstemp.clone();
		break;
	}
	default:
		break;
	}
	// cout << "cout <<" << endl;
	return chessboard;
}

bool FindCorners::is_element_in_vector(vector<float> v, int element)
{
	vector<float>::iterator it;
	it = find(v.begin(), v.end(), element);
	if (it != v.end())
	{
		return true;
	}
	else
	{
		return false;
	}
}

vector<Point2f> FindCorners::predictCorners(std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2,
											std::vector<cv::Point2f> &p3,std::vector<cv::Point2f> &pred)
{
	cv::Point2f v1, v2;
	float a1, a2, a3;
	float s1, s2, s3;
	pred.resize(p1.size());
	for (int i = 0; i < p1.size(); i++)
	{
		// compute vectors
		v1 = p2[i] - p1[i];
		v2 = p3[i] - p2[i];
		// predict angles
		a1 = atan2(v1.y, v1.x);
		a2 = atan2(v2.y, v2.x);
		a3 = 2.0 * a2 - a1;
		//cout << "a3 = " << a3 << ", size = "<< p1.size() << endl;

		//predict scales
		s1 = norm2d(v1);
		s2 = norm2d(v2);
		s3 = 2 * s2 - s1;
		pred[i] = p3[i] + 0.75 * s3 * cv::Point2f(cos(a3), sin(a3));
		//cout << "pred[i] = " << pred[i] << endl;
	}
	return pred;
	//cout << "test " << endl;
}

void FindCorners::assignClosestCorners(std::vector<cv::Point2f> &cand,std::vector<cv::Point2f> &pred, std::vector<int> &idx)
{
	if (cand.size() < pred.size())
	{
		idx.resize(1);
		idx[0] = -1;
		//cout << "return error if not enough candidates are available." << endl;
	}
	idx.resize(pred.size());

	//build distance matrix
	cv::Mat D = cv::Mat::zeros(cand.size(), pred.size(), CV_32FC1);
	float mind = FLT_MAX;
	for (int i = 0; i < D.cols; i++) //������
	{
		cv::Point2f delta;
		for (int j = 0; j < D.rows; j++)
		{
			delta = cand[j] - pred[i];
			float s = norm2d(delta);
			D.at<float>(j, i) = s;
			if (s < mind)
			{
				mind = s;
			}
		}
	}

	// search greedily for closest corners
	for (int k = 0; k < pred.size(); k++)
	{
		bool isbreak = false;
		for (int i = 0; i < D.rows; i++)
		{
			for (int j = 0; j < D.cols; j++)
			{
				if (abs(D.at<float>(i, j) - mind) < 10e-10)
				{
					idx[j] = i;
					for (int m = 0; m < D.cols; m++)
					{
						D.at<float>(i, m) = FLT_MAX;
					}
					for (int m = 0; m < D.rows; m++)
					{
						D.at<float>(m, j) = FLT_MAX;
					}
					isbreak = true;
					break;
				}
			}
			if (isbreak == true)
				break;
		}
		mind = FLT_MAX;
		for (int i = 0; i < D.rows; i++)
		{
			for (int j = 0; j < D.cols; j++)
			{
				if (D.at<float>(i, j) < mind)
				{
					mind = D.at<float>(i, j);
				}
						}
		}
	}
}
