#include "DetectCorners.h"
#include <limits.h>
#include<fstream>
#include<numeric>
using namespace cv;
using namespace std;


inline float distv(cv::Vec2f& a, cv::Vec2f &b)
{
	return std::sqrt((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]));
}

inline float mean_l(std::vector<float> &resultSet)
{
	double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
	double mean = sum / resultSet.size(); //��ֵ  
	return mean;
}

inline float stdev_l(std::vector<float> &resultSet, float &mean)
{
	double accum = 0.0;
	mean = mean_l(resultSet);
	std::for_each(std::begin(resultSet), std::end(resultSet), [&](const double d) {
		accum += (d - mean)*(d - mean);
	});
	double stdev = sqrt(accum / (resultSet.size() - 1)); //���� 
	return stdev;
}

inline float stdevmean(std::vector<float> &resultSet)
{
	float stdvalue, meanvalue;

	stdvalue = stdev_l(resultSet, meanvalue);

	return stdvalue / meanvalue;
}

Mat FindCorners::structureRecovery(Corners mcorners){

  //Mat chessboards = Mat::zeros(3,3,CV_32F);
  vector<Mat> chessboards;
  for(int i = 0; i < mcorners.p.size(); i++){
    if(i%100 == 0){
      cout << i <<"/" << mcorners.p.size()<<endl;
    }
	  Mat chessboard = initChessboard(mcorners,i);
	  if(chessboard.empty()||chessboardEnergy(chessboard,mcorners)>0){
	    continue;
	  }
	 
		std::vector<cv::Mat>  proposal(4);
	  vector<float>p_energy(4);
	 
    float energy = chessboardEnergy(chessboard, mcorners);
	  for(int j = 0; j < 4; j++){
	    proposal[j].push_back(growChessboard(chessboard, mcorners, j));
	    p_energy[j] = chessboardEnergy(proposal[j],mcorners);
	  }

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
		}
		else
		{
			break;
		}
     //find best proposal
    // auto smallest_energy = std::min_element(std::begin(p_energy),std::end(p_energy));
    // //print the smallest element and its position:
    // cout << "min_element is " << *smallest_energy 
    //   <<" at position "<< std::distance(std::begin(p_energy),std::end(p_energy)) << endl;
    // int min_idx = std::distance(std::begin(p_energy),std::end(p_energy));
    // float min_value = *smallest_energy;
    //   //accept best proposal, if energy is reduced
    // if(p_energy[min_idx] < energy){
    //   chessboard = proposal[min_idx];
    //   return chessboard;
    //     // if(false){
    //     // }
    // }
    // else{
    //   break;
    // }
  
    
    //if chessboard has low energy (corresponding to high quality)
    // if(chessboardEnergy(chessboard,mcorners) < -10){
    //   //check if new chessboard proposal overlaps with existing chessboards
    //   Mat overlap = Mat::zeros(chessboards.size(),2,CV_32F);

    //   for(int i = 0; i < chessboards.size();i++){
    //     for(int j = 0; j < chessboards[i].rows; j++){
    //       for(int k = 0; k < chessboards[i].cols; k++){
    //         for(int m = 0; m < chessboard.rows; m++){
    //           for(int n = 0; n < chessboard.cols; n++){
    //             if(chessboards[i].at<float>(j,k) == chessboard.at<float>(m,n)){
    //               overlap.at<float>(i,0) = 1;
    //               overlap.at<float>(i,1) = chessboardEnergy(chessboards[i],mcorners);
    //               break;
    //             }
    //           }
    //         }
    //       }
    //     }
    //   }

      // int Num = 0;
      // Mat overlap_temp = overlap.colRange(0,1);
      // //add chessboard(and replace overlapping if neccessary)
      // //判断 overlap(:,1)中是否有非零元素，any(A) 若有非零元素即返回true
      // for(int i = 0; i < overlap.rows; i++){
      //   if(overlap.at<float>(i,0) == 0){
      //     Num++;//如果所有元素都是0，那么Num应该是等于rows个数
      //   }
      // }

      // if(Num == overlap.rows - 1){
      //   chessboards.push_back(chessboard);//if ~any(overlap(:,1)) chessboards{end+1} = chessboard
      // }
      // else{
      //   int idx;
      //   //  if(findValue(overlap_temp,1) == true)
      //   for(int i = 0; i < overlap_temp.rows; i++){
      //     if(overlap_temp.at<float>(i,0) == 1){
      //       idx = i;
      //       break;
      //     }  
      //   }
      //   int temp_idx;
      //   if(overlap.at<float>(idx,1) != 0 ){
      //     temp_idx = 1;
      //   }
      //   else{
      //     temp_idx = 0;
      //   }        
      //   if(temp_idx <= chessboardEnergy(chessboard,mcorners)){
      //     //chessboards[idx] = ;
      //     chessboards.push_back(chessboard);
      //   }
      // }
    //}
  }
}
	  
	


bool findValue(const cv::Mat &mat, float value) {
    for(int i = 0;i < mat.rows;i++) {
        const float* row = mat.ptr<float>(i);
        if(std::find(row, row + mat.cols, value) != row + mat.cols)
            return true;
    }
    return false;
}

Mat initChessboard(Corners& mcorners,int idx){
    // return if not enough corners
    if(mcorners.p.size()<9){
        cout << "The seed does not have the enough corners.Please retry!";
        //chessboard.release();//return empty!
	//return -1;
    }

    //extract feature index and orientation(central element)
    Mat chessboard = -1 * Mat::ones(3,3,CV_32S);

    Vec2f v1 = mcorners.v1[idx];
    Vec2f v2 = mcorners.v2[idx];
    chessboard.at<float>(1,1) = idx;    
    vector<float> dist1(2);
    vector<float> dist2(6);
 
    //find left/right/top/bottom neighbor
    // float top_neighbor = 0.0;
    // float top_min_dist = 0.0;

    // float bottom_neighbor = 0.0;
    // float bottom_min_dist = 0.0;
    
    // float left_neighbor = 0.0;
    // float left_min_dist = 0.0;
   
    // float right_neighbor = 0.0;
    // float right_min_dist = 0.0;
    

    //right
      directionalNeighbor(idx, +1 * v1, chessboard, mcorners, chessboard.at<int>(1, 2), dist1[0]);
	directionalNeighbor(idx, -1 * v1, chessboard, mcorners, chessboard.at<int>(1, 0), dist1[1]);
	directionalNeighbor(idx, +1 * v2, chessboard, mcorners, chessboard.at<int>(2, 1), dist2[0]);
	directionalNeighbor(idx, -1 * v2, chessboard, mcorners, chessboard.at<int>(0, 1), dist2[1]);


    std::cout << "Mat:Chessboard = " << chessboard << endl; 
    
    //find top-left/top-right/bottom-left/bottom-right neighbors

    // float top_left_neighbor = 0.0;
    // float top_left_min_dist = 0.0;
    
    // float top_right_neighbor = 0.0;
    // float top_right_min_dist = 0.0;
   
    // float bottom_left_neighbor = 0.0;
    // float bottom_left_min_dist = 0.0;
    
    // float bottom_right_neighbor = 0.0;
    // float bottom_right_min_dist = 0.0;

    //top-left
    directionalNeighbor(chessboard.at<int>(1, 0), -1 * v2, chessboard, mcorners, chessboard.at<int>(0, 0), dist2[2]);
	  directionalNeighbor(chessboard.at<int>(1, 0), +1 * v2, chessboard, mcorners, chessboard.at<int>(2, 0), dist2[3]);
	  directionalNeighbor(chessboard.at<int>(1, 2), -1 * v2, chessboard, mcorners, chessboard.at<int>(0, 2), dist2[4]);
	  directionalNeighbor(chessboard.at<int>(1, 2), +1 * v2, chessboard, mcorners, chessboard.at<int>(2, 2), dist2[5]);
    
    //distance_1 deviation 

    //distance_2 deviation
//     int dist_temp1;
//     int dist_temp2;
//     for(int i = 0; i < distance_1.size() && i < distance_2.size();i++){  
//       if(distance_1[i] == INT_MAX){
//         dist_temp1 = 1;
//       } 
//       else{
//         dist_temp1 = 0;
//       }
      
//       if(distance_2[i] == INT_MAX){
//         dist_temp2 = 1;
//       }
//       else{
//         dist_temp2 = 0;
//       } 
//     }

//     if(dist_temp1 || dist_temp2 ||deviation(distance_1)/mean(distance_1) > 0.3 || deviation(distance_2)/mean(distance_2) > 0.3)
// 	  cout << "It is not homogenously distributed, please try again." << endl;
       
//     return chessboard;
    
// }
    bool sigood = false;
		sigood = sigood||(dist1[0]<0) || (dist1[1]<0);
		sigood = sigood || (dist2[0]<0) || (dist2[1]<0) || (dist2[2]<0) || (dist2[3]<0) || (dist2[4]<0) || (dist2[5]<0);
		

		sigood = sigood || (stdevmean(dist1) > 0.3) || (stdevmean(dist2) > 0.3);

		if (sigood == true)
		{
			chessboard.release();
			return chessboard;
		}
		return chessboard;
}
//均值 mean
// float mean(vector<float> distance){
//     float sum = std::accumulate(std::begin(distance),std::end(distance),0.0);
//     float mean = sum / distance.size();
//     return mean;  
// }
// //标准差 deviation 
// float deviation(vector<float>distance){
//     float sum = std::accumulate(std::begin(distance),std::end(distance),0.0);
//     float mean = sum / distance.size();
    
//     float accum = 0.0;
//     std::for_each(std::begin(distance),std::end(distance), [&](const float d){
//       accum += (d-mean)*(d-mean);
//     });
    
//     double stdev = sqrt(accum/(distance.size()-1));
//     return stdev;
  
// }
int directionalNeighbor(int idx, cv::Vec2f v, cv::Mat chessboard, Corners& corners, int& neighbor_idx, float& min_dist)
{

#if 1
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
  				unused[xy] = -1;
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
	  	i++;
  	}

    std::vector<float> dist_edge;
	  std::vector<float> dist_point;

  	cv::Vec2f idxp = cv::Vec2f(corners.p[idx].x, corners.p[idx].y);
	// direction and distance to unused corners
	  for (int i = 0; i < unused.size(); i++)
	  {
  		int ind = unused[i];
	  	cv::Vec2f diri = cv::Vec2f(corners.p[ind].x, corners.p[ind].y) - idxp;
	  	float disti = diri[0] * v[0] + diri[1] * v[1];

	  	cv::Vec2f de = diri - disti*v;
	  	dist_edge.push_back(distv(de,cv::Vec2f(0, 0)));
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
				unused[xy] = -1;//flag the used idx
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

		cv::Vec2f de = diri - disti*v;
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
    
    
    
//     vector<Point2f> dir;
//     vector<float> dist;
//     vector<float>dist_edge;
//     for(int i = 0;i < unused.size();i++){
//       dir.push_back(corners.p[i] - corners.p[idx]);
//       dist.push_back(dir[i].x*v[0] + dir[i].y * v[1]);
//       Point2f dist_edgePoints = Point2f(dir[i].x - dist[i] * v[0],dir[i].y - dist[i] * v[1]);
//       dist_edge.push_back(sqrt(dist_edgePoints.x * dist_edgePoints.x + dist_edgePoints.y * dist_edgePoints.y));
//     }
    
//     vector<float> dist_point(dist);
    
//     for(int i = 0;i < unused.size();i++){
//       if(dist[i] < 0)
// 	      dist_point[i] = INT_MAX;      
//     }
    
//     vector<float> temp;
	
//     for(int i = 0; i < dist_point.size();i++){
//       temp[i] = dist_point[i] + 5 * dist_edge[i];
//     }
      
//     auto smallest = std::min_element(begin(temp), end(temp)); 
//     int min_idx = distance(std::begin(temp), smallest);
//     cout << "min element is " << *smallest<< " at position " << min_idx << endl;  
//     float min_dist = *smallest;
    
//     neighbor_idx = unused[min_idx];
    
// //     for(int i = 0; i < min_idx;i++){
// //       neighbor_idx.push_back(unused[i]);
// //     }
// }

float chessboardEnergy(cv::Mat chessboard, Corners& corners)
{
	//energy: number of corners
	float E_corners = -1 * chessboard.size().area();
	//energy: structur
	float E_structure = 0;
	//walk through rows
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols-2; j++)
		{
			std::vector<cv::Vec2f> x;
			float E_structure0 = 0;
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(i, k);
				x.push_back(corners.p[n]);
			}
			E_structure0 = distv(x[0] + x[2] - 2 * x[1], cv::Vec2f(0,0));
			float tv = distv(x[0] - x[2], cv::Vec2f(0, 0));
			E_structure0 = E_structure0 / tv;
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	//walk through columns
	for (int i = 0; i < chessboard.cols; i++)
		for (int j = 0; j < chessboard.rows-2; j++)
		{
			std::vector<cv::Vec2f> x;
			float E_structure0 = 0;
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(k, i);
				x.push_back(corners.p[n]);
			}
			E_structure0 = distv(x[0] + x[2] - 2 * x[1], cv::Vec2f(0, 0));
			float tv = distv(x[0] - x[2], cv::Vec2f(0, 0));
			E_structure0 = E_structure0 / tv;
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	// final energy
	float E = E_corners + 1 *chessboard.size().area()*E_structure;

	return E;
}
// float chessboardEnergy(Mat chessboard,Corners corners){
//   float E_corners = -1 * chessboard.rows * chessboard.cols; 
//   float E_structure = 0.0;

//   vector<Point2f> x;
//   for(int j = 0; j < chessboard.rows; j++){
//     for(int k = 0; k < chessboard.cols-2; k++){
//       x.push_back(corners.p[chessboard.at<float>(j,k)]);
//       x.push_back(corners.p[chessboard.at<float>(j,k+1)]);
//       x.push_back(corners.p[chessboard.at<float>(j,k+2)]); 
//       E_structure = max(E_structure,norm2d(x[0] + x[2] - 2 * x[1])/norm2d(x[0]-x[2]));
//     }
//   }
  
//   for(int j = 0; j < chessboard.cols; j++){
//     for(int k = 0; k < chessboard.rows-2; k++){
//       x.push_back(corners.p[chessboard.at<float>(k,j)]);
//       x.push_back(corners.p[chessboard.at<float>(k+1,j)]);
//       x.push_back(corners.p[chessboard.at<float>(k+2,j)]); 
//       E_structure = max(E_structure,norm2d(x[0] + x[2] - 2 * x[1])/norm2d(x[0]-x[2]));
//     }
//   }  
  
//   float E = E_corners + 1 * chessboard.rows * chessboard.cols * E_structure;
//   return E;
// }
    
float norm2d(cv::Point2f o){
    return sqrt(o.x*o.x + o.y*o.y);
}

Mat growChessboard(Mat chessboard, Corners& corners, int border_type){
  //if there do not exist any chessboards.
  if(chessboard.empty()){
    std::cout << "Chessboard is Empty.So sad=.=" << endl;
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
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;
			}
		}

	int nsize = unused.size();

	for (int i = 0; i < nsize; )
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
  std::vector<cv::Vec2f> cand;
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
		std::vector<cv::Vec2f> p1, p2, p3,pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == chessboard.cols - 3)
				{				
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (col == chessboard.cols - 2)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (col == chessboard.cols - 1)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 0, 1, 0,0);

		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, chesstemp.cols - 1) = unused[idx[i]];//��
		}
		chessboard = chesstemp.clone();

		break;
	}
  case 1:
	{
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row == chessboard.rows - 3)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (row == chessboard.rows - 2)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (row == chessboard.rows - 1)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
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
			chesstemp.at<int>(chesstemp.rows - 1, i) = unused[idx[i]];//��
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 2:
	{
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == 2)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (col == 1)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (col == 0)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 1, 0, 0, 0);//��
		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, 0) = unused[idx[i]];
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 3:
	{
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row ==  2)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (row == 1)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (row == 0)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}
		cv::copyMakeBorder(chessboard, chesstemp, 1, 0, 0, 0, 0, 0);//��
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
	return chessboard;
}
//   vector<Point2f> cand;
//   for(int i = 0; i < unused.size();i++){
//     cand.push_back(p[unused[i]]);
//   }
//   Mat out;
//   vector<Point2f> p1;
//   vector<Point2f> p2;
//   vector<Point2f> p3;
//   vector<Point2f> pred;
//   vector<float> index;
//   vector<float> unused_Temp;
//   Mat M2;
//   switch(border_type)
//   {
//           //chessboard = [chessboard unused(idx)']
//     case 1:
      
//       for(int i = 0; i < chessboard.rows;i++){
// 	      p1.push_back(p[chessboard.at<float>(i,chessboard.cols - 3)]);
// 	      p2.push_back(p[chessboard.at<float>(i,chessboard.cols - 2)]);
// 	      p3.push_back(p[chessboard.at<float>(i,chessboard.cols - 1)]);
//       }
//       //predict Corners
//       pred = predictCorners(p1,p2,p3);
//       index = assignClosestCorners(cand,pred);

//       if(is_element_in_vector(index,0) == false){
	      
// 	      for(int i = 0; i < index.size(); i++){
// 	        unused_Temp.push_back(unused[index[i]]);
// 	      }
// 	//Vector<float> transfer to Matrix
// 	      M2  = Mat(unused_Temp.size(),1,CV_32F);
// 	      memcpy(M2.data,unused_Temp.data(),unused_Temp.size()*sizeof(float));
// 	      hconcat( chessboard,M2.t(), out);
//       }
//       break;
//       //      chessboard = [chessboard; unused(idx)];
//     case 2:
      
//       for(int i = 0; i < chessboard.cols;i++){
// 	      p1.push_back(p[chessboard.at<float>(chessboard.rows - 3,i)]);
// 	      p2.push_back(p[chessboard.at<float>(chessboard.rows - 2,i)]);
// 	      p3.push_back(p[chessboard.at<float>(chessboard.rows - 1, i)]);
//       }
//       //predict Corners
//       pred = predictCorners(p1,p2,p3);
//       index = assignClosestCorners(cand,pred);

//       if(is_element_in_vector(index,0) == false){
	      
// 	      for(int i = 0; i < index.size(); i++){
// 	        unused_Temp.push_back(unused[index[i]]);
// 	      }
// 	//Vector<float> transfer to Matrix
// 	      M2=Mat(unused_Temp.size(),1,CV_32F);
// 	      memcpy(M2.data,unused_Temp.data(),unused_Temp.size()*sizeof(float));
// 	      vconcat(chessboard, M2, out);
//       }
//       break;
//       //      chessboard = [unused(idx)' chessboard];
//     case 3:
      
//       for(int i = 0; i < chessboard.rows;i++){
// 	      p1.push_back(p[chessboard.at<float>(i,2)]);
// 	      p2.push_back(p[chessboard.at<float>(i,1)]);
// 	      p3.push_back(p[chessboard.at<float>(i,0)]);
//       }
//       //predict Corners
//       pred = predictCorners(p1,p2,p3);
//       index = assignClosestCorners(cand,pred);

//       if(is_element_in_vector(index,0) == false){
// 	      for(int i = 0; i < index.size(); i++){
// 	        unused_Temp.push_back(unused[index[i]]);
// 	      }
// 	//Vector<float> transfer to Matrix
//     	  M2 = Mat(unused_Temp.size(),1,CV_32F);
//     	  memcpy(M2.data,unused_Temp.data(),unused_Temp.size()*sizeof(float));
// 	      hconcat( M2.t(),chessboard,out);
//       }
//       break;
//       //      chessboard = [unused(idx); chessboard];
//     case 4:
//       for(int i = 0; i < chessboard.cols;i++){
// 	      p1.push_back(p[chessboard.at<float>(2, i)]);
// 	      p2.push_back(p[chessboard.at<float>(1, i)]);
// 	      p3.push_back(p[chessboard.at<float>(0, i)]);
//       }
//       //predict Corners
//       pred = predictCorners(p1,p2,p3);
//       index = assignClosestCorners(cand,pred);

//       if(is_element_in_vector(index,0) == false){
// 	      for(int i = 0; i < index.size(); i++){
// 	        unused_Temp.push_back(unused[index[i]]);
// 	      }
// 	//Vector<float> transfer to Matrix
// 	      M2=Mat(unused_Temp.size(),1,CV_32F);
// 	      memcpy(M2.data,unused_Temp.data(),unused_Temp.size()*sizeof(float));
// 	      vconcat( M2,chessboard, out);
//       }  
//       break; 
//   } 
//   return out;
// }

bool is_element_in_vector(vector<float> v,int element){  
    vector<float>::iterator it;  
    it = find(v.begin(),v.end(),element);  
    if (it != v.end()){  
        return true;  
    }  
    else{  
        return false;  
    }  
} 

void predictCorners(std::vector<cv::Vec2f>& p1, std::vector<cv::Vec2f>& p2, 
	std::vector<cv::Vec2f>& p3, std::vector<cv::Vec2f>& pred)
{
	cv::Vec2f v1, v2;
	float a1, a2, a3;
	float s1, s2, s3;
	pred.resize(p1.size());
	for (int i = 0; i < p1.size(); i++)
	{
		// compute vectors
		v1 = p2[i] - p1[i];
		v2 = p3[i] - p2[i];
		// predict angles
		a1 = atan2(v1[1], v1[0]);
		a2 = atan2(v1[1], v1[0]);
		a3 = 2.0 * a2 - a1;

		//predict scales
		s1 = distv(v1, cv::Vec2f(0, 0));
		s2 = distv(v2, cv::Vec2f(0, 0));
		s3 = 2 * s2 - s1;
		pred[i] = p3[i] + 0.75*s3*cv::Vec2f(cos(a3), sin(a3));
	}
}
// vector<Point2f> predictCorners(vector<Point2f> p1, vector<Point2f> p2, vector<Point2f>p3){
//   vector<Point2f> v1;
//   vector<Point2f> v2;
//   //v1 = p2-p1;v2 = p3 - p2;
//   for(int i = 0; i< p1.size(); i++){
//     v1.push_back(p2[i] - p1[i]);
//     v2.push_back(p3[i] - p2[i]);
// //     v1.push_back(Vec2f(p2[i].x - p1[i].x,p2[i].y - p1[i].y));
// //     v2.push_back(Vec2f(p3[i].x - p2[i].x,p3[i].y - p2[i].y));
//   }
  
//   vector<float> a1;
//   vector<float> a2;
//   vector<float> a3;
//   //a1 = atan2(v1(:,2),v1(:,1)); a2 = atan2(v2(:,2),v2(:,1))
//   //a3 = 2 * a2 - a1
//   for(int i = 0; i < v1.size();i++){
//     a1.push_back(atan2(v1[i].y,v1[i].x));
//     a2.push_back(atan2(v2[i].y,v2[i].x));
//     a3.push_back(2 * a2[i] - a1[i]);
//   }
  
//   //% predict scales:
//   vector<float> s1;
//   vector<float> s2;
//   vector<float> s3;
//   for(int i = 0; i< v1.size(); i++){
//     s1.push_back(sqrt(v1[i].x * v1[i].x + v1[i].y * v1[i].y));
//     s2.push_back(sqrt(v2[i].x * v2[i].x + v2[i].y * v2[i].y));
//     s3.push_back(2 * s2[i] - s1[i]);
//   }
// //  predict p3 (the factor 0.75 ensures that under extreme
// //  distortions (omnicam) the closer prediction is selected)
//   vector<Point2f> s3_Point;
//   vector<Point2f> a3_Angle;
//   vector<Point2f> pred;
//   for(int i = 0; i < s3.size();i++){
//     s3_Point.push_back(cv::Point2f(s3[i],s3[i]));
//     a3_Angle.push_back(cv::Point2f(cos(a3[i]),sin(a3[i])));
//     pred.push_back(cv::Point2f(p3[i].x + 0.75 * s3_Point[i].x 
//     + a3_Angle[i].x, p3[i].y + 0.75 * s3_Point[i].y + a3_Angle[i].y));
//     }
//     return pred;
   
// }

// vector<float> assignClosestCorners(vector<Point2f>cand, vector<Point2f>pred){
//   int idx;
//   if(cand.size() < pred.size()){
//     idx = 0;
//     cout << "return error if not enough candidates are available." << endl;
//   }
  
//   //build distance matrix
  
//   vector<Point2f> delta;
//   vector<Point2f> pred_temp;
//   Mat D = Mat::zeros(cand.size(),pred.size(),CV_32F);
//   int rows = cand.size();
//   int cols = pred.size();
  
//   for(int i = 0; i < cols; i++){
//     for(int j = 0; j < rows; j++){
//       pred_temp.push_back(cv::Point2f(pred[i].x,pred[i].y));
//       delta.push_back(cand[i]-pred_temp[i]);
//       D.at<float>(j,i) = norm2d(delta[i]);
//       pred_temp.clear();
//       delta.clear();
//     }
//   }
  
//   //search greedily for closest corners
//   double a,b;
//   cv::Point min_point,max_point;
//   vector<float>index_Mat(cand.size(),0);

//   for(int i = 0; i < pred.size(); i++){
//     minMaxLoc(D,&a,&b,&min_point,&max_point);
// //     cout << "a = " << a <<"; "<< "b = "<<"; "<< "min_point = "<< min_point
// //     << "; " << "max_point = "<< max_point <<endl;
//     index_Mat[min_point.y] = min_point.x;
//     for(int j = 0; j < D.cols; j++){
//       D.at<float>(min_point.x,j) = INT_MAX;
//     }
//     for(int k = 0;k < D.rows;k++){
//       D.at<float>(k,min_point.y) = INT_MAX;
//     }
//   }
//   return index_Mat;
// }

void assignClosestCorners(std::vector<cv::Vec2f>&cand, std::vector<cv::Vec2f>&pred, std::vector<int> &idx)
{
	//return error if not enough candidates are available
	if (cand.size() < pred.size())
	{
		idx.resize(1);
		idx[0] = -1;
		return;
	}
	idx.resize(pred.size());

	//build distance matrix
	cv::Mat D = cv::Mat::zeros(cand.size(), pred.size(), CV_32FC1);
	float mind = FLT_MAX;
	for (int i = 0; i < D.cols; i++)//������
	{
		cv::Vec2f delta;
		for (int j = 0; j < D.rows; j++)
		{
			delta = cand[j] - pred[i];
			float s = distv(delta, cv::Vec2f(0, 0));
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
						D.at<float>(m,j) = FLT_MAX;
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


      
    
   
    
   
    
      
    
   
    
   