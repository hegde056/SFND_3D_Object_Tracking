# SFND : Camera Based 3D Object Tracking 
Camera Based 3D Object Tracking implemented using C++ as part of project submission for Camera section of Sensor Fusion Nanodegree (Udacity).

-------------
### Solution Report : 

<p align="center">
<img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/Result_sample.PNG" width="500" height="300" /></p> 


- ####  FP.1 Match 3D Objects
	- Task:  Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.
	- Implementation : 
		```
      void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
      {
          int prevBoxSize = prevFrame.boundingBoxes.size();
          int currBoxSize = currFrame.boundingBoxes.size();
          int ptBoxScore[prevBoxSize][currBoxSize] = {};

          //iterate over k-pt matches/outerloop
          for(auto itr = matches.begin();itr!=matches.end()-1;++itr)
          {
              //Prev kpt ,pt
              cv::KeyPoint prevKeyPt = prevFrame.keypoints[itr->queryIdx];
              cv::Point prevPt = cv::Point(prevKeyPt.pt.x,prevKeyPt.pt.y);

              //Curr kpt ,pt
              cv::KeyPoint currKeyPt = currFrame.keypoints[itr->trainIdx];
              cv::Point currPt = cv::Point(currKeyPt.pt.x,currKeyPt.pt.y);

              //to get a list of boxIds for bounding boxes enclosing the points (prev and curr)
              std::vector<int> prevBoxIDList , currBoxIDList;

              //populate BoxIDList if pt in bounding box roi
              for (int i=0;i< prevBoxSize;++i)
              {
                  if(prevFrame.boundingBoxes[i].roi.contains(prevPt))
                  {
                      prevBoxIDList.push_back(i);
                  }
              }
              for (int j=0;j< currBoxSize;++j)
              {
                  if(currFrame.boundingBoxes[j].roi.contains(currPt))
                  {
                      currBoxIDList.push_back(j);
                  }
              }

              for(auto prevID:prevBoxIDList)
              {
                  for(auto currID:currBoxIDList)
                  {
                      ptBoxScore[prevID][currID]+=1;
                  }
              }


          }//eol for pt matches

          //to find highest count in current frame for each box in prevframe
          for(int i = 0;i< prevBoxSize;++i)
          {
              int maxCnt = 0;
              int maxID = 0;

              for(int j=0;j< currBoxSize;++j)
              {
                  if(ptBoxScore[i][j] > maxCnt)
                  {
                      maxCnt = ptBoxScore[i][j];
                      maxID = j;
                  }
              }
              bbBestMatches[i] = maxID;
          }
      }
		```



- ####  FP.2 Compute Lidar-based TTC
 	- Task:  Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.
	- Implementation :
	  	<p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/TTC_Lidar_draw.png" width="500" height="300" />  </p> 
    <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/TTC_Lidar_formula.png" width="500" height="300" />  </p> 
    Only those lidar points within the ego lane are considered for TTC calculation. Additionally to get a fairly accurate value, the average distances are considered for previous and current lidar points and used in the TTC calculation formula. 
		```
      void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                           std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
      {
          // ...
          double dt = 1.0/frameRate   ; //time btw 2 measurements in seconds
          double laneWidth = 4.0 ;//assumed width of the ego lane 

          //find closest dist to lidar pts within ego lane 


          std::vector<double> xPrev;
          for(auto it = lidarPointsPrev.begin();it!=lidarPointsPrev.end();++it)
          {
              if(abs(it->y) <= laneWidth/2.0)
                  xPrev.push_back(it->x);
          }
          std::vector<double> xCurr;
          for(auto it = lidarPointsCurr.begin();it!=lidarPointsCurr.end();++it)
          {
              if(abs(it->y) <= laneWidth/2.0)
                  xCurr.push_back(it->x);
          }

          double minXPrev = 0 , minXCurr = 0 ;

          if(xPrev.size() > 0 && xCurr.size()>0)
          {
              for(auto x : xPrev)
                  minXPrev += x ;
              minXPrev = minXPrev/xPrev.size();
              for(auto y : xCurr)
                  minXCurr += y ;
              minXCurr = minXCurr/xCurr.size();
          }
          //compute TTC from both measurements
          TTC = minXCurr * dt / (minXPrev - minXCurr);

          std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TTC Lidar : "<<TTC<<" s"<<std::endl;

      }
       ```
 - ####  FP.3 Associate Keypoint Correspondences with Bounding Boxes
 	- Task : Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.
 	- Implementation  :  
 	 Associating keypoint correspondences to the bounding boxes is acheived by iterating over the keypoint matches and calculating the euclidean distance if the points fall in the bounding box ROI. Mean distance over all the calculated euclidean points is comuputed. Once again the iterating over the keypoint matches , those matches whose points fall into the bounding box ROI are added to the bounding box kptMatches, subject to the distance threshold of  *(Mean distance * 1.5)* to remove outliers far off from the mean.   
      ```
      void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
      {
          // ...

          //calculatig the mean kptMatches
          std::vector<double> euclideanDist_list ; 
          for(auto itr = kptMatches.begin();itr!= kptMatches.end();++itr)
          {
              cv::KeyPoint prevKeyPt = kptsPrev[itr->queryIdx];
              cv::KeyPoint currKeyPt = kptsCurr[itr->trainIdx];

              if(boundingBox.roi.contains(currKeyPt.pt))
              {
                  euclideanDist_list.push_back(cv::norm(currKeyPt.pt-prevKeyPt.pt));
              }

          }

          double distanceMean = std::accumulate(euclideanDist_list.begin(),euclideanDist_list.end(),0.0) / euclideanDist_list.size();

          //adding kptMatches to bounding boxes based on distance
          for(auto itr = kptMatches.begin();itr!= kptMatches.end();++itr)
          {
              cv::KeyPoint prevKeyPt = kptsPrev[itr->queryIdx];
              cv::KeyPoint currKeyPt = kptsCurr[itr->trainIdx];

              if(boundingBox.roi.contains(currKeyPt.pt))
              {
                  double dist = cv::norm(currKeyPt.pt-prevKeyPt.pt);

                  if(dist < distanceMean*1.5)
                  {
                      boundingBox.kptMatches.push_back(*itr);
                      boundingBox.keypoints.push_back(currKeyPt);
                  }
              }

          }

      }
      ```
  
 - ####  FP.4 Compute Camera-based TTC
 	- Task : Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame
  	- Implementation :
  	<p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/TTC_Camera_draw.png" width="500" height="300" />  </p> 
    <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/TTC_Camera_formula.png" width="500" height="300" />  </p>
  	 By observing relative height change on the image sensor, the time to collision can be calculated. The bounding boxes however cannot be directly used as they do not always reflect the true vehicle dimensions and the aspect ratio differs between images. Using bounding box height or width for TTC computation would thus lead to significant estimation errors. Hence Keypoint correspondences with bounding boxes are used.  The distance between all keypoints on the vehicle relative to each other helps compute a robust estimate of the height ratio in the TTC equation. To further improve the estimate minimizing the effect of outliers is to consider the median of the associated keypoints.
     
      ```
      void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                            std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
      {
          // ...
          // compute distance ratios between all matched keypoints
          vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
          for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
          { // outer kpt. loop

              // get current keypoint and its matched partner in the prev. frame
              cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
              cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

              for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
              { // inner kpt.-loop

                  double minDist = 100.0; // min. required distance

                  // get next keypoint and its matched partner in the prev. frame
                  cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
                  cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

                  // compute distances and distance ratios
                  double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
                  double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

                  if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
                  { // avoid division by zero

                      double distRatio = distCurr / distPrev;
                      distRatios.push_back(distRatio);
                  }
              } // eof inner loop over all matched kpts
          }     // eof outer loop over all matched kpts

          // only continue if list of distance ratios is not empty
          if (distRatios.size() == 0)
          {
              TTC = NAN;
              return;
          }

          // compute camera-based TTC from distance ratios
          //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

          //double dT = 1 / frameRate;
          //TTC = -dT / (1 - meanDistRatio);

          // STUDENT TASK (replacement for meanDistRatio)
          double medianDistRatio ;
          if(distRatios.size()%2 == 0) 
          {
              auto m1_it = distRatios.begin() + distRatios.size()/2;
              auto m2_it = distRatios.begin() + distRatios.size()/2 - 1 ;

              std::nth_element(distRatios.begin(),m1_it,distRatios.end() );
              auto e1 = *m1_it;

              std::nth_element(distRatios.begin(),m2_it,distRatios.end() );
              auto e2 = *m2_it;

              medianDistRatio = (e1 + e2)/2;
          }
          else
          {
              auto median_it = distRatios.begin() + distRatios.size()/2;
              std::nth_element(distRatios.begin(),median_it,distRatios.end() );
              medianDistRatio = *median_it;
          }
          double dT = 1 / frameRate;
          TTC = -dT / (1 - medianDistRatio);
          std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TTC Camera : "<<TTC<<" s"<<std::endl;
      }
        
  
 - ####  FP.5 Performance Evaluation 1	
 	- Task : Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.
  	- Evaluation : 
  	
    	*Frame* | *TTC Lidar (s)*
          ----|-----
          1	| 12.289
          2 | 13.354
          3 | 16.384
          4 | 14.076
          5 | 12.729
          6 | 13.751
          7 | 13.731
          8 | 13.790
          9 | 12.058
          10 | 11.864
          11 | 11.968
          12  | 9.887
          13 | 9.425
          14 | 9.302
          15 | 8.32
          16 | 8.89
          17 | 11.03
          18 | 8.535

   Some examples of implausable estimates :    
   Example 1  : 
  	<p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/F1.PNG" width="300" height="200" />  </p> 
    <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/F2.PNG" width="300" height="200" />  </p>
    <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/F3.PNG" width="300" height="200" />  </p>
    <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/F4.PNG" width="300" height="200" /> 
    </p>
    
   Example 2  : 
  	<p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/F16.PNG" width="300" height="200" />  </p>
    <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/F17.PNG" width="300" height="200" /> </p>
    <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/F18.PNG" width="300" height="200" /> 
    </p>
   
* Although in the FP.2 task the effect of outliers on TTC is minimized, it could not be completely removed. There still are outliers as can be seen above (eg. edges like mirrors of preceding vehicle) which accumulate to  inaccurate TTC measurement
* It can be infered from the images that the traffic scenario is at a stop sign where the vehicles are slowing down. Considering the TTC Lidar calculation formula from above, smaller moving distance causes higher change in TTC value. This also contributes to the implausible TTC measurement.
	

  
 - #### FP.6 Performance Evaluation 2	
 	- Task : Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.
  	- Evaluation :The detailed evaluation is documented in the spreadsheet : [FP_6_Evaluation.xlsx](https://github.com/SFND_3D_Object_Tracking/blob/master/FP_6_Evaluation.xlsx) . 
    <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/FP_6_BRISK_det_combinations.png" width="500" height="300" /> 
    </p> 
        <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/FP_6_FAST_det_combinations.png" width="500" height="300" /> 
    </p> 
        <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/FP_6_AKAZE_det_combinations.png" width="500" height="300" /> 
    </p> 
        <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/FP_6_SIFT_det_combinations.png" width="500" height="300" /> 
    </p> 
        <p align="center">
    <img src="https://github.com/hegde056/SFND_3D_Object_Tracking/blob/master/media/FP_6_SHITOMASI_det_combinations.png" width="500" height="300" /> 
    </p> 

 * The AKAZE and SIFT detectors were more stable and accurate compared to other combinations. 
 	Best combinations : 
    	`FAST + BRISK`,
       	`BRISK + ORB`,
        `AKAZE + AKAZE`,
        `SIFT + SIFT`
        
 * The ORB and HARRIS detectors created very unreliable results. 
 * As it can be seen from the graphs, the camera TTC calculation is way off at some frames. Keypoint mismatch between frames and slight shift of keypoints in next frame is observed in matching. This results to inaccurate calculation of TTC.
 * The model assumed for the TTC calculation is constant velocity model. However in the real scenario the preceding vehicle accelerates/decelerates and this affects the accuracy of the TTC calculation.  


-------------
	



