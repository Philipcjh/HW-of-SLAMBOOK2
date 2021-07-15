#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "usage: feature_extraction img" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  assert(img.data != nullptr);

  //-- 初始化
  std::vector<KeyPoint> keypoints_orb,keypoints_sift,keypoints_surf,keypoints_kaze;
  Ptr<FeatureDetector> detector_orb = ORB::create(1000);
  Ptr<FeatureDetector> detector_sift= SIFT::create(1000);
  Ptr<FeatureDetector> detector_surf= xfeatures2d::SURF::create(400);
  Ptr<FeatureDetector> detector_kaze= KAZE::create();

  //-- Orb特征点
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector_orb->detect(img, keypoints_orb);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout<<"number of keypoints="<<keypoints_orb.size()<<endl;
  cout<<"time of orb="<<time_used.count()<<endl;
  cout<<"***************************************"<<endl;
  
  Mat outimg_orb;
  drawKeypoints(img,keypoints_orb, outimg_orb, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("ORB features", outimg_orb);
  
  //-- Sift特征点
  t1 = chrono::steady_clock::now();
  detector_sift->detect(img, keypoints_sift);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout<<"number of keypoints="<<keypoints_sift.size()<<endl;
  cout<<"time of sift="<<time_used.count()<<endl;
  cout<<"***************************************"<<endl;
  
  Mat outimg_sift;
  drawKeypoints(img,keypoints_sift, outimg_sift, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("SIFT features", outimg_sift);
  
  //-- Surf特征点
  t1 = chrono::steady_clock::now();
  detector_surf->detect(img, keypoints_surf);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout<<"number of keypoints="<<keypoints_surf.size()<<endl;
  cout<<"time of surf="<<time_used.count()<<endl;
  cout<<"***************************************"<<endl;
  
  Mat outimg_surf;
  drawKeypoints(img,keypoints_surf, outimg_surf, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("SURF features", outimg_surf);
  
  //-- Kaze特征点
  t1 = chrono::steady_clock::now();
  detector_surf->detect(img, keypoints_kaze);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout<<"number of keypoints="<<keypoints_kaze.size()<<endl;
  cout<<"time of kazef="<<time_used.count()<<endl;
  cout<<"***************************************"<<endl;
  
  Mat outimg_kaze;
  drawKeypoints(img,keypoints_kaze, outimg_kaze, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("KAZE features", outimg_kaze);
  
  waitKey(0);

  
  return 0;
}
