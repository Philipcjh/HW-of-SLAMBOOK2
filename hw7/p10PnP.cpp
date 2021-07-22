#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace cv;

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

//Ref:http://www.ceres-solver.org/nnls_tutorial.html#bundle-adjustment
struct PnPReprojectionError {
  PnPReprojectionError(Point2f pts_2d, Point3f pts_3d)
      : _pts_2d(pts_2d), _pts_3d(pts_3d) {}

  template <typename T>
  bool operator()(const T* const rotation_vector,
                  const T* const translation_vector,
                  T* residuals) const {
		    
    T p_transformed[3], p_origin[3];
    p_origin[0]=T(_pts_3d.x);
    p_origin[1]=T(_pts_3d.y);
    p_origin[2]=T(_pts_3d.z);
    ceres::AngleAxisRotatePoint(rotation_vector, p_origin, p_transformed);
    
    //旋转后加上平移向量
    p_transformed[0] += translation_vector[0]; 
    p_transformed[1] += translation_vector[1]; 
    p_transformed[2] += translation_vector[2];

    //归一化
    T xp = p_transformed[0] / p_transformed[2];
    T yp = p_transformed[1] / p_transformed[2];

    
    double fx=520.9, fy=521.0, cx=325.1, cy=249.7;
    // Compute final projected point position.
    T predicted_x = fx * xp + cx;
    T predicted_y = fy * yp + cy;

    // The error is the difference between the predicted and observed position.
    residuals[0] = T(_pts_2d.x) - predicted_x;
    residuals[1] = T(_pts_2d.y) - predicted_y;
    return true;
  }

   // 2，3，3指输出维度（residuals）为2
   //待优化变量（rotation_vector，translation_vector）维度分别为3
   static ceres::CostFunction* Create(const Point2f _pts_2d,
                                      const Point3f _pts_3d) {
     return (new ceres::AutoDiffCostFunction<PnPReprojectionError, 2, 3, 3>(
                 new PnPReprojectionError(_pts_2d, _pts_3d)));
   }

  Point2f _pts_2d;
  Point3f _pts_3d;
};

// 通过引入Sophus库简化计算，并使用雅克比矩阵的解析解代替自动求导
class PnPSE3ReprojectionError : public ceres::SizedCostFunction<2, 6> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PnPSE3ReprojectionError(Eigen::Vector2d pts_2d, Eigen::Vector3d pts_3d) :
            _pts_2d(pts_2d), _pts_3d(pts_3d) {}

    virtual ~PnPSE3ReprojectionError() {}

    virtual bool Evaluate(
      double const* const* parameters, double *residuals, double **jacobians) const {

        Eigen::Map<const Eigen::Matrix<double,6,1>> se3(*parameters);	

        Sophus::SE3d T = Sophus::SE3d::exp(se3);

        Eigen::Vector3d Pc = T * _pts_3d;

        Eigen::Matrix3d K;
        double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
        K << fx, 0, cx, 
	     0, fy, cy, 
	     0, 0, 1;

        Eigen::Vector2d error =  _pts_2d - (K * Pc).hnormalized();

        residuals[0] = error[0];
        residuals[1] = error[1];

        if(jacobians != NULL) {
            if(jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[0]);
	      
                double x = Pc[0];
                double y = Pc[1];
                double z = Pc[2];

                double x2 = x*x;
                double y2 = y*y;
                double z2 = z*z;
		
		//雅克比矩阵推导看书187页公式(7.46)
                J(0,0) = -fx/z;
                J(0,1) =  0;
                J(0,2) =  fx*x/z2;
                J(0,3) =  fx*x*y/z2;
                J(0,4) = -fx-fx*x2/z2;
                J(0,5) =  fx*y/z;
                J(1,0) =  0;
                J(1,1) = -fy/z;
                J(1,2) =  fy*y/z2;
                J(1,3) =  fy+fy*y2/z2;
                J(1,4) = -fy*x*y/z2;
                J(1,5) = -fy*x/z;
            }
        }

        return true;
    }

private:
    const Eigen::Vector2d _pts_2d;
    const Eigen::Vector3d _pts_3d;
};


int main(int argc, char **argv){
  if (argc != 5) {
    cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 建立3D点
  Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  vector<Vector3d> pts_3d_eigen;
  vector<Vector2d> pts_2d_eigen;
  
  for (DMatch m:matches) {
    ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)   // bad depth
      continue;
    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));//第一个相机观察到的3D点坐标
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);//特征点在第二个相机中的投影
    pts_3d_eigen.push_back(Vector3d(p1.x * dd, p1.y * dd, dd));
    pts_2d_eigen.push_back(Vector2d(keypoints_2[m.trainIdx].pt.x, keypoints_2[m.trainIdx].pt.y));
  }

  cout << "3d-2d pairs: " << pts_3d.size() << endl;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
  Mat R;
  cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;
  cout << endl;
  //ceres求解PnP, 使用自动求导
  cout << "以下是ceres求解（自动求导）" << endl;
  double r_ceres[3]={0,0,0};
  double t_ceres[3]={0,0,0};
  
  ceres::Problem problem;
  for (size_t i = 0; i < pts_2d.size(); ++i) {
    ceres::CostFunction* cost_function =
	PnPReprojectionError::Create(pts_2d[i],pts_3d[i]);
    problem.AddResidualBlock(cost_function, 
			     nullptr /* squared loss */,
			     r_ceres,
			     t_ceres);
  }
  
  t1 = chrono::steady_clock::now();
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  
  Mat r_ceres_cv=(Mat_<double>(3, 1) <<r_ceres[0], r_ceres[1], r_ceres[2]);
  Mat t_ceres_cv=(Mat_<double>(3, 1) <<t_ceres[0], t_ceres[1], t_ceres[2]);
  cv::Rodrigues(r_ceres_cv, R);
  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t_ceres_cv << endl;
  
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in ceres cost time: " << time_used.count() << " seconds." << endl<< endl;
  
  //ceres求解PnP, 使用雅克比矩阵的解析解
  cout << "以下是ceres求解（雅克比矩阵给出解析解）" << endl;
  
  Sophus::Vector6d se3;
  se3<<0,0,0,0,0,0;// 初始化非常重要
  
  ceres::Problem problem_;
  for(int i=0; i<pts_3d_eigen.size(); ++i) {
      ceres::CostFunction *cost_function;
      cost_function = new PnPSE3ReprojectionError(pts_2d_eigen[i], pts_3d_eigen[i]);
      problem_.AddResidualBlock(cost_function, NULL, se3.data());
      
  }
  
  t1 = chrono::steady_clock::now();
  ceres::Solver::Options options_;
  options_.dynamic_sparsity = true;
  options_.max_num_iterations = 100;
  options_.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options_.minimizer_type = ceres::TRUST_REGION;
  options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options_.trust_region_strategy_type = ceres::DOGLEG;
  options_.minimizer_progress_to_stdout = true;
  options_.dogleg_type = ceres::SUBSPACE_DOGLEG;

  ceres::Solver::Summary summary_;
  ceres::Solve(options_, &problem_, &summary_);
  std::cout << summary_.BriefReport() << "\n";
  
  std::cout << "estimated pose: \n" << Sophus::SE3d::exp(se3).matrix() << std::endl;
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in ceres cost time: " << time_used.count() << " seconds." << endl;
  
  return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}