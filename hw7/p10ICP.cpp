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

void pose_estimation_3d3d(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t
);

//Ref:http://www.ceres-solver.org/nnls_tutorial.html#bundle-adjustment
struct ICPReprojectionError {
  ICPReprojectionError(Point3f pts1_3d, Point3f pts2_3d)
      : _pts1_3d(pts1_3d), _pts2_3d(pts2_3d) {}

  template <typename T>
  bool operator()(const T* const rotation_vector,
                  const T* const translation_vector,
                  T* residuals) const {
		    
    T p_transformed[3], p_origin[3];
    p_origin[0]=T(_pts2_3d.x);
    p_origin[1]=T(_pts2_3d.y);
    p_origin[2]=T(_pts2_3d.z);
    ceres::AngleAxisRotatePoint(rotation_vector, p_origin, p_transformed);
    
    //旋转后加上平移向量
    p_transformed[0] += translation_vector[0]; 
    p_transformed[1] += translation_vector[1]; 
    p_transformed[2] += translation_vector[2];

    //计算error
    residuals[0] = T(_pts1_3d.x) - p_transformed[0];
    residuals[1] = T(_pts1_3d.y) - p_transformed[1];
    residuals[2] = T(_pts1_3d.z) - p_transformed[2];
    return true;
  }

   // 3，3，3指输出维度（residuals）为3
   //待优化变量（rotation_vector，translation_vector）维度分别为3
   static ceres::CostFunction* Create(const Point3f _pts1_3d,
                                      const Point3f _pts2_3d) {
     return (new ceres::AutoDiffCostFunction<ICPReprojectionError, 3, 3, 3>(
                 new ICPReprojectionError(_pts1_3d, _pts2_3d)));
   }

  Point3f _pts1_3d;
  Point3f _pts2_3d;
};


// 通过引入Sophus库简化计算，并使用雅克比矩阵的解析解代替自动求导
class ICPSE3ReprojectionError : public ceres::SizedCostFunction<3, 6> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ICPSE3ReprojectionError(Eigen::Vector3d pts1_3d, Eigen::Vector3d pts2_3d) :
            _pts1_3d(pts1_3d), _pts2_3d(pts2_3d) {}

    virtual ~ICPSE3ReprojectionError() {}

    virtual bool Evaluate(
      double const* const* parameters, double *residuals, double **jacobians) const {

        Eigen::Map<const Eigen::Matrix<double,6,1>> se3(*parameters);	

        Sophus::SE3d T = Sophus::SE3d::exp(se3);

        Eigen::Vector3d Pc = T * _pts2_3d;

        Eigen::Vector3d error =  _pts1_3d - Pc;

        residuals[0] = error[0];
        residuals[1] = error[1];
	residuals[2] = error[2];

        if(jacobians != NULL) {
            if(jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> J(jacobians[0]);
	      
                double x = Pc[0];
                double y = Pc[1];
                double z = Pc[2];
		
		//雅克比矩阵推导看书187页公式(7.45),因为这里变换后的P'在计算error时是被减的，所以应该是(7.45)取负
                J(0,0) = -1; J(0,1) = 0; J(0,2) = 0; J(0,3) = 0; J(0,4) = -z; J(0,5) = y; 
		            J(1,0) = 0; J(1,1) = -1; J(1,2) = 0; J(1,3) = z; J(1,4) = 0; J(1,5) = -x;
		            J(2,0) = 0; J(2,1) = 0; J(2,2) = -1; J(2,3) = -y; J(2,4) = x; J(2,5) = 0;
            }
        }

        return true;
    }

private:
    const Eigen::Vector3d _pts1_3d;
    const Eigen::Vector3d _pts2_3d;
};


int main(int argc, char **argv){
  if (argc != 5) {
    cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 建立3D点
  Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts1, pts2;
  vector<Vector3d> pts1_eigen, pts2_eigen;

  for (DMatch m:matches) {
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)   // bad depth
      continue;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    pts1_eigen.push_back(Vector3d(p1.x * dd1, p1.y * dd1, dd1));
    pts2_eigen.push_back(Vector3d(p2.x * dd2, p2.y * dd2, dd2));
  }

  cout << "3d-3d pairs: " << pts1.size() << endl;
  Mat R, t;
  pose_estimation_3d3d(pts1, pts2, R, t);
  cout << "ICP via SVD results: " << endl;
  cout << "R = " << R << endl;
  cout << "t = " << t << endl;
  cout << "R_inv = " << R.t() << endl;
  cout << "t_inv = " << -R.t() * t << endl;
  cout << endl;
  
  //ceres求解PnP, 使用自动求导
  cout << "以下是ceres求解（自动求导）" << endl;
  double r_ceres[3]={0,0,0};
  double t_ceres[3]={0,0,0};
  
  ceres::Problem problem;
  for (size_t i = 0; i < pts1.size(); ++i) {
    ceres::CostFunction* cost_function =
	ICPReprojectionError::Create(pts1[i],pts2[i]);
    problem.AddResidualBlock(cost_function, 
			     nullptr /* squared loss */,
			     r_ceres,
			     t_ceres);
  }
  
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
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
  
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in ceres cost time: " << time_used.count() << " seconds." << endl;
  
  //ceres求解PnP, 使用雅克比矩阵的解析解
  cout << "以下是ceres求解（雅克比矩阵给出解析解）" << endl;
  
  Sophus::Vector6d se3;
  se3<<0,0,0,0,0,0;// 初始化非常重要
  
  ceres::Problem problem_;
  for(int i=0; i<pts1_eigen.size(); ++i) {
      ceres::CostFunction *cost_function;
      cost_function = new ICPSE3ReprojectionError(pts1_eigen[i], pts2_eigen[i]);
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

void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
  Point3f p1, p2;     // center of mass
  int N = pts1.size();
  for (int i = 0; i < N; i++) {
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = Point3f(Vec3f(p1) / N);
  p2 = Point3f(Vec3f(p2) / N);
  vector<Point3f> q1(N), q2(N); // remove the center
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }

  // compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }
  cout << "W=" << W << endl;

  // SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  cout << "U=" << U << endl;
  cout << "V=" << V << endl;

  Eigen::Matrix3d R_ = U * (V.transpose());
  if (R_.determinant() < 0) {
    R_ = -R_;
  }
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

  // convert to cv::Mat
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}