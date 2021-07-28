#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace cv;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// paths
string left_file = "./left.png";
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");    // other files

// g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T
);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T
);

// bilinear interpolation
inline double GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return double(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};

//1元边，测量值维度是1,对应测量值类型为灰度差，顶点对应的数据类型都是VertexPose
class EdgeProjection : public g2o::BaseUnaryEdge<1, double, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjection(const Eigen::Vector2d &px_ref, const double depth_ref, const Mat img1, const Mat img2)
	  : _px_ref(px_ref), _depth_ref(depth_ref), _img1(img1), _img2(img2) {}

  virtual void computeError() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d point_ref =_depth_ref * Eigen::Vector3d((_px_ref[0] - cx) / fx, (_px_ref[1] - cy) / fy, 1);
    Eigen::Vector3d point_cur = T * point_ref;
    double u1 = fx * point_cur[0] / point_cur[2] + cx, v1 = fy * point_cur[1] / point_cur[2] + cy;
    _error(0, 0) = _measurement - GetPixelValue(_img2, u1, v1);
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d point_ref =_depth_ref * Eigen::Vector3d((_px_ref[0] - cx) / fx, (_px_ref[1] - cy) / fy, 1);
    Eigen::Vector3d point_cur = T * point_ref;
    double u1 = fx * point_cur[0] / point_cur[2] + cx, v1 = fy * point_cur[1] / point_cur[2] + cy;
    
    Eigen::Matrix<double, 6, 1> J;
	      
    double X = point_cur[0], Y = point_cur[1], Z = point_cur[2];
    double Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
		
    Matrix26d J_pixel_xi;
    Eigen::Vector2d J_img_pixel;

    J_pixel_xi(0, 0) = fx * Z_inv;
    J_pixel_xi(0, 1) = 0;
    J_pixel_xi(0, 2) = -fx * X * Z2_inv;
    J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
    J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
    J_pixel_xi(0, 5) = -fx * Y * Z_inv;

    J_pixel_xi(1, 0) = 0;
    J_pixel_xi(1, 1) = fy * Z_inv;
    J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
    J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
    J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
    J_pixel_xi(1, 5) = fy * X * Z_inv;

    J_img_pixel = Eigen::Vector2d(
	0.5 * (GetPixelValue(_img2, u1 + 1 , v1 ) - GetPixelValue(_img2, u1 - 1 , v1 )),
        0.5 * (GetPixelValue(_img2, u1 , v1 + 1 ) - GetPixelValue(_img2, u1 , v1 - 1 ))
    );

    // total jacobian
    J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();
   
    _jacobianOplusXi << J[0], J[1], J[2], J[3], J[4], J[5];
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}

private:
    const Eigen::Vector2d _px_ref;
    const double _depth_ref;
    const Mat _img1;
    const Mat _img2;
};



int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;
    
    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    Sophus::SE3d T;
    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
	cout<< "现在是第" << i <<"次"<<endl;
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T);
    }

    return 0;
}

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T
){
  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;  
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
  vertex_pose->setId(0);
  vertex_pose->setEstimate(T);
  optimizer.addVertex(vertex_pose);

  // edges
  int index = 1;
  
  //新增部分：第一个相机作为顶点连接的边
  for (size_t i = 0; i < px_ref.size(); ++i) {
    
    EdgeProjection *edge = new EdgeProjection(px_ref[i], depth_ref[i], img1, img2);
    edge->setId(index);
    edge->setVertex(0, vertex_pose);
    edge->setMeasurement(GetPixelValue(img1, px_ref[i][0], px_ref[i][1]));
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    optimizer.addEdge(edge);
    index++;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated =\n" << vertex_pose->estimate().matrix() << endl;
  
  T = vertex_pose->estimate();
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T
){
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    
    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
	
	cout<<"pyramid"<< level+1 <<'\t';
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T);
    }
}