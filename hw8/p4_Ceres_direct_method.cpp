#include <iostream>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include <chrono>
#include <boost/format.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// paths
string left_file = "./left.png";
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");    // other files

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param se3
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::Vector6d &se3
);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param se3
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::Vector6d &se3
);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

class DirectMethodLuminaError : public ceres::SizedCostFunction<1, 6> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DirectMethodLuminaError(Vector2d px_ref, double depth_ref, Mat img1, Mat img2) :
            _px_ref(px_ref), _depth_ref(depth_ref), _img1(img1), _img2(img2) {}

    virtual ~DirectMethodLuminaError() {}

    virtual bool Evaluate(
      double const* const* parameters, double *residuals, double **jacobians) const {

        Eigen::Map<const Eigen::Matrix<double,6,1>> se3(*parameters);	

        Sophus::SE3d T = Sophus::SE3d::exp(se3);
	Eigen::Vector3d point_ref =_depth_ref * Eigen::Vector3d((_px_ref[0] - cx) / fx, (_px_ref[1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T * point_ref;
	double u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy;
	
        residuals[0] = GetPixelValue(_img1, _px_ref[0], _px_ref[1]) - GetPixelValue(_img2, u, v);

        if(jacobians != NULL) {
            if(jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 1>> J(jacobians[0]);
	      
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
			    0.5 * (GetPixelValue(_img2, u + 1 , v ) - GetPixelValue(_img2, u - 1 , v )),
			    0.5 * (GetPixelValue(_img2, u , v + 1 ) - GetPixelValue(_img2, u , v - 1 ))
		);
		
		J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();
            }
        }
        return true;
    }
    
    
private:
    const Vector2d _px_ref;
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
    
    Sophus::Vector6d se3;
      
    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, se3);
	cout<< "现在是第" << i <<"次"<<endl;
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, se3);
    }

    return 0;
}

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::Vector6d &se3
){
    ceres::Problem problem;
    for(int j=0; j<px_ref.size(); ++j) {
	ceres::CostFunction *cost_function;
	cost_function = new DirectMethodLuminaError(px_ref[j], depth_ref[j], img1, img2);
	problem.AddResidualBlock(cost_function, NULL, se3.data());  
    }
  
    auto t1 = chrono::steady_clock::now();
    ceres::Solver::Options options;
    options.dynamic_sparsity = true;
    options.max_num_iterations = 100;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    options.dogleg_type = ceres::SUBSPACE_DOGLEG;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << "\n";
  
    std::cout << "estimated pose: \n" << Sophus::SE3d::exp(se3).matrix() << std::endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp in ceres cost time: " << time_used.count() << " seconds." << endl;
    
  
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::Vector6d &se3
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
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, se3);
    }
}

