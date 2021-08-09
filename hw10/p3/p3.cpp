#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 7, 1> Vector7d;
typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 6, 6, RowMajor> Matrix6dRow;

typedef struct {
  int index = 0;
  Vector7d pose = Vector7d();
  Vector6d se3 = Vector6d();
}VertexSE3;

typedef struct {
  int index1 = 0;
  int index2 = 0;
  Vector7d pose = Vector7d();
  Matrix6d inform = Matrix6d().setIdentity();
  SE3d SE3 = SE3d();
}EdgeSE3;

string output = "./result.g2o";

Matrix6d JRInv(const SE3d &e) {
    Matrix6d J;
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log());
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());
    J = J * 0.5 + Matrix6d::Identity();
//     J = Matrix6d::Identity();    // try Identity if you want
    return J;
}

// 从平移向量+四元数转化为李代数
// 注意Eigen四元数构造形式为[w,x,y,z]，文件格式中的是[x,y,z,w]
Vector6d QuaterAndTransTose3(const Vector7d &pose){
    Quaterniond q(pose[6], pose[3], pose[4], pose[5]);
    q.normalize();
    Vector3d t(pose[0], pose[1], pose[2]);
    SE3d T(q,t);
    Vector6d se3 = T.log();
    return se3;
}

Vector7d se3ToQuaterAndTrans(const Vector6d &se3){
    SE3d T = SE3d::exp(se3);
    Vector7d pose;
    pose.block<3,1>(0,0) = T.translation();
    pose.block<4,1>(3,0) = T.unit_quaternion().coeffs();
    return pose;
}

class SE3Param : public ceres::LocalParameterization{
public:  
    SE3Param() {}
    virtual ~SE3Param() {};
    
    // 李代数左乘更新
    virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const{
	// 通过Eigen::Map将double数组映射到Eigen::Vector6d结构
	Eigen::Map<const Vector6d> se3(x);
	Eigen::Map<const Vector6d> delta_se3(delta);
	Eigen::Map<Vector6d> x_plus_delta_se3(x_plus_delta);
	
	x_plus_delta_se3 = (SE3d::exp(delta_se3) * SE3d::exp(se3)).log();
	return true;	
    }
    
    // x对delta的雅克比矩阵
    virtual bool ComputeJacobian(const double* x,
                               double* jacobian) const{
	Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J(jacobian);
	J.setIdentity();			 
	return true;			 
    }
    
    // 参数x的自由度（可能有冗余），对于se3是6,对于四元数是4
    virtual int GlobalSize() const { return 6; }  
    
    // delta_x所在正切空间的自由度，对于se3是6,对于四元数是3
    virtual int LocalSize() const { return 6; }
    
};

class PoseGraphLieAlgebra : public ceres::SizedCostFunction<6, 6, 6>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    PoseGraphLieAlgebra(SE3d Tr, Matrix6d information): 
	    _Tr(Tr), _information(information) {}
	    
    ~PoseGraphLieAlgebra(){}
    
    virtual bool Evaluate(double const* const* parameters, 
			  double *residuals, 
			  double **jacobians) const{
	Eigen::Map<const Vector6d> posei(parameters[0]);	
	SE3d Ti = SE3d::exp(posei);
	
	Eigen::Map<const Vector6d> posej(parameters[1]);
	SE3d Tj = SE3d::exp(posej);
	
	SE3d error_se3 = _Tr.inverse() * Ti.inverse() * Tj;
	
	Eigen::Map<Vector6d> residuals_vec(residuals);
	
	residuals_vec = error_se3.log();
	
	// Cholesky分解(平方根法)
	Matrix6d sqrt_information = Eigen::LLT<Matrix6d>(_information.inverse()).matrixLLT().transpose();
	
	// 左乘
	residuals_vec.applyOnTheLeft(sqrt_information);
	
	if (jacobians != NULL){
	    Matrix6d Jr_inv = JRInv(error_se3);
	    Matrix6d Ad_T2 = Tj.inverse().Adj();
	    if (jacobians[0] != NULL){
		Eigen::Map<Matrix6dRow> Ji(jacobians[0]);
		Ji = sqrt_information * (-Jr_inv) * Ad_T2;
	    }
	    
	    if (jacobians[1] != NULL){
		Eigen::Map<Matrix6dRow> Jj(jacobians[1]);
		Jj = sqrt_information * Jr_inv * Ad_T2;
	    }
	}
	
	return true;
    }
	    
private:
    const SE3d _Tr;
    const Matrix6d _information;
};

void Writeg2o(string &filename,
	      vector<VertexSE3> V, 
	      const vector<EdgeSE3> E){
    ofstream fout(filename);
    for (size_t i = 0; i < V.size(); ++i) {
        fout << "VERTEX_SE3:QUAT ";
        fout << V[i].index << " ";
	V[i].pose = se3ToQuaterAndTrans(V[i].se3);
	for (size_t j = 0; j < 7; ++j){
	  fout << V[i].pose[j] << " ";
	}
	fout << "\n";
    }
    for (size_t i = 0; i < E.size(); ++i) {
        fout << "EDGE_SE3:QUAT ";
        fout << E[i].index1 << " ";
	fout << E[i].index2 << " ";
	for (size_t j = 0; j < 7; ++j){
	  fout << E[i].pose[j] << " ";
	}
	for (size_t j = 0; j < 6; ++j){
	  for (size_t k = j; k < 6; ++k){
	    fout << E[i].inform(j, k) << " ";
	  }
	}
	fout << "\n";
    }
    
    fout.close();
}

int main (int argc, char **argv){ 
  
    google::InitGoogleLogging(argv[0]);
    
    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3_lie sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }
    
    vector<VertexSE3> AllVertexs;
    vector<EdgeSE3> AllEdges;
    
    ceres::Problem problem;
    ceres::LocalParameterization *local_param = new SE3Param();
    ceres::LossFunction *loss = new ceres::HuberLoss(1.0);
    
    while (!fin.eof()){
	string name;
	fin >> name;
	if (name == "VERTEX_SE3:QUAT"){
	    VertexSE3 V;
	    fin >> V.index;
	    
	    for (size_t i = 0; i < 7; ++i){
		fin >> V.pose[i]; 
	    }
	    V.se3 = QuaterAndTransTose3(V.pose);
	    AllVertexs.push_back(V);	    
	}
	else if (name == "EDGE_SE3:QUAT"){
	    EdgeSE3 E;
	    fin >> E.index1 >> E.index2;
	    
	    for (size_t i = 0; i < 7; ++i){
		fin >> E.pose[i]; 
	    }
	    
	    E.SE3 = SE3d::exp(QuaterAndTransTose3(E.pose));
	    
	    for (size_t i = 0; i < 6; ++i){
		for (size_t j = i; j < 6; ++j){
		    fin >> E.inform(i, j);
		    if (j != i){
			E.inform(j, i) = E.inform(i, j);
		    }
		}    
	    }
	    
	    AllEdges.push_back(E);
	    
	    
	    ceres::CostFunction *cost_function = new PoseGraphLieAlgebra(E.SE3, E.inform);
	    problem.AddResidualBlock(cost_function,
				     loss,
				     AllVertexs[E.index1].se3.data(),
				     AllVertexs[E.index2].se3.data());
	    problem.SetParameterization(AllVertexs[E.index1].se3.data(), local_param);
	    problem.SetParameterization(AllVertexs[E.index2].se3.data(), local_param);
	    
	}
    }
    fin.close();
    
    //固定第一个顶点
    problem.SetParameterBlockConstant(AllVertexs[0].se3.data());
    
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    
    ceres::Solver::Summary summary;             
    ceres::Solve ( options, &problem, &summary ); 
    
    cout << summary.FullReport() << endl;
    
    
    Writeg2o(output, AllVertexs, AllEdges);
    
    return 0;
}