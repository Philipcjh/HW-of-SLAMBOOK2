#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#define MATRIX_SIZE 10

int main(int argc, char **argv) {
    //生成一个10×10的随机矩阵A
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> A= MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    
    //生成一个10×1的列向量b
    Matrix<double, MATRIX_SIZE, 1> b= MatrixXd::Random(MATRIX_SIZE, 1);    
    
    //直接求逆
    Matrix<double, MATRIX_SIZE, 1> x1=A.inverse()*b;
    cout<<"直接求逆得到的x为"<<endl;
    cout<<x1.transpose()<<endl;
    
    //QR分解
    Matrix<double, MATRIX_SIZE, 1> x2=A.householderQr().solve(b);
    cout<<"QR分解得到的x为"<<endl;
    cout<<x2.transpose()<<endl;
    
    //LU分解
    Matrix<double, MATRIX_SIZE, 1> x3=A.lu().solve(b);
    cout<<"LU分解得到的x为"<<endl;
    cout<<x3.transpose()<<endl;
    
    //LLT分解,此处A不满足正定，故求解出错
    Matrix<double, MATRIX_SIZE, 1> x4=A.llt().solve(b);
    cout<<"LLT分解得到的x为"<<endl;
    cout<<x4.transpose()<<endl;
    
    //LDLT分解,此处A不满足正半定或副半定，故求解出错
    Matrix<double, MATRIX_SIZE, 1> x5=A.ldlt().solve(b);
    cout<<"LDLT分解得到的x为"<<endl;
    cout<<x5.transpose()<<endl;
        
  return 0;
}