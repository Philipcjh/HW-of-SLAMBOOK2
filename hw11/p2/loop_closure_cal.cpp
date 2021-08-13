#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry> 
#include <unistd.h>

using namespace std;
using namespace Eigen;

// path to groundtruth file
string groundtruth_file = "./groundtruth.txt";
// 设置检测的间隔，使得检测具有稀疏性的同时覆盖整个环境
int delta = 50; 
// 齐次变换矩阵差的范数，小于该值时认为位姿非常接近
double threshold = 0.4; 

int main(int argc, char **argv) {

  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
  vector<string> times;
  
  ifstream fin(groundtruth_file);
  if (!fin) {
    cout << "cannot find trajectory file at " << groundtruth_file << endl;
    return 1;
  }
  
  int num = 0;
  
  while (!fin.eof()) {
    string time_s;
    double tx, ty, tz, qx, qy, qz, qw;
    fin >> time_s >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
    Twr.pretranslate(Vector3d(tx, ty, tz));
    // 相当于从第150个位姿开始，这是因为标准轨迹的记录早于照片拍摄(前120个位姿均无对应照片)
    if (num > 120 && num % delta == 0){
      times.push_back(time_s);
      poses.push_back(Twr);
    }
    num++;
  }
  cout << "read total " << num << " pose entries" << endl;
  cout << "selected total " << poses.size() << " pose entries" << endl;
  

  //设置检测到回环后重新开始检测图片间隔数量
  cout << "**************************************************" << endl;
  cout << "Detection Start!!!" << endl;
  cout << "**************************************************" << endl;
  for (size_t i = 0 ; i < poses.size() - delta; i += delta){
    for (size_t j = i + delta ; j < poses.size() ; j++){
      Matrix4d Error = (poses[i].inverse() * poses[j]).matrix() - Matrix4d::Identity();
      if (Error.norm() < threshold){
	cout << "第" << i << "张照片与第" << j << "张照片构成回环" << endl;
	cout << "位姿误差为" << Error.norm() << endl;
	cout << "第" << i << "张照片的时间戳为" << endl << times[i] << endl;
	cout << "第" << j << "张照片的时间戳为" << endl << times[j] << endl;
	cout << "**************************************************" << endl;
	break;
      }
    } 
  }
  cout << "Detection Finish!!!" << endl;
  cout << "**************************************************" << endl;
  return 0;
}

