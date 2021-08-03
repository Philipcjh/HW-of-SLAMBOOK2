#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rotation.h"
#include "random.h"

/// 从文件读入BAL dataset
class BALProblem {
public:
    /// load bal data from text file
    explicit BALProblem(const std::string &filename, const std::string &filename_w, bool use_quaternions = false);

    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    /// save results to text file
    void WriteToFile(const std::string &filename) const;

    /// save results to ply pointcloud
    void WriteToPLYFile(const std::string &filename) const;

    void Normalize();

    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);

    int camera_block_size() const { return use_quaternions_ ? 13 : 12; }

    int point_block_size() const { return 3; }

    int num_cameras() const { return num_cameras_; }

    int num_points() const { return num_points_; }

    int num_observations() const { return num_observations_; }

    int num_parameters() const { return num_parameters_; }

    const int *point_index() const { return point_index_; }

    const int *camera_index() const { return camera_index_; }

    const double *observations() const { return observations_; }

    const double *parameters() const { return parameters_; }

    const double *cameras() const { return parameters_; }

    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; }

    /// camera参数的起始地址
    double *mutable_cameras() { return parameters_; }

    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }

    double *mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    double *mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * point_block_size();
    }

    const double *camera_for_observation(int i) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    const double *point_for_observation(int i) const {
        return points() + point_index_[i] * point_block_size();
    }

private:
    void CameraToAngelAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;

    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    bool use_quaternions_;

    int *point_index_;      // 每个observation对应的point index
    int *camera_index_;     // 每个observation对应的camera index
    double *observations_;
    double *parameters_;
};

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1)
        std::cerr << "Invalid UW data file. ";
}


int main(int argc, char **argv) {
   
    BALProblem bal_problem(argv[1], argv[2]);
    return 0;
}

BALProblem::BALProblem(const std::string &filename, const std::string &filename_w, bool use_quaternions) {
  
    std::cout << "Data Processing Started!"<< std::endl;
    
    FILE *fptr = fopen(filename.c_str(), "r");
    FILE *fptr_w = fopen(filename_w.c_str(), "w");
    
    
    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    };
    
    if (fptr_w == NULL) {
        std::cerr << "Error: unable to open file " << filename_w;
        return;
    };
    

    // This wil die horribly on invalid files. Them's the breaks.
    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);
    
    fprintf(fptr_w, "%d %d %d\n", num_cameras_, num_points_, num_observations_);

    std::cout << "Header: " << num_cameras_
              << " " << num_points_
              << " " << num_observations_
              << std::endl;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", camera_index_ + i);
        FscanfOrDie(fptr, "%d", point_index_ + i);
	
	fprintf(fptr_w, "%d %d", camera_index_[i], point_index_[i]);
	
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
	    fprintf(fptr_w, " %1f", observations_[2 * i + j]);
        }
        fprintf(fptr_w, "\n");
    }
     
    for (int i = 0; i < num_parameters_; ++i) {
        FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    
    fclose(fptr);
    
    for (int i = 0; i < num_parameters_; ++i){
	if (i <= 9 * num_cameras_){
	  // fprint cameras parameters
	  if (i % 9 == 0 && i != 0){
	      fprintf(fptr_w, "%lf\n", parameters_ [i - 3]);
	      fprintf(fptr_w, "%lf\n", 0.f);
	      fprintf(fptr_w, "%lf\n", 0.f);
	  }
	  fprintf(fptr_w, "%lf\n", parameters_ [i]);
	}
	else{
	  fprintf(fptr_w, "%lf\n", parameters_ [i]);
	}
	
    }
    
    
    fclose(fptr_w);
    
    std::cout << "Data Processing Successfully!"<< std::endl;

}