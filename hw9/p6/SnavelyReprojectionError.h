#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),
                                                                           observed_y(observation_y) {}

    template<typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }
    
    
    // camera : 12 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length in x direction , [7] second order radial distortion , [8] forth order radial distortion
    // [9-11] : camera parameter, [9] focal length in y direction, [10] first coefficient tangential distortion, [11] scond coefficient tangential distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions) {
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center fo distortion
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion
        const T &k1 = camera[7];
        const T &k2 = camera[8];

        T r2 = xp * xp + yp * yp;
        T radial_distortion = T(1.0) + r2 * (k1 + k2 * r2);
	
	// Apply first and second coefficient of tangential distortion
	const T &p1 = camera[10];
	const T &p2 = camera[11];
	
	T xy = xp * yp;
	T x2 = xp * xp;
	T y2 = yp * yp;
	
	T tangential_distortion_x = T(2.0) * p1 * xy + p2 * (r2 + T(2.0) * x2);
	T tangential_distortion_y = p1 * (r2 + T(2.0) * y2) + T(2.0) * p2 * xy;

        const T &fx = camera[6];
	const T &fy = camera[9];
        predictions[0] = fx * (radial_distortion * xp + tangential_distortion_x);
        predictions[1] = fy * (radial_distortion * yp + tangential_distortion_y);

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 12, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

#endif // SnavelyReprojection.h

