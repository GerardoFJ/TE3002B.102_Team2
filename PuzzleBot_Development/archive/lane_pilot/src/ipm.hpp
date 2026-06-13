// Inverse Perspective Mapping (IPM) for the Puzzlebot ground plane.
//
// A one-time camera-to-ground homography (calibrated with a checkerboard laid
// flat on the floor, see tools/calibrate_ground_homography.py) lets us warp the
// rectified camera image into a metric "bird's-eye" top-down image and convert
// any bird's-eye pixel back to a real (X, Y) position in meters in the robot's
// base_link frame. This is what gives the rest of the stack a notion of "how
// many centimeters ahead" the line is.
//
// Frame conventions (REP-103, base_link):
//   X = forward distance from the base_link origin (the wheel axle midpoint)
//   Y = lateral, +left
//
// Bird's-eye image layout:
//   row 0      = far  (X = x_max)        bottom row = near (X = x_min)
//   col 0      = left (+y_half)          right col  = right (-y_half)

#ifndef LANE_PILOT__IPM_HPP_
#define LANE_PILOT__IPM_HPP_

#include <cmath>
#include <string>

#include <opencv2/core.hpp>

namespace lane_pilot
{

// Geometry / resolution of the metric bird's-eye image.
struct BirdViewSpec
{
  double x_min = 0.05;    // nearest forward distance shown (m)
  double x_max = 0.70;    // farthest forward distance shown (m)
  double y_half = 0.35;   // half lateral extent (m); view spans [-y_half, +y_half]
  double mpp = 0.0025;    // meters per pixel

  int width() const
  {
    return std::max(1, static_cast<int>(std::lround((2.0 * y_half) / mpp)));
  }
  int height() const
  {
    return std::max(1, static_cast<int>(std::lround((x_max - x_min) / mpp)));
  }
};

class Ipm
{
public:
  // Load H_img2ground (3x3 homography mapping a rectified image pixel (u,v,1)
  // to homogeneous ground coords (X*w, Y*w, w) in base_link meters) from an
  // OpenCV YAML written by tools/calibrate_ground_homography.py.
  // Returns false and fills *err on failure.
  bool load(const std::string & homography_file, std::string * err = nullptr);

  // (Re)build the warp matrix for the given bird's-eye geometry.
  void set_view(const BirdViewSpec & spec);
  const BirdViewSpec & view() const {return spec_;}

  bool ready() const {return ready_;}

  // Warp a full-resolution rectified camera image (BGR or gray) into the metric
  // bird's-eye view. Pixels with no source data (outside the camera FOV, e.g.
  // the near blind-spot rows) come out black.
  void warp(const cv::Mat & src, cv::Mat & bird) const;

  // Bird's-eye pixel (col,row) -> ground (X,Y) in base_link meters.
  void bird_to_ground(double col, double row, double & X, double & Y) const;
  // Ground (X,Y) in base_link meters -> bird's-eye pixel (col,row).
  void ground_to_bird(double X, double Y, double & col, double & row) const;

private:
  void rebuild();

  cv::Matx33d H_img2ground_ = cv::Matx33d::eye();  // image px -> ground meters
  cv::Matx33d warp_img2bird_ = cv::Matx33d::eye(); // image px -> bird px
  BirdViewSpec spec_;
  bool have_h_ = false;
  bool ready_ = false;
};

}  // namespace lane_pilot

#endif  // LANE_PILOT__IPM_HPP_
