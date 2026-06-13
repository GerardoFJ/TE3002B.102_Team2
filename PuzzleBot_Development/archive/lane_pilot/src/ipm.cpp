#include "ipm.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace lane_pilot
{

bool Ipm::load(const std::string & homography_file, std::string * err)
{
  cv::FileStorage fs(homography_file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    if (err) {*err = "could not open homography file: " + homography_file;}
    return false;
  }
  cv::Mat H;
  fs["H_img2ground"] >> H;
  fs.release();
  if (H.empty() || H.rows != 3 || H.cols != 3) {
    if (err) {*err = "H_img2ground missing or not 3x3 in " + homography_file;}
    return false;
  }
  cv::Mat Hd;
  H.convertTo(Hd, CV_64F);
  H_img2ground_ = cv::Matx33d(reinterpret_cast<double *>(Hd.data));
  have_h_ = true;
  rebuild();
  return ready_;
}

void Ipm::set_view(const BirdViewSpec & spec)
{
  spec_ = spec;
  rebuild();
}

void Ipm::rebuild()
{
  if (!have_h_) {
    ready_ = false;
    return;
  }
  const double W = static_cast<double>(spec_.width());
  const double H = static_cast<double>(spec_.height());

  // A_g2b: ground (X,Y,1) -> bird pixel (col,row,1). Affine, derived from the
  // bird's-eye layout described in ipm.hpp:
  //   col = (W-1)/2          - (W-1)/(2*y_half)         * Y
  //   row = x_max*(H-1)/dx   - (H-1)/dx                 * X     , dx = x_max-x_min
  const double dx = std::max(1e-9, spec_.x_max - spec_.x_min);
  const double cY = (W - 1.0) / (2.0 * spec_.y_half);
  const double cX = (H - 1.0) / dx;
  cv::Matx33d A_g2b(
    0.0, -cY, (W - 1.0) / 2.0,
    -cX, 0.0, spec_.x_max * cX,
    0.0, 0.0, 1.0);

  // image px -> bird px = A_g2b * (image px -> ground)
  warp_img2bird_ = A_g2b * H_img2ground_;
  ready_ = true;
}

void Ipm::warp(const cv::Mat & src, cv::Mat & bird) const
{
  cv::warpPerspective(
    src, bird, cv::Mat(warp_img2bird_),
    cv::Size(spec_.width(), spec_.height()),
    cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

void Ipm::bird_to_ground(double col, double row, double & X, double & Y) const
{
  const double W = static_cast<double>(spec_.width());
  const double H = static_cast<double>(spec_.height());
  X = spec_.x_max - row * (spec_.x_max - spec_.x_min) / std::max(1.0, H - 1.0);
  Y = spec_.y_half - col * (2.0 * spec_.y_half) / std::max(1.0, W - 1.0);
}

void Ipm::ground_to_bird(double X, double Y, double & col, double & row) const
{
  const double W = static_cast<double>(spec_.width());
  const double H = static_cast<double>(spec_.height());
  col = (spec_.y_half - Y) * std::max(1.0, W - 1.0) / (2.0 * spec_.y_half);
  row = (spec_.x_max - X) * std::max(1.0, H - 1.0) / std::max(1e-9, spec_.x_max - spec_.x_min);
}

}  // namespace lane_pilot
