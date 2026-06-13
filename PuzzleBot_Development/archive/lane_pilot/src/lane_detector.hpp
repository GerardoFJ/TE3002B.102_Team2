// Middle-line detector operating on the metric bird's-eye image.
//
// The lane has three roughly-parallel lines (two borders + the middle line the
// robot should follow). In the bird's-eye view these appear as parallel curves.
// We trace the middle line by marching row-by-row from the nearest visible row
// upward, picking the line cluster that stays continuous with the one we are
// already tracking. The track is seeded from the previous frame (temporal
// continuity) or, on cold start, from the cluster nearest the image center
// (robot assumed roughly centered on the middle line).
//
// Output is a list of centerline points already converted to (X, Y) meters in
// base_link, ready to be accumulated in the odom frame by lane_pilot_node.

#ifndef LANE_PILOT__LANE_DETECTOR_HPP_
#define LANE_PILOT__LANE_DETECTOR_HPP_

#include <vector>

#include <opencv2/core.hpp>

#include "ipm.hpp"

namespace lane_pilot
{

struct DetectorParams
{
  int black_threshold = 110;   // gray <= thr counts as a line pixel (dark line)
  bool invert = true;          // true: dark line on light floor (THRESH_BINARY_INV)
  int blur_ksize = 3;          // pre-threshold blur (odd, <3 disables)
  int open_px = 0;             // morphological OPEN kernel (0 disables) — removes speckle
  int close_px = 0;            // morphological CLOSE kernel (0 disables) — fills line gaps
  int min_cluster_px = 6;      // reject lateral runs narrower than this (thin seams)
  int max_cluster_px = 60;     // reject lateral runs wider than this (blobs)
  int row_step = 2;            // sample every Nth bird's-eye row
  double max_jump_m = 0.06;    // max lateral jump between sampled rows (m)
  int min_points = 6;          // fewer centerline points than this -> invalid

  // Region of interest (base_link meters) — only search for the line inside this
  // ground band, to ignore off-track floor and the background above the horizon
  // that the IPM warps into the far/side parts of the bird's-eye image.
  double roi_x_min = 0.05;     // nearest forward distance searched
  double roi_x_max = 0.55;     // farthest forward distance searched
  double roi_y_half = 0.25;    // lateral half-width searched
};

struct CenterPoint
{
  double x;   // base_link forward (m)
  double y;   // base_link lateral, +left (m)
};

struct DetectionResult
{
  std::vector<CenterPoint> points;     // sorted near -> far (ascending x)
  std::vector<cv::Point> bird_pts;     // selected pixels in bird image (debug)
  cv::Mat mask;                        // bird's-eye binary mask (debug; may be empty)
  double confidence = 0.0;             // fraction of sampled rows that tracked
  bool valid = false;
};

class LaneDetector
{
public:
  explicit LaneDetector(const DetectorParams & p)
  : params_(p) {}

  void set_params(const DetectorParams & p) {params_ = p;}

  // bird: metric bird's-eye BGR image from Ipm::warp; ipm: pixel<->ground map.
  DetectionResult detect(const cv::Mat & bird, const Ipm & ipm, bool keep_debug);

  // Forget the temporal track (call when the line is lost for a while).
  void reset() {have_prev_ = false;}

private:
  DetectorParams params_;
  bool have_prev_ = false;
  double prev_seed_col_ = -1.0;
};

}  // namespace lane_pilot

#endif  // LANE_PILOT__LANE_DETECTOR_HPP_
