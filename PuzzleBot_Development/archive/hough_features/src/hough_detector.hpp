// Hough-segment feature extractor for the Puzzlebot vision pipeline.
//
// ROS-free by design: this header pulls in only OpenCV and the standard
// library so the same detector can be reused by Phase 2's offline label
// generator and unit tests, not just the ROS node.
//
// Algorithm follows the reference notebook (PDF Reto TE3003B, Section 5)
// with one deliberate divergence noted in extract_hough_features().
//
// IMPORTANT — feature scale assumption
// ------------------------------------
// Four of the 14 features (CENTER_BOTTOM, VANISHING_X, MEAN_LENGTH,
// MAX_LENGTH) are in raw processing-frame pixels. Their values therefore
// depend on the node's `detect_scale` parameter. If detect_scale differs
// between when training data is recorded and when the model is run,
// downstream ML inference will silently produce garbage. Pin it.
//
// The other 10 features are normalized or counts and are scale-invariant.

#pragma once

#include <array>
#include <vector>

#include <opencv2/opencv.hpp>

namespace hough_features
{

// Fixed feature order shipped with the package (also mirrored in
// share/hough_features/feature_index.json for Python consumers).
enum FeatureIndex : int
{
  N_LINES         = 0,
  MEAN_ANGLE      = 1,
  STD_ANGLE       = 2,
  MEAN_ABS_ANGLE  = 3,
  MEAN_LENGTH     = 4,
  MAX_LENGTH      = 5,
  LEFT_COUNT      = 6,
  RIGHT_COUNT     = 7,
  BALANCE_LR      = 8,
  CENTER_BOTTOM   = 9,
  CENTER_ERROR    = 10,
  VANISHING_X     = 11,
  VANISHING_ERROR = 12,
  CONFIDENCE      = 13,
  N_FEATURES      = 14,
};

struct DetectorParams
{
  // Canny edge thresholds.
  double canny1 = 60.0;
  double canny2 = 160.0;

  // HoughLinesP parameters (PDF Section 4 defaults).
  int hough_threshold = 35;
  double min_line_length = 25.0;
  double max_line_gap = 15.0;

  // Vertical ROI crop (fraction of image height). Defaults keep the
  // bottom 60 % of the frame — enough to retain the vanishing point at
  // the horizon while removing sky/ceiling clutter.
  double roi_top_frac = 0.40;
  double roi_bottom_frac = 1.00;

  // Horizontal ROI crop. Defaults to full width; Hough is less sensitive
  // to frame brackets than adaptive thresholding so aggressive horizontal
  // crops are not needed by default.
  double roi_left_frac = 0.00;
  double roi_right_frac = 1.00;

  // Downscale factor applied BEFORE detection. 0.5 = 1280x720 → 640x360.
  // Performance and (more importantly) the scale of CENTER_BOTTOM,
  // VANISHING_X, MEAN_LENGTH, MAX_LENGTH all depend on this. Keep it
  // identical between training data collection and inference.
  double detect_scale = 0.5;

  // Safety bound for the O(N^2) vanishing-point intersection loop. PDF
  // uses 80; with detect_scale=0.5 the typical segment count is ~30-50
  // so this is rarely hit.
  int max_vp_pairs = 80;
};

struct DetectionResult
{
  // Segments in PROCESSING-frame coordinates (post-ROI shift, post-scale).
  // To overlay on the original image, the node has to undo the ROI offset
  // and the scale.
  std::vector<cv::Vec4f> segments;

  // Canny edges, only populated when detect() was called with
  // keep_edges_for_debug=true. Same coord system as `segments`.
  cv::Mat edges;

  // Fixed-order feature vector.
  std::array<float, N_FEATURES> features{};

  // The dimensions the detector actually ran at — after ROI + downscale.
  // The debug overlay uses these to map back into the original frame.
  int processed_width = 0;
  int processed_height = 0;

  // Where the processing ROI lives in the ORIGINAL (post-scale-up) image,
  // for debug overlay drawing.
  cv::Rect roi_in_original;
};

class HoughDetector
{
public:
  explicit HoughDetector(DetectorParams params);

  // Run the full pipeline on a BGR input image. `keep_edges_for_debug`
  // saves the intermediate Canny output into result.edges — the node
  // skips this when nobody is subscribed to the debug topic.
  DetectionResult detect(const cv::Mat & bgr, bool keep_edges_for_debug);

  void set_params(const DetectorParams & params) { params_ = params; }
  const DetectorParams & params() const { return params_; }

private:
  // PDF Section 5: linear interpolation x(y) along a segment, or NaN if
  // the segment is horizontal.
  static float line_x_at_y(
    float x1, float y1, float x2, float y2, float y_query);

  static float estimate_corridor_center_at_bottom(
    const std::vector<cv::Vec4f> & segments, int width, int height);

  static float estimate_vanishing_x(
    const std::vector<cv::Vec4f> & segments, int width, int height,
    int max_pairs);

  // PDF Section 5 returns a dict; we return a fixed-size std::array
  // indexed by FeatureIndex.
  static std::array<float, N_FEATURES> extract_hough_features(
    const std::vector<cv::Vec4f> & segments, int width, int height,
    int max_vp_pairs);

  DetectorParams params_;
};

}  // namespace hough_features
