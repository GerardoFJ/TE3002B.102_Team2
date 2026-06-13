// YOLO26 TensorRT inference node.
//
// Runs a TensorRT FP16 engine (built from best.pt -> ONNX -> trtexec) on the
// compressed camera stream and republishes an annotated compressed image.
//
// Why C++/TensorRT and not the Python ultralytics path: JetPack 4.6 only ships
// the cp36 TensorRT Python binding, but this stack runs cp38, so ultralytics
// cannot load a .engine. The TensorRT C++ API has no such constraint and is the
// fastest route on this board.
//
// The engine's I/O (discovered from the ONNX):
//   input  "images"  : (1, 3, 640, 640) float32, RGB, /255, letterboxed
//   output "output0" : (1, 300, 6) float32, end-to-end (NMS-free) head.
//                      each row = [x1, y1, x2, y2, score, class] in the 640x640
//                      letterboxed input space. Rows are score-sorted; padded
//                      rows have score ~0, so a single confidence threshold is
//                      all the post-processing we need (no NMS).

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"

namespace {

// Default class names baked from the trained model (best.pt names dict).
const std::vector<std::string> kDefaultClasses = {
    "away", "left", "right", "stop", "straight",
    "traffic_light_green", "traffic_light_red", "traffic_light_yellow",
    "workers"};

// Minimal TensorRT logger: warnings and errors only.
class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char * msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      RCLCPP_INFO(rclcpp::get_logger("yolo_trt.trt"), "%s", msg);
    }
  }
};

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err__ = (call);                                          \
    if (err__ != cudaSuccess) {                                          \
      throw std::runtime_error(std::string("CUDA error: ") +            \
                               cudaGetErrorString(err__));               \
    }                                                                     \
  } while (0)

int64_t volume(const nvinfer1::Dims & d) {
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
  return v;
}

}  // namespace

class YoloTrtNode : public rclcpp::Node {
 public:
  YoloTrtNode() : rclcpp::Node("yolo_trt") {
    // Default to the 416 engine: best speed/accuracy balance on the Nano
    // (~9 fps, correct traffic-light class). best.engine is 640 (slower, most
    // accurate); best_320.engine is faster but misclassifies the light.
    engine_path_ = declare_parameter<std::string>(
        "engine_path", "/home/puzzlebot/best_416.engine");
    const std::string in_topic = declare_parameter<std::string>(
        "input_topic", "/camera/image_rect/compressed");
    const std::string out_topic = declare_parameter<std::string>(
        "output_topic", "/yolo/image/compressed");
    conf_thr_ = static_cast<float>(declare_parameter<double>("conf", 0.25));
    jpeg_quality_ = static_cast<int>(declare_parameter<int>("jpeg_quality", 70));
    class_names_ = declare_parameter<std::vector<std::string>>(
        "class_names", kDefaultClasses);

    // Use all cores: the GPU inference is async (we block in
    // cudaStreamSynchronize), so OpenCV decode/encode/blob can use the CPU
    // fully during the wait.
    cv::setNumThreads(4);

    loadEngine();
    allocateBuffers();

    pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        out_topic, rclcpp::SensorDataQoS().keep_last(1));
    sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        in_topic, rclcpp::SensorDataQoS().keep_last(1),
        std::bind(&YoloTrtNode::onImage, this, std::placeholders::_1));

    stats_timer_ = create_wall_timer(
        std::chrono::seconds(5), std::bind(&YoloTrtNode::logStats, this));

    RCLCPP_INFO(get_logger(),
                "yolo_trt started: engine=%s in=%s out=%s %dx%d conf=%.2f "
                "classes=%zu",
                engine_path_.c_str(), in_topic.c_str(), out_topic.c_str(),
                in_w_, in_h_, conf_thr_, class_names_.size());
  }

  ~YoloTrtNode() override {
    if (stream_) cudaStreamDestroy(stream_);
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    // TRT objects: delete is valid for TensorRT >= 8.0.
    delete context_;
    delete engine_;
    delete runtime_;
  }

 private:
  void loadEngine() {
    std::ifstream f(engine_path_, std::ios::binary);
    if (!f.good()) {
      throw std::runtime_error("cannot open engine file: " + engine_path_);
    }
    f.seekg(0, std::ios::end);
    const size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    f.read(blob.data(), size);

    runtime_ = nvinfer1::createInferRuntime(logger_);
    engine_ = runtime_->deserializeCudaEngine(blob.data(), size);
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("createExecutionContext failed");

    // Resolve binding indices and shapes (explicit-batch, static shapes).
    const int nb = engine_->getNbBindings();
    for (int i = 0; i < nb; ++i) {
      const nvinfer1::Dims d = engine_->getBindingDimensions(i);
      if (engine_->bindingIsInput(i)) {
        in_idx_ = i;
        // Expect (1, 3, H, W).
        in_h_ = d.d[d.nbDims - 2];
        in_w_ = d.d[d.nbDims - 1];
      } else {
        out_idx_ = i;
        // Expect (1, N, 6).
        out_rows_ = d.d[d.nbDims - 2];
        out_cols_ = d.d[d.nbDims - 1];
      }
      RCLCPP_INFO(get_logger(), "binding[%d] %s input=%d dims=%s", i,
                  engine_->getBindingName(i), engine_->bindingIsInput(i),
                  dimsStr(d).c_str());
    }
    if (in_idx_ < 0 || out_idx_ < 0) {
      throw std::runtime_error("could not resolve input/output bindings");
    }
    if (out_cols_ != 6) {
      RCLCPP_WARN(get_logger(),
                  "output has %d cols (expected 6 = x1,y1,x2,y2,score,class)",
                  out_cols_);
    }
    in_count_ = volume(engine_->getBindingDimensions(in_idx_));
    out_count_ = volume(engine_->getBindingDimensions(out_idx_));
  }

  void allocateBuffers() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaMalloc(&d_input_, in_count_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_, out_count_ * sizeof(float)));
    h_input_.resize(in_count_);
    h_output_.resize(out_count_);
    bindings_.resize(engine_->getNbBindings(), nullptr);
    bindings_[in_idx_] = d_input_;
    bindings_[out_idx_] = d_output_;
  }

  // Letterbox the BGR image into a square in_w_ x in_h_, recording the scale
  // and padding so detections can be mapped back to original coordinates.
  void preprocess(const cv::Mat & bgr, float & scale, int & pad_x,
                  int & pad_y) {
    const int w = bgr.cols, h = bgr.rows;
    scale = std::min(static_cast<float>(in_w_) / w,
                     static_cast<float>(in_h_) / h);
    const int nw = static_cast<int>(std::round(w * scale));
    const int nh = static_cast<int>(std::round(h * scale));
    pad_x = (in_w_ - nw) / 2;
    pad_y = (in_h_ - nh) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(in_h_, in_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, nw, nh)));

    // HWC BGR uint8 -> NCHW RGB float /255. blobFromImage is SIMD-optimized
    // and multithreaded, far faster than a per-pixel loop on the A57.
    // swapRB=true does BGR->RGB; the blob is contiguous, so copy straight in.
    cv::Mat blob = cv::dnn::blobFromImage(
        canvas, 1.0 / 255.0, cv::Size(in_w_, in_h_), cv::Scalar(),
        /*swapRB=*/true, /*crop=*/false, CV_32F);
    std::memcpy(h_input_.data(), blob.ptr<float>(),
                in_count_ * sizeof(float));
  }

  void onImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
    const auto t0 = std::chrono::steady_clock::now();

    cv::Mat bgr = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (bgr.empty()) {
      RCLCPP_WARN(get_logger(), "imdecode failed");
      return;
    }

    float scale;
    int pad_x, pad_y;
    preprocess(bgr, scale, pad_x, pad_y);

    CUDA_CHECK(cudaMemcpyAsync(d_input_, h_input_.data(),
                               in_count_ * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
      RCLCPP_WARN(get_logger(), "enqueueV2 failed");
      return;
    }
    CUDA_CHECK(cudaMemcpyAsync(h_output_.data(), d_output_,
                               out_count_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    drawDetections(bgr, scale, pad_x, pad_y);

    std::vector<uchar> enc;
    cv::imencode(".jpg", bgr, enc,
                 {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_});
    auto out = std::make_unique<sensor_msgs::msg::CompressedImage>();
    out->header = msg->header;
    out->format = "jpeg";
    out->data = std::move(enc);
    pub_->publish(std::move(out));

    const auto t1 = std::chrono::steady_clock::now();
    sum_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ++frames_;
  }

  void drawDetections(cv::Mat & img, float scale, int pad_x, int pad_y) {
    const float * out = h_output_.data();
    for (int i = 0; i < out_rows_; ++i) {
      const float * d = out + i * out_cols_;
      const float score = d[4];
      if (score < conf_thr_) continue;  // rows are score-sorted, but keep simple
      // Map letterbox coords back to the original image.
      const float x1 = (d[0] - pad_x) / scale;
      const float y1 = (d[1] - pad_y) / scale;
      const float x2 = (d[2] - pad_x) / scale;
      const float y2 = (d[3] - pad_y) / scale;
      const int cls = static_cast<int>(d[5]);

      const cv::Point p1(std::max(0, static_cast<int>(std::round(x1))),
                         std::max(0, static_cast<int>(std::round(y1))));
      const cv::Point p2(std::min(img.cols - 1, static_cast<int>(std::round(x2))),
                         std::min(img.rows - 1, static_cast<int>(std::round(y2))));

      const cv::Scalar color = colorFor(cls);
      cv::rectangle(img, p1, p2, color, 2);
      std::string label = (cls >= 0 && cls < static_cast<int>(class_names_.size()))
                              ? class_names_[cls]
                              : ("id" + std::to_string(cls));
      char buf[16];
      std::snprintf(buf, sizeof(buf), " %.2f", score);
      label += buf;
      int base = 0;
      const cv::Size ts =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
      const int ly = std::max(p1.y, ts.height + 4);
      cv::rectangle(img, cv::Point(p1.x, ly - ts.height - 4),
                    cv::Point(p1.x + ts.width + 2, ly), color, cv::FILLED);
      cv::putText(img, label, cv::Point(p1.x + 1, ly - 3),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
                  cv::LINE_AA);
    }
  }

  static cv::Scalar colorFor(int cls) {
    // Deterministic distinct-ish colors.
    const int h = (cls * 47) % 180;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, 200, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
  }

  void logStats() {
    if (frames_ == 0) {
      RCLCPP_INFO(get_logger(), "no frames in last 5s (camera up?)");
      return;
    }
    RCLCPP_INFO(get_logger(), "yolo_trt: %.1f fps, %.1f ms/frame (%d frames)",
                frames_ / 5.0, sum_ms_ / frames_, frames_);
    frames_ = 0;
    sum_ms_ = 0.0;
  }

  static std::string dimsStr(const nvinfer1::Dims & d) {
    std::string s = "(";
    for (int i = 0; i < d.nbDims; ++i) {
      s += std::to_string(d.d[i]);
      if (i + 1 < d.nbDims) s += ",";
    }
    return s + ")";
  }

  // Params
  std::string engine_path_;
  float conf_thr_ = 0.25f;
  int jpeg_quality_ = 70;
  std::vector<std::string> class_names_;

  // TensorRT
  TrtLogger logger_;
  nvinfer1::IRuntime * runtime_ = nullptr;
  nvinfer1::ICudaEngine * engine_ = nullptr;
  nvinfer1::IExecutionContext * context_ = nullptr;
  int in_idx_ = -1, out_idx_ = -1;
  int in_w_ = 640, in_h_ = 640;
  int out_rows_ = 300, out_cols_ = 6;
  int64_t in_count_ = 0, out_count_ = 0;

  // CUDA buffers
  cudaStream_t stream_ = nullptr;
  void * d_input_ = nullptr;
  void * d_output_ = nullptr;
  std::vector<float> h_input_, h_output_;
  std::vector<void *> bindings_;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr stats_timer_;

  // Stats
  int frames_ = 0;
  double sum_ms_ = 0.0;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<YoloTrtNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("yolo_trt"), "fatal: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
