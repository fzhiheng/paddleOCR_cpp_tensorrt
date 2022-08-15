# include "Convert.h"
# include "postprocess_op.h"
# include "preprocess_op.h"
# include <opencv2/opencv.hpp>

using namespace cv;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

namespace OCR {

class TextDetect: public Convert{
    public:
        explicit TextDetect(const std::string &det_onnx_dir, const std::string &det_engine_dir,
                            const std::string &save_engine_dir,
                            const int &max_side_len, const double &det_db_thresh,
                            const double &det_db_box_thresh, const double &det_db_unclip_ratio,
                            const bool &use_dilation, const bool &use_polygon_score,
                            const std::string & precision):Convert(precision){

            this->max_side_len_ = max_side_len;
            this->det_db_thresh_ = det_db_thresh;
            this->det_db_box_thresh_ = det_db_box_thresh;
            this->det_db_unclip_ratio_ = det_db_unclip_ratio;
            this->use_dilation_ = use_dilation;
            this->use_polygon_score_ = use_polygon_score;

            Model_Init(det_engine_dir, det_onnx_dir, save_engine_dir);
        };
        void Model_Infer(cv::Mat& Input_Image, vector<vector<vector<int>>> &boxes, vector<double> &times);
        ~TextDetect();

    private:
        //config
        double det_db_thresh_ = 0.3;
        double det_db_box_thresh_ = 0.5;
        double det_db_unclip_ratio_ = 2.0;
        bool use_dilation_ = false;
        bool use_polygon_score_ = false; // if use_polygon_score_ is true, it will be slow

        // input image
        int max_side_len_ = 640;

        vector<float> mean_ = {0.485f, 0.456f, 0.406f};
        vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
        ResizeImgType0 resize_op_;
        Normalize normalize_op_;
        Permute permute_op_;
        PostProcessor post_processor_;

    };

}// namespace OCR

