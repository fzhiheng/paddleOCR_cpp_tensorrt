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

    class TextClassifier: public Convert{

    public:
        explicit TextClassifier(const std::string &cls_onnx_dir, const std::string &cls_engine_dir,
                                const std::string &save_engine_dir,
                                const int &cls_batch_num, const double &cls_thresh,
                                const std::string & precision):Convert({1,3,48,192},{1,3,48,192},{8,3,48,192},precision){

            this->cls_batch_num_ = cls_batch_num;
            this->cls_thresh_ = cls_thresh;

            Model_Init(cls_engine_dir, cls_onnx_dir, save_engine_dir);
        };

        void Model_Infer(std::vector<cv::Mat> img_list,
                         std::vector<int> &cls_labels,
                         std::vector<float> &cls_scores,
                         vector<double> &times);

        double cls_thresh_ = 0.9;
        ~TextClassifier();

    private:

        std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
        std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
        bool is_scale_ = true;
        int cls_batch_num_ = 1;

        ClsResizeImg resize_op_;
        Normalize normalize_op_;
        PermuteBatch permute_op_;

    };

}// namespace OCR

