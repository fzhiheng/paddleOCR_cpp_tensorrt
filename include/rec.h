# include "Convert.h"
# include "postprocess_op.h"
# include "preprocess_op.h"
# include <opencv2/opencv.hpp>
# include "utility.h"

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

class TextRec: public Convert{
public:
    explicit TextRec(const std::string &rec_onnx_dir, const std::string &rec_engine_dir,
                     const std::string &rec_onnx_input_name, const std::string &rec_onnx_output_name,
                     const std::string &rec_char_dict_path, const int &rec_batch_num,
                     const int &rec_img_h, const int &rec_img_w, const std::string & precision
                     ):Convert({1, 3, rec_img_h, rec_img_w}, {1, 3, rec_img_h, rec_img_w}, {8, 3, rec_img_h, rec_img_w*5},precision),
                     INPUT_BLOB_NAME(rec_onnx_input_name.c_str()),
                     OUTPUT_BLOB_NAME(rec_onnx_output_name.c_str()){

        this->label_path_ = rec_char_dict_path;
        this->rec_batch_num_ = rec_batch_num;
        this->rec_img_h_ = rec_img_h;
        this->rec_img_w_ = rec_img_w;
        this->label_list_ = Utility::ReadDict(this->label_path_);
        this->label_list_.insert(this->label_list_.begin(), "#");
        this->label_list_.push_back(" ");

        Model_Init(rec_engine_dir, rec_onnx_dir);
    };
    void Model_Infer(vector<cv::Mat> img_list, std::vector<std::string> &rec_texts,
                     std::vector<float> &rec_text_scores, vector<double> &times);
    ~TextRec();

private:
    //task
    std::vector<std::string> label_list_;
    string label_path_ = "../models/txt/ppocr_keys_v1.txt";

    const char *INPUT_BLOB_NAME = "x";
    const char *OUTPUT_BLOB_NAME = "softmax_5.tmp_0";

    int rec_batch_num_= 1;
    int rec_img_h_ = 48;
    int rec_img_w_ = 320;
    std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};
    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

    CrnnResizeImg resize_op_;
    Normalize normalize_op_;
    PermuteBatch permute_op_;
//    PostProcessor post_processor_;

};

}// namespace OCR