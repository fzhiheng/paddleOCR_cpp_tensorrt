# include "ocr.h"
# include "args.h"
# include "utility.h"
# include <stdio.h>

namespace OCR {

    ocr::ocr() {
        if (FLAGS_det) {
            this->td = new TextDetect(
                    FLAGS_det_onnx_dir, FLAGS_det_engine_dir,
                    FLAGS_det_onnx_input_name, FLAGS_det_onnx_output_name,
                    FLAGS_max_side_len, FLAGS_det_db_thresh,
                    FLAGS_det_db_box_thresh, FLAGS_det_db_unclip_ratio,
                    FLAGS_use_dilation, FLAGS_use_polygon_score,
                    FLAGS_build_precision);
        }
        if (FLAGS_cls && FLAGS_use_angle_cls) {
            this->tc = new TextClassifier(
                    FLAGS_cls_onnx_dir, FLAGS_cls_engine_dir,
                    FLAGS_cls_onnx_input_name, FLAGS_cls_onnx_output_name,
                    FLAGS_cls_batch_num, FLAGS_cls_thresh,
                    FLAGS_build_precision);
        }
        if (FLAGS_rec) {
            this->tr = new TextRec(
                    FLAGS_rec_onnx_dir, FLAGS_rec_engine_dir,
                    FLAGS_rec_onnx_input_name, FLAGS_rec_onnx_output_name,
                    FLAGS_rec_char_dict_path, FLAGS_rec_batch_num,
                    FLAGS_rec_img_h, FLAGS_rec_img_w,
                    FLAGS_build_precision);
        }
    }

    void ocr::det(cv::Mat inputImg, std::vector<OCRPredictResult> &ocr_results,
                  std::vector<double> &det_times) {
        if (inputImg.channels() != 3) {
            Utility::get_3channels_img(inputImg);
        }
        std::vector<std::vector<std::vector<int>>> boxes;
        this->td->Model_Infer(inputImg, boxes, det_times);
        for (std::size_t i = 0; i < boxes.size(); i++) {
            OCRPredictResult res;
            res.box = boxes[i];
            ocr_results.push_back(res);
        }
    }


    void ocr::rec(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results,
                  std::vector<double> &rec_times) {
        std::vector<std::string> rec_texts(img_list.size(), "");
        std::vector<float> rec_text_scores(img_list.size(), 0);
        this->tr->Model_Infer(img_list, rec_texts, rec_text_scores, rec_times);
        for (std::size_t i = 0; i < rec_texts.size(); i++) {
            ocr_results[i].text = rec_texts[i];
            ocr_results[i].score = rec_text_scores[i];
        }
    }


    void ocr::cls(std::vector<cv::Mat> img_list,
                  std::vector<OCRPredictResult> &ocr_results,
                  std::vector<double> &cls_times) {
        std::vector<int> cls_labels(img_list.size(), 0);
        std::vector<float> cls_scores(img_list.size(), 0);
        this->tc->Model_Infer(img_list, cls_labels, cls_scores, cls_times);
        for (std::size_t i = 0; i < cls_labels.size(); i++) {
            ocr_results[i].cls_label = cls_labels[i];
            ocr_results[i].cls_score = cls_scores[i];
        }
    }

    std::vector<OCRPredictResult>
    ocr::run(cv::Mat &inputImg, bool cls, bool rec, vector<vector<double>> &ocr_times) {

        std::vector<OCRPredictResult> ocr_result;
        // time[0]:preprocess time[1]:infer time[2]:postprocess
        std::vector<double> time_info_det = {0, 0, 0};
        std::vector<double> time_info_cls = {0, 0, 0};
        std::vector<double> time_info_rec = {0, 0, 0};
        std::vector<std::vector<double>> time_info{{0,0,0},{0,0,0},{0,0,0}};
        // ---------------------- det ----------------------
        this->det(inputImg, ocr_result, time_info[0]);

        // crop image
        std::vector<cv::Mat> img_list;
        for (std::size_t j = 0; j < ocr_result.size(); j++) {
            cv::Mat crop_img;
            crop_img = Utility::GetRotateCropImage(inputImg, ocr_result[j].box);
            img_list.push_back(crop_img);
//            cv::namedWindow("tmp", cv::WINDOW_NORMAL);
//            cv::imshow("tmp", crop_img);
//            cv::waitKey(0);
        }

        // ---------------------- cls ----------------------
        if (cls && this->tc != nullptr) {
            this->cls(img_list, ocr_result, time_info[1]);
            for (std::size_t i = 0; i < img_list.size(); i++) {
                if (ocr_result[i].cls_label % 2 == 1 &&
                    ocr_result[i].cls_score > this->tc->cls_thresh_) {
                    cv::rotate(img_list[i], img_list[i], 1); // 0: rotate 90 degree, 1 rotate 180 degree, 2 rotate 270 degree
                }
            }
        }
        // ---------------------- rec ----------------------
        if (rec) {
            this->rec(img_list, ocr_result, time_info[2]);
        }
        for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
                ocr_times[i][j] += time_info[i][j];
            }
        }

        return ocr_result;
    }



    ocr::~ocr() {
        if (this->td != nullptr) {
            delete this->td;
        }
        if (this->tr != nullptr) {
            delete this->tr;
        }
        if (this->tc != nullptr) {
            delete this->tc;
        }
    }

}