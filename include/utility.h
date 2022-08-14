#pragma once

#include <dirent.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <vector>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace OCR {
    struct OCRPredictResult {
        std::vector<std::vector<int>> box;
        std::string text;
        float score = -1.0;
        float cls_score;
        int cls_label = -1;
    };

    class Utility {
    public:
        static std::vector<std::string> ReadDict(const std::string &path);
        static void VisualizeBboxes(const cv::Mat &srcimg,
                                    const std::vector<OCRPredictResult> &ocr_result,
                                    const std::string &save_path);

        template <class ForwardIterator>
        inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
            return std::distance(first, std::max_element(first, last));
        }
        static void GetAllFiles(const char *dir_name, std::vector<std::string> &all_inputs);
        static cv::Mat GetRotateCropImage(const cv::Mat &srcimage, std::vector<std::vector<int>> box);
        static std::vector<int> argsort(const std::vector<float>& array);
        static std::string basename(const std::string &filename);
        static bool PathExists(const std::string &path);
        static void CreateDir(const std::string &path);
        static void print_result(const std::vector<OCRPredictResult> &ocr_result);

        static void get_3channels_img(cv::Mat &img);
        static void print_ocr_time(const std::vector<std::vector<double>> &, int, int);

    };

} // namespace OCR