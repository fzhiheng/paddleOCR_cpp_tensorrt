# include "det.h"
# include "rec.h"
# include "cls.h"
# include "args.h"

namespace OCR {

    class ocr {
    public:
        ocr();

        std::vector<OCRPredictResult> run(cv::Mat &inputImg, bool cls, bool rec, vector<vector<double>> &ocr_times);

        ~ocr();

    private:

        TextDetect *td = nullptr;
        TextClassifier *tc = nullptr;
        TextRec *tr = nullptr;

        void det(cv::Mat img, std::vector<OCRPredictResult> &ocr_results,
                 std::vector<double> &times);

        void rec(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results,
                 std::vector<double> &times);

        void cls(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results,
                 std::vector<double> &times);

    };
}