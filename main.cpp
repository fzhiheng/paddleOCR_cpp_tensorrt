# include "ocr.h"
# include "args.h"

# include <dirent.h>
# include <sys/stat.h>
# include <sys/types.h>

using namespace OCR;

int main(int argc, char **argv){
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (!Utility::PathExists(FLAGS_image_dir)) {
        std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir
                  << endl;
        exit(1);
    }
    ocr * m_ocr = new ocr();
    std::vector<String> all_img_names;
    cv::glob(FLAGS_image_dir, all_img_names);
    cout << "all process img is:" << all_img_names.size()<<endl;
    int det_box_count = 0;

    std::vector<std::vector<double>> ocr_times{{0,0,0},{0,0,0},{0,0,0}};
    for(std::size_t i=0; i<all_img_names.size(); i++){

        cv::Mat test_img = cv::imread(all_img_names[i], -1);
        cout <<"process:"<<all_img_names[i] << endl;
        std::vector<OCRPredictResult> ocr_results = m_ocr->run(test_img, FLAGS_cls, FLAGS_rec, ocr_times);
        det_box_count += ocr_results.size();
        Utility::print_result(ocr_results);

        if (FLAGS_visualize && FLAGS_det) {
            cv::Mat srcimg = cv::imread(all_img_names[i],-1);
            if (!srcimg.data) {
                std::cerr << "[ERROR] image read failed! image path: "
                          << all_img_names[i] << endl;
                exit(1);
            }
            std::string file_name = Utility::basename(all_img_names[i]);
            std::string file_path = FLAGS_output;

            if (!Utility::PathExists(file_path)) {
                Utility::CreateDir(file_path);
            }
            Utility::VisualizeBboxes(srcimg, ocr_results, file_path + "/" + to_string(i) + file_name);
        }
        cout << "***************************" << endl;
    }

    Utility::print_ocr_time(ocr_times, all_img_names.size(), det_box_count);

    delete m_ocr;
    return 0;
}