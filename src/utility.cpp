#include <utility.h>
#include "args.h"

namespace OCR {

    std::vector<std::string> Utility::ReadDict(const std::string &path) {
        std::ifstream in(path);
        std::string line;
        std::vector<std::string> m_vec;
        if (in) {
            while (getline(in, line)) {
                m_vec.push_back(line);
            }
        } else {
            std::cout << "no such label file: " << path << ", exit the program..."<< std::endl;
            exit(1);
        }
        return m_vec;
    }

    void Utility::VisualizeBboxes(const cv::Mat &srcimg,
                                  const std::vector<OCRPredictResult> &ocr_result,
                                  const std::string &save_path) {
        cv::Mat img_vis;
        srcimg.copyTo(img_vis);
        for (std::size_t n = 0; n < ocr_result.size(); n++) {
            cv::Point rook_points[4];
            int mid_x = 0;
            int mid_y = 0;
            for (std::size_t m = 0; m < ocr_result[n].box.size(); m++) {
                rook_points[m] = cv::Point(int(ocr_result[n].box[m][0]), int(ocr_result[n].box[m][1]));
                mid_x += int(ocr_result[n].box[m][0]);
                mid_y += int(ocr_result[n].box[m][1]);
            }
            mid_x =  mid_x / 4;
            mid_y =  mid_y / 4;

            const cv::Point *ppt[1] = {rook_points};
            int npt[] = {4};
            cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(255, 0, 0), 2, 8, 0);
//          if there is no freetype in your opencv
//            cv::putText(img_vis,ocr_result[n].text,rook_points[0],0,0.8,CV_RGB(255, 0, 0),2);

//          use freetype
            cv::Ptr<cv::freetype::FreeType2> ft2;
            ft2=cv::freetype::createFreeType2();
            ft2->loadFontData("/usr/share/fonts/truetype/arphic/uming.ttc",0);
            ft2->putText(img_vis, ocr_result[n].text, rook_points[0], 20, CV_RGB(255, 0, 0), 1, 8, false);
        }

        cv::imwrite(save_path, img_vis);
        std::cout << "The detection visualized image saved in " + save_path
                  << std::endl;
    }

    // list all files under a directory
    void Utility::GetAllFiles(const char *dir_name,
                              std::vector<std::string> &all_inputs) {
      if (NULL == dir_name) {
        std::cout << " dir_name is null ! " << std::endl;
        return;
      }
      struct stat s;
      lstat(dir_name, &s);
      if (!S_ISDIR(s.st_mode)) {
        std::cout << "dir_name is not a valid directory !" << std::endl;
        all_inputs.push_back(dir_name);
        return;
      } else {
        struct dirent *filename; // return value for readdir()
        DIR *dir;                // return value for opendir()
        dir = opendir(dir_name);
        if (NULL == dir) {
          std::cout << "Can not open dir " << dir_name << std::endl;
          return;
        }
        std::cout << "Successfully opened the dir !" << std::endl;
        while ((filename = readdir(dir)) != NULL) {
          if (strcmp(filename->d_name, ".") == 0 ||
              strcmp(filename->d_name, "..") == 0)
            continue;
          // img_dir + std::string("/") + all_inputs[0];
          all_inputs.push_back(dir_name + std::string("/") +
                               std::string(filename->d_name));
        }
      }
    }

    cv::Mat Utility::GetRotateCropImage(const cv::Mat &srcimage,
                                std::vector<std::vector<int>> box) {
      cv::Mat image;
      srcimage.copyTo(image);
      std::vector<std::vector<int>> points = box;

      int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
      int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
      int left = int(*std::min_element(x_collect, x_collect + 4));
      int right = int(*std::max_element(x_collect, x_collect + 4));
      int top = int(*std::min_element(y_collect, y_collect + 4));
      int bottom = int(*std::max_element(y_collect, y_collect + 4));

      cv::Mat img_crop;
      image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

      for (std::size_t i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
      }

      int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                    pow(points[0][1] - points[1][1], 2)));
      int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                     pow(points[0][1] - points[3][1], 2)));

      cv::Point2f pts_std[4];
      pts_std[0] = cv::Point2f(0., 0.);
      pts_std[1] = cv::Point2f(img_crop_width, 0.);
      pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
      pts_std[3] = cv::Point2f(0.f, img_crop_height);

      cv::Point2f pointsf[4];
      pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
      pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
      pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
      pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

      cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

      cv::Mat dst_img;
      cv::warpPerspective(img_crop, dst_img, M,
                          cv::Size(img_crop_width, img_crop_height),
                          cv::BORDER_REPLICATE);

      if (float(dst_img.rows) >= float(dst_img.cols) * 3) { //1.5
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
      } else {
        return dst_img;
      }
    }

    std::vector<int> Utility::argsort(const std::vector<float>& array)
    {
        const int array_len(array.size());
        std::vector<int> array_index(array_len, 0);
        for (int i = 0; i < array_len; ++i)
            array_index[i] = i;

        std::sort(array_index.begin(), array_index.end(),
            [&array](int pos1, int pos2) {return (array[pos1] < array[pos2]); });

        return array_index;
    }

    std::string Utility::basename(const std::string &filename) {
        if (filename.empty()) {
            return "";
        }

        auto len = filename.length();
        auto index = filename.find_last_of("/\\");

        if (index == std::string::npos) {
            return filename;
        }

        if (index + 1 >= len) {

            len--;
            index = filename.substr(0, len).find_last_of("/\\");

            if (len == 0) {
                return filename;
            }

            if (index == 0) {
                return filename.substr(1, len - 1);
            }

            if (index == std::string::npos) {
                return filename.substr(0, len);
            }

            return filename.substr(index + 1, len - index - 1);
        }

        return filename.substr(index + 1, len - index);
    }

    bool Utility::PathExists(const std::string &path) {
#ifdef _WIN32
        struct _stat buffer;
  return (_stat(path.c_str(), &buffer) == 0);
#else
        struct stat buffer;
        return (stat(path.c_str(), &buffer) == 0);
#endif // !_WIN32
    }

    void Utility::CreateDir(const std::string &path) {
#ifdef _WIN32
        _mkdir(path.c_str());
#else
        mkdir(path.c_str(), 0777);
#endif // !_WIN32
    }

    void Utility::print_result(const std::vector<OCRPredictResult> &ocr_result) {
        for (std::size_t i = 0; i < ocr_result.size(); i++) {
            std::cout << i << "\t";
            // det
            std::vector<std::vector<int>> boxes = ocr_result[i].box;
            if (boxes.size() > 0) {
                std::cout << "det boxes: [";
                for (std::size_t n = 0; n < boxes.size(); n++) {
                    std::cout << '[' << boxes[n][0] << ',' << boxes[n][1] << "]";
                    if (n != boxes.size() - 1) {
                        std::cout << ',';
                    }
                }
                std::cout << "] ";
            }

            // rec
            if (ocr_result[i].score != -1.0) {
//        std::cout << " rec text: " << std::endl;
//        std::cout << ocr_result[i].text << std::endl;
                std::cout << " rec text: " << ocr_result[i].text<<std::endl<< " rec score: " << ocr_result[i].score << " ";
            }

            // cls
            if (ocr_result[i].cls_label != -1) {
                std::cout << " cls label: " << ocr_result[i].cls_label
                          << " cls score: " << ocr_result[i].cls_score;
            }
            std::cout << std::endl;
        }
    }

    void Utility::get_3channels_img(cv::Mat &img){

        if(img.channels()==4){
            cv::Mat _chs[4];
            split(img, _chs);
            cv::Mat new_img(img.rows, img.cols, CV_8UC3);
            cv::Mat _new_chas[3];
            split(new_img, _new_chas);
            for(int i=0; i<3; i++){
                _new_chas[i] = 255 - _chs[3];
            }
            cv::Mat _dst;
            merge(_new_chas, 3, _dst);
            img = _dst;
        }
        if(img.channels()==1){
            cv::Mat new_img(img.rows, img.cols, CV_8UC3);
            cv::Mat _new_chas[3];
            split(new_img, _new_chas);
            for(int i=0; i<3; i++){
                _new_chas[i] = 255 - img;
            }
            cv::Mat _dst;
            merge(_new_chas, 3, _dst);
            img = _dst;
        }
    }


    void Utility::print_ocr_time(const std::vector<std::vector<double>> &ocr_times, int det_img_count, int rec_img_count){
        std::vector<std::string> det_cls_rec{"detection", "classifier", "recognize"};
        std::vector<std::string> pre_infer_post{"preprocess", "inference", "postprocess"};
        bool det_cls_rec_flag[3]={FLAGS_det, FLAGS_cls, FLAGS_rec};
        double three_stage_total_time[3]={0};
        double toatl_time = 0;
        for(std::size_t i=0; i < ocr_times.size(); ++i){
            if(!det_cls_rec_flag[i]){
                continue;
            } else{
                double tmp_time=0;
                std::cout<<" ------------------ " << det_cls_rec[i] <<" analysis ------------------"<<std::endl;
                for(int j=0; j<3; ++j){
                    three_stage_total_time[j] += ocr_times[i][j];
                    tmp_time += ocr_times[i][j];
                    std::cout<<'['<< pre_infer_post[j]<<"] : average time is "<<ocr_times[i][j]/det_img_count
                             << " ms, average fps is "<<det_img_count*1000.0/ocr_times[i][j]<<std::endl;
                }
                std::cout<<det_cls_rec[i]<<" average time is "<< tmp_time/det_img_count
                         << " ms, average fps is "<<det_img_count*1000.0/tmp_time<<std::endl;
                std::cout<<std::endl;

                // cls and rec need to be analysed alone
                if(i!=0){
                    std::cout<<" ****** "<< det_cls_rec[i] <<" analysis based on all boxes ******"<<std::endl;
                    for(int j=0; j<3; ++j){
                        std::cout<<'['<< pre_infer_post[j]<<"] : average time is "<<ocr_times[i][j]/rec_img_count
                                 << " ms, average fps is "<<rec_img_count*1000.0/ocr_times[i][j]<<std::endl;
                    }
                    std::cout<<det_cls_rec[i]<<" average time is "<< tmp_time/rec_img_count
                             << " ms, average fps is "<<rec_img_count*1000.0/tmp_time<<std::endl;
                    std::cout<<std::endl;

                }
            }
        }

        std::cout<<" ------------------ All analyse ------------------"<<std::endl;
        for(int m=0; m<3; ++m){
            std::cout<<'['<< pre_infer_post[m]<<"] : average time is "<<three_stage_total_time[m]/det_img_count
                     << " ms, average fps is "<<det_img_count*1000.0/three_stage_total_time[m]<<std::endl;
            toatl_time += three_stage_total_time[m];
        }
        std::cout<<det_img_count<<" imgs: average time is "<< toatl_time/det_img_count << " ms, average fps is "<<det_img_count*1000.0/toatl_time<<std::endl;
    }

} // namespace OCR