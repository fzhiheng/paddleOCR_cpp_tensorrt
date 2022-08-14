# include "rec.h"

namespace OCR{

void TextRec::Model_Infer(vector<cv::Mat> img_list, std::vector<std::string> &rec_texts,
                          std::vector<float> &rec_text_scores, vector<double> &times){

    std::chrono::duration<float> preprocess_diff =
            std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
    std::chrono::duration<float> inference_diff =
            std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
    std::chrono::duration<float> postprocess_diff =
            std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

    int img_num = img_list.size();
    std::vector<float> width_list; //存储所有待识别图像的宽高比
    for (int i = 0; i < img_num; i++){
        width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
    }
    
    std::vector<int> indices = Utility::argsort(width_list);//对宽高比由小到大进行排序，并获取indices

    int rec_batch_num = this->rec_batch_num_;
    if(img_num>0){
        if (this->rec_batch_num_ > get_engine_max_batch()){
            rec_batch_num = get_engine_max_batch();
            std::cerr<<"Your rec_batch_num is: " <<this->rec_batch_num_ <<
            " greater than MAX_DIMS_[0], and is reset to: "<<get_engine_max_batch()<<" !"<<std::endl;
        }
    }
    for(int beg_img_no = 0; beg_img_no < img_num; beg_img_no += rec_batch_num){
        /////////////////////////// preprocess ///////////////////////////////
        auto preprocess_start = std::chrono::steady_clock::now();
        int end_img_no = min(img_num, beg_img_no + rec_batch_num);
        int batch_num = end_img_no - beg_img_no;
        int imgH = this->rec_image_shape_[1];
        int imgW = this->rec_image_shape_[2];
        float max_wh_ratio = imgW * 1.0 / imgH;

    //    I do not think this step bellow is necessary, because we want to get max_wh_ratio,
    //    indices is just the index from smallest to largest according to wh_ratio
    //    max_wh_ratio = max(max_wh_ratio, width_list[indices[end_img_no-1]]); // maybe it will be faster
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            int h = img_list[indices[ino]].rows;
            int w = img_list[indices[ino]].cols;
            float wh_ratio = w * 1.0 / h;
            max_wh_ratio = max(max_wh_ratio, wh_ratio);
        }

        // 将img按照从小到大的宽高比依次处理并放入norm_img_batch中
        // 处理方法为resize到高为rec_img_h，宽为rec_img_h*max_wh_ratio
        // 并做归一化。
        int batch_width = imgW;
        std::vector<cv::Mat> norm_img_batch;
        for (int ino = beg_img_no; ino < end_img_no; ino ++) {
            cv::Mat srcimg;
            img_list[indices[ino]].copyTo(srcimg);
            cv::Mat resize_img;
            this->resize_op_.Run(srcimg, resize_img, max_wh_ratio, this->rec_image_shape_);
            this->normalize_op_.Run(&resize_img, this->mean_, this->scale_, true);
            norm_img_batch.push_back(resize_img);
            batch_width = max(resize_img.cols, batch_width);
        }

        auto preprocess_end = std::chrono::steady_clock::now();
        preprocess_diff += preprocess_end - preprocess_start;
      
        ////////////////////////// inference /////////////////////////
        void* buffers[2];

        // 为buffer[0]指针（输入）定义空间大小
        int data_size = batch_num * 3 * imgH * batch_width;
        float *inBlob = new float[data_size];
        this->permute_op_.Run(norm_img_batch, inBlob);

        int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        CHECK(cudaMalloc(&buffers[inputIndex], data_size * sizeof(float)));

        auto inference_start = std::chrono::steady_clock::now();
        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        // 将数据放到gpu上
        CHECK(cudaMemcpyAsync(buffers[inputIndex], inBlob, data_size * sizeof(float), cudaMemcpyHostToDevice, stream));

        //#### 将输入图像的大小写入context中 #######
        context->setOptimizationProfile(0); // 让convert.h创建engine的动态输入配置生效
        auto in_dims = context->getBindingDimensions(inputIndex); //获取带有可变维度的输入维度信息
        in_dims.d[0]=batch_num;
        in_dims.d[1]=3;
        in_dims.d[2]=imgH;
        in_dims.d[3]=batch_width;
        
        context->setBindingDimensions(inputIndex, in_dims); // 根据输入图像大小更新输入维度

        // 为buffer[1]指针（输出）定义空间大小
        int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        auto out_dims = context->getBindingDimensions(outputIndex);

        int output_size=1;
        for(int j=0; j<out_dims.nbDims; j++) 
            output_size *= out_dims.d[j];

        float *outBlob = new float[output_size];
        CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

        // 做推理
        context->enqueue(1, buffers, stream, nullptr);
        // 从gpu取数据到cpu上
        CHECK(cudaMemcpyAsync(outBlob, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));

        auto inference_end = std::chrono::steady_clock::now();
        inference_diff += inference_end - inference_start;
      
        ////////////////////// postprocess ///////////////////////////
        auto postprocess_start = std::chrono::steady_clock::now();

        vector<int> predict_shape;
        for(int j=0; j<out_dims.nbDims; j++) 
            predict_shape.push_back(out_dims.d[j]);
        
        for (int m = 0; m < predict_shape[0]; m++) { // m = batch_size
//            pair<vector<string>, double> temp_box_res;
            std::string str_res;
            int argmax_idx;
            int last_index = 0;
            float score = 0.f;
            int count = 0;
            float max_value = 0.0f;

            for (int n = 0; n < predict_shape[1]; n++) { // n = 2*l + 1
                argmax_idx =
                    int(Utility::argmax(&outBlob[(m * predict_shape[1] + n) * predict_shape[2]],
                                        &outBlob[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                max_value =
                    float(*std::max_element(&outBlob[(m * predict_shape[1] + n) * predict_shape[2]],
                                            &outBlob[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                    score += max_value;
                    count += 1;
                    str_res += this->label_list_[argmax_idx];
                }
                last_index = argmax_idx;
            }
            score /= count;
            if (isnan(score)){
                continue;
            }
            rec_texts[indices[beg_img_no + m]] = str_res;
            rec_text_scores[indices[beg_img_no + m]] = score;
        }

        delete [] inBlob;
        delete [] outBlob;
        auto postprocess_end = std::chrono::steady_clock::now();
        postprocess_diff += postprocess_end - postprocess_start;
    }

    times[0] = std::max(double(preprocess_diff.count() * 1000),0.0);
    times[1] = std::max(double(inference_diff.count() * 1000),0.0);
    times[2] = std::max(double(postprocess_diff.count() * 1000),0.0);

}
TextRec::~TextRec(){

}

} // namespace OCR