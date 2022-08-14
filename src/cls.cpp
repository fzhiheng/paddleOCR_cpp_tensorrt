//
// Created by fzh on 2022/8/11.
//
# include "cls.h"

namespace OCR {

    void TextClassifier::Model_Infer(std::vector<cv::Mat> img_list,
                                 std::vector<int> &cls_labels,
                                 std::vector<float> &cls_scores,
                                 vector<double> &times){


        std::chrono::duration<float> preprocess_diff =
                std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
        std::chrono::duration<float> inference_diff =
                std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
        std::chrono::duration<float> postprocess_diff =
                std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

        // --------------------- preprocess ---------------------
        int img_num = img_list.size();
        std::vector<int> cls_image_shape = {3, 48, 192};
        for (int beg_img_no = 0; beg_img_no < img_num;
             beg_img_no += this->cls_batch_num_) {
            auto preprocess_start = std::chrono::steady_clock::now();
            int end_img_no = min(img_num, beg_img_no + this->cls_batch_num_);
            int batch_num = end_img_no - beg_img_no;
            // preprocess
            std::vector<cv::Mat> norm_img_batch;
            for (int ino = beg_img_no; ino < end_img_no; ino++) {
                cv::Mat srcimg;
                img_list[ino].copyTo(srcimg);
                cv::Mat resize_img;
                this->resize_op_.Run(srcimg, resize_img, cls_image_shape);
                this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                                        this->is_scale_);
                norm_img_batch.push_back(resize_img);
            }
            int data_size = batch_num * cls_image_shape[0] * cls_image_shape[1] * cls_image_shape[2];
            float *inBlob = new float[data_size];
            this->permute_op_.Run(norm_img_batch, inBlob);
            auto preprocess_end = std::chrono::steady_clock::now();
            preprocess_diff += preprocess_end - preprocess_start;

            // --------------------- inference ---------------------
            void * buffers[2];
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
            in_dims.d[2]=cls_image_shape[1];
            in_dims.d[3]=cls_image_shape[2];

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

            // ---------------------postprocess ---------------------
            auto postprocess_start = std::chrono::steady_clock::now();
            vector<int> predict_shape;
            for(int j=0; j<out_dims.nbDims; j++)
                predict_shape.push_back(out_dims.d[j]);
            for (int batch_idx = 0; batch_idx < predict_shape[0]; batch_idx++) {
                int label = int(
                        Utility::argmax(&outBlob[batch_idx * predict_shape[1]],
                                        &outBlob[(batch_idx + 1) * predict_shape[1]]));
                float score = float(*std::max_element(
                        &outBlob[batch_idx * predict_shape[1]],
                        &outBlob[(batch_idx + 1) * predict_shape[1]]));
                cls_labels[beg_img_no + batch_idx] = label;
                cls_scores[beg_img_no + batch_idx] = score;
            }

            delete[] inBlob;
            delete[] outBlob;

            auto postprocess_end = std::chrono::steady_clock::now();
            postprocess_diff += postprocess_end - postprocess_start;
        }
        times[0] = double(preprocess_diff.count() * 1000);
        times[1] = double(inference_diff.count() * 1000);
        times[2] = double(postprocess_diff.count() * 1000);
    }
    TextClassifier::~TextClassifier(){
    }

}

