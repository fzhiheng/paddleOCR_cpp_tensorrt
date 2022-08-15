#include <gflags/gflags.h>

// common args
DEFINE_string(build_precision, "fp16", "Precision be one of fp32/fp16, it works only in building engine.");
DEFINE_string(output, "./output/", "Save benchmark log path.");
DEFINE_string(save_engine_dir, "../myEngines/", "Dir of output engine.");
//DEFINE_string(image_dir, "/home/fzh/AVP/dataset/OCR/slot_seg_mask_white2_512_90", "Dir of input image.");
DEFINE_string(image_dir, "../testImgs/11.jpg", "Dir of input image.");


// detection related
DEFINE_string(det_onnx_dir, "../myModels/det.onnx","Path of det onnx model.");
DEFINE_string(det_engine_dir, "../myEngines/det_fp162.engine","Path of det engine model.");
DEFINE_int32(max_side_len, 640, "max_side_len of input image.");
DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.");
DEFINE_double(det_db_box_thresh, 0.6, "Threshold of det_db_box_thresh.");
DEFINE_double(det_db_unclip_ratio, 1.5, "Threshold of det_db_unclip_ratio.");
DEFINE_bool(use_dilation, false, "Whether use the dilation on output map.");
DEFINE_bool(use_polygon_score, false, "Whether use the polygon or not.");
DEFINE_bool(visualize, true, "Whether show the detection results.");

// classification related
DEFINE_string(cls_onnx_dir, "", "Path of classifier onnx model.");
DEFINE_string(cls_engine_dir, "", "Path of classifier engine model.");
DEFINE_bool(use_angle_cls, false, "Whether use angle cls.");
DEFINE_double(cls_thresh, 0.9, "Threshold of cls_thresh.");
DEFINE_int32(cls_batch_num, 1, "cls_batch_num.");

// recognition related
DEFINE_string(rec_onnx_dir, "../myModels/ch_rec_v3.onnx","Path of recognition onnx model.");
DEFINE_string(rec_engine_dir, "../myEngines/ch_rec_v3_fp16.engine","Path of recognition engine model.");
DEFINE_string(rec_char_dict_path, "../myModels/dict_txt/ppocr_keys_v1.txt", "Path of dictionary.");
DEFINE_int32(rec_batch_num, 1, "rec_batch_num.");
DEFINE_int32(rec_img_h, 48, "rec image height");
DEFINE_int32(rec_img_w, 320, "rec image width");

// ocr forward related
DEFINE_bool(det, true, "Whether use det in forward.");
DEFINE_bool(rec, true, "Whether use rec in forward.");
DEFINE_bool(cls, false, "Whether use cls in forward.");