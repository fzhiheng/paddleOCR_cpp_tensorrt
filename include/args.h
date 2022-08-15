#pragma once

#include <gflags/gflags.h>

// common args
DECLARE_string(build_precision);
DECLARE_string(output);
DECLARE_string(save_engine_dir);
DECLARE_string(image_dir);

// detection related
DECLARE_string(det_onnx_dir);
DECLARE_string(det_engine_dir);
DECLARE_int32(max_side_len);
DECLARE_double(det_db_thresh);
DECLARE_double(det_db_box_thresh);
DECLARE_double(det_db_unclip_ratio);
DECLARE_bool(use_dilation);
DECLARE_bool(use_polygon_score);
DECLARE_bool(visualize);

// classification related
DECLARE_string(cls_onnx_dir);
DECLARE_string(cls_engine_dir);
DECLARE_int32(cls_batch_num);
DECLARE_double(cls_thresh);
DECLARE_bool(use_angle_cls);


// recognition related
DECLARE_string(rec_onnx_dir);
DECLARE_string(rec_engine_dir);
DECLARE_string(rec_char_dict_path);
DECLARE_int32(rec_batch_num);
DECLARE_int32(rec_img_h);
DECLARE_int32(rec_img_w);


// forward related
DECLARE_bool(det);
DECLARE_bool(rec);
DECLARE_bool(cls);
