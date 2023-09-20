python run_infer.py \
--gpu='1' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=16 \
--model_mode=fast \
--model_path=/data/zrf/hover_net-master/checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/data/zrf/data/processed_jwy/G3212069-D1-V11N29-105-L220623030 \
--output_dir=/data/zrf/data/cell_segjwy/hover \
--mem_usage=0.1 \
--draw_dot \
--save_qupath