# cd /data/tangning/proj_6 && python eval_item_sid_distribution.py \
#   --item-user-map-path /data/tangning/proj_6/amazon_raw/item_to_user_ids.json \
#   --exp baseline=/data/tangning/proj_6/outputs/exp_baseline_no_mi \
#   --exp mi_w02=/data/tangning/proj_6/outputs/exp_mi_w02 \
#   --sid-mode full \
#   --min-users-per-item 2 \
#   --summary-csv-path /data/tangning/proj_6/outputs/sid_eval/summary_baseline_vs_mi.csv \
#   --per-item-dir /data/tangning/proj_6/outputs/sid_eval/per_item


# cd /data/tangning/proj_6 && python eval_item_sid_distribution.py \
#   --item-user-map-path /data/tangning/proj_6/amazon_raw/item_to_user_ids.json \
#   --exp baseline_cb32_allusers=/data/tangning/proj_6/outputs/exp_baseline_cb32_allusers_devcuda-0_s42 \
#   --sid-mode full \
#   --min-users-per-item 2 \
#   --summary-csv-path /data/tangning/proj_6/outputs/sid_eval/summary_baseline_cb32_allusers.csv \
#   --per-item-dir /data/tangning/proj_6/outputs/sid_eval/per_item

cd /data/tangning/proj_6 && python eval_item_sid_distribution.py \
  --item-user-map-path /data/tangning/proj_6/amazon_raw/item_to_user_ids.json \
  --exp mi_w1_a5_b01_tau002_k4=/data/tangning/proj_6/outputs/exp_mi_cb32_w1p0_a5p0_b0p1_tau0p02_k4_allusers_s42 \
  --sid-mode full \
  --min-users-per-item 2 \
  --summary-csv-path /data/tangning/proj_6/outputs/sid_eval/summary_mi_w1_a5_b01_tau002_k2_allusers.csv \
  --per-item-dir /data/tangning/proj_6/outputs/sid_eval/per_item