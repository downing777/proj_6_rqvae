# prefix_tree_recommender

Train user semantic IDs (3-layer RQ-VAE) from raw Amazon reviews.

## What This Version Does

1. `build_amazon_data.py` reads review JSONL data (supports two formats):
   - legacy: `reviewerID/reviewText/summary/unixReviewTime/asin`
   - step4: `user_id/text/title/timestamp/(parent_asin|asin)`
2. Encodes **every review** into an embedding and saves cache.
3. Builds per-user embeddings by averaging all review embeddings of each user.
4. Builds:
   - `UserDataset`: user -> review indices + user embedding
   - `ItemDataset`: item -> user indices + user count
5. `train_user.py` trains user RQ-VAE with either:
   - `--sample-by user`
   - `--sample-by item`
6. Exports user semantic IDs and training artifacts.

## Main Scripts

- `build_amazon_data.py`
- `train_user.py`

## Default Data Path

- `/data/tangning/proj_6/dataset/step4/final_target_user_reviews_by_category`

## Quick Start

```bash
cd /data/tangning/proj_6/prefix_tree_recommender
python build_amazon_data.py --output-path outputs/amazon_user_item_dataset.pt
python train_user.py --built-data-path outputs/amazon_user_item_dataset.pt --output-dir outputs
```

## Example Commands

Build from all category files, then sample by users:

```bash
python build_amazon_data.py \
  --reviews-path /data/tangning/proj_6/dataset/step4/final_target_user_reviews_by_category \
  --output-path outputs/amazon_user_item_dataset.pt

python train_user.py \
  --built-data-path outputs/amazon_user_item_dataset.pt \
  --sample-by user \
  --train-iterations 5000 \
  --output-dir outputs
```

Build only beauty file, then sample by items (2 users sampled per item):

```bash
python build_amazon_data.py \
  --reviews-path /data/tangning/proj_6/dataset/step4/final_target_user_reviews_by_category \
  --category-glob beauty_and_personal_care \
  --output-path outputs_beauty/amazon_user_item_dataset.pt

python train_user.py \
  --built-data-path outputs_beauty/amazon_user_item_dataset.pt \
  --sample-by item \
  --users-per-item 2 \
  --train-batch-size 128 \
  --output-dir outputs_beauty
```

## Outputs

- `outputs/amazon_user_item_dataset.pt` (from build step)
- `outputs/user_review_stats.csv`
- `outputs/item_user_stats.csv`
- `outputs/user_semantic_ids.csv`
- `outputs/user_semantic_ids.jsonl`
- `outputs/user_rqvae.pt`

## Useful Flags

- `--force-recompute-embeddings`
- `--min-reviews-per-user`
- `--max-reviews-per-user`
- `--embedding-model`
- `--wandb-logging`
