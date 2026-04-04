from datasets import load_dataset

# 下载数据集
dataset = load_dataset(
    "akaifun/title_gen"   # 可改成你的磁盘路径
)

# 保存到本地（关键）
dataset.save_to_disk("/data/title_gen")

print("✅ Dataset downloaded and saved to /data/title_gen")