from datasets import load_dataset
from tqdm import tqdm

# Load the dataset
dataset = load_dataset('monology/pile-uncopyrighted', cache_dir='/research/data/zyu401/HF_cache', split="train")

target_tokens = 26_000_000_000  # 26 billion tokens

batch_size = 1000
total_tokens = 0

check = 0
with open("/research/data/anirudh/train.txt", 'w', encoding='utf-8') as file:
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]['text']

        for line in batch:
            if check == 0:
                print(line)
                check =1 
            line = list(filter(None, line.split("\n")))
            line.append("")
            file.write("\n".join(line) + "\n")

print("Target tokens achieved. Lines saved to 'train.txt'")
