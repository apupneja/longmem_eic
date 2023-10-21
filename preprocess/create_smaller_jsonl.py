import os
import json
from glob import glob

def split_jsonl_files(input_folder, output_folder, lines_per_file):
    os.makedirs(output_folder, exist_ok=True)

    jsonl_files = glob(os.path.join(input_folder, '*.jsonl'))

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as infile:
            lines = infile.readlines()

        for i in range(0, len(lines), lines_per_file):
            chunk = lines[i:i + lines_per_file]

            output_file = os.path.join(output_folder, f'{os.path.basename(jsonl_file)}_{i // lines_per_file + 1}.jsonl')

            with open(output_file, 'w') as outfile:
                outfile.writelines(chunk)

if __name__ == "__main__":
    input_folder = '/data/zyu401_data/anirudh/pile/'  # Replace with the path to your input folder
    output_folder = '/data/zyu401_data/anirudh/pile_split/'  # Replace with the path where you want to save the split files
    lines_per_file = 1000  # Specify the number of lines per output file

    split_jsonl_files(input_folder, output_folder, lines_per_file)
