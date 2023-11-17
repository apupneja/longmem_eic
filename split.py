def split_large_file(input_file, output_prefix, num_files):
    with open(input_file, 'r') as file:
        total_lines = sum(1 for line in file)

    lines_per_file = total_lines // num_files

    with open(input_file, 'r') as file:
        for i in range(num_files):
            print(i)
            output_file = f"{output_prefix}_{i + 1}.txt"

            with open(output_file, 'w') as out_file:
                for _ in range(lines_per_file):
                    line = file.readline()
                    if not line:
                        break
                    out_file.write(line)

if __name__ == "__main__":
    input_file_path = "/research/data/anirudh/re/train.txt"
    output_prefix = "/research/data/anirudh/re/train"
    num_files = 4

    split_large_file(input_file_path, output_prefix, num_files)
