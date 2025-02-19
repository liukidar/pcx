# Define input and output file names
input_file = "intermediate_best.txt"
cyclic_file = "hsearch_cyclic_intermediate_best.txt"
mixed_file = "hsearch_mixed_intermediate_best.txt"

# Read the original file and sort lines into the correct files
with open(input_file, "r") as f:
    cyclic_lines = []
    mixed_lines = []

    for line in f:
        if "mixed" in line:
            mixed_lines.append(line)
        elif "cyclic" in line:
            cyclic_lines.append(line)

# Write the sorted lines to the new files
with open(cyclic_file, "w") as f:
    f.writelines(cyclic_lines)

with open(mixed_file, "w") as f:
    f.writelines(mixed_lines)

print(f"Lines containing 'cyclic' moved to: {cyclic_file}")
print(f"Lines containing 'mixed' moved to: {mixed_file}")
