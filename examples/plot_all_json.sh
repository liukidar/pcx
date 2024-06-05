#!/bin/bash

# Directory containing the JSON files
json_dir="/users-2/amine/pcax24/examples/data/results_pc/two_layer_nn/CrossEntropyLoss" # uncomment for pc
#json_dir="/users-2/amine/pcax24/examples/data/results_bp/two_layer_nn/CrossEntropyLoss" # uncomment for bp

# Directory to save the plots
output_dir="/users-2/amine/pcax24/examples/plots_pc" # uncomment for pc
#output_dir="/users-2/amine/pcax24/examples/plots_bp" # uncomment for bp

# Plotter script
plotter_script="/users-2/amine/pcax24/examples/plotters/two_layer_plotter.py"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Loop over each JSON file in the specified directory
for json_file in "$json_dir"/*.json; do
  # Extract the filename without the extension
  filename=$(basename "$json_file" .json)
  # Construct the output file path
  output_file="$output_dir/$filename"
  # Print the current file being processed
  echo "Processing $json_file..."
  # Run the plotter script
  python "$plotter_script" "$json_file" --saveto "$output_file" --bg white --fg black --transparent
done

echo "Plotting complete. Plots saved to $output_dir."
