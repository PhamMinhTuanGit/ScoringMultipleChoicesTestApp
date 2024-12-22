import os
import image_processing
import grid_info
import visualization

# Directories
input_dir = 'D:\\4_2_AREAS\\testset1\\testset1\\images\\'
output_dir = 'D:\\4_2_AREAS\\testset1\\testset1\\centers\\'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Process image files only
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, f"processed_{filename}")

        try:
            # Process the image
            centers = image_processing.getNails(input_image_path)
            gridmatrix = grid_info.getGridmatrix(centers)
            flaten_matrix = grid_info.flattenMatrix(gridmatrix)

            # Visualize and save the output
            visualization.drawDots(input_image_path, flaten_matrix, output_path=output_image_path)
            print(f"Processed and saved: {output_image_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Processing complete.")
