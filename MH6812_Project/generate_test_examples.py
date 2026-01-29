
import os
import glob
from inference import predict

# Directory containing user provided images
test_dir = 'test_examples'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Define the mapping from filename keywords to instructions
# If a filename contains the keyword, that instruction will be used.
style_map = {
        "forest dark": "Brighten the dark forest significantly to reveal details, but keep the color temperature cool and natural, avoiding any red tint.",
        "dark":   "Brighten the image significantly but add a strong blue cooling filter to correct the color tone and remove yellowness.",
        "flat":   "Apply a strong high dynamic range effect, making shadows darker and highlights much brighter for maximum contrast.",
        "flat_1": "Apply a strong high dynamic range effect with vivid colors, but ensure the image remains bright and clear.",
        "cold":   "Increase the brightness significantly and make the image look sunny and warm.",
        "bright": "Reduce the highlights slightly to recover details, but keep the image bright and airy, do not darken the shadows.",
        "sunset": "Make the sunset extremely intense, fiery and dramatic, boosting saturation and warmth to the maximum.",
        "road":   "Drastically increase brightness and contrast to reveal the details of the buildings and mountains currently hidden in the dark."
    }

print(f"Scanning '{test_dir}' for images...")

# Find all images
extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(test_dir, ext)))

# Filter out output files (avoid processing already processed files)
input_files = [f for f in image_files if '_after' not in f]

if not input_files:
    print(f"\n[!] No input images found in '{test_dir}'.")
    print("Please upload your images to that folder.")
    print("Recommended naming convention: 'myphoto_cyberpunk.jpg', 'landscape_sunny.png', etc.")
    print("Available keywords:", ", ".join(style_map.keys()))
else:
    print(f"Found {len(input_files)} images. Processing...\n")

for input_path in input_files:
    filename = os.path.basename(input_path)
    name_lower = filename.lower()
    
    # Determine instruction based on filename
    instruction = None
    style_name = "Default"
    
    # Sort keys by length descending to match longest keyword first (e.g. match 'flat_1' before 'flat')
    sorted_keys = sorted(style_map.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        if key in name_lower:
            instruction = style_map[key]
            style_name = key
            break
    
    # If no keyword matched, default or skip? 
    # Let's provide a default instruction if user didn't name it specifically
    if instruction is None:
        print(f"[-] Skipped '{filename}': Filename does not contain a known style keyword.")
        print(f"    Please rename it to include one of: {', '.join(style_map.keys())}")
        continue

    print(f"Processing: {filename}")
    print(f"  > Style: {style_name}")
    print(f"  > Instruction: {instruction}")
    
    # Construct output filename
    # e.g. image_cyberpunk.jpg -> image_cyberpunk_after.jpg
    file_root, _ = os.path.splitext(filename)
    output_filename = f"{file_root}_after.jpg"
    output_path = os.path.join(test_dir, output_filename)
    
    try:
        predict(input_path, instruction, output_image=output_path)
        print(f"  > Output saved to: {output_filename}\n")
    except Exception as e:
        print(f"  > Error processing {filename}: {e}\n")

print("Done.")
