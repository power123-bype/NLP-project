
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
        "forest dark": "Change the time of day from night to bright noon. Maximize exposure to make the dark forest extremely bright and clear.",
        "dark":   "Brighten the image significantly and add a subtle cooling filter to correct the color tone, making it look fresh and natural.",
        "flat":   "Increase global contrast and saturation to make the image look vivid and three-dimensional, removing the gray haze.",
        "flat_1": "Apply a vibrant high-contrast look, enhancing colors and details while keeping the image bright and sharp.",
        "cold":   "Warm up the image to give it a golden hour glow, increasing brightness to make it look sunny and inviting.",
        "bright": "Slightly reduce the overexposed highlights to recover sky details, but maintain the overall bright and airy atmosphere.",
        "sunset": "Enhance the sunset colors by boosting orange and red tones, increasing contrast for a dramatic and breathtaking view."
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
