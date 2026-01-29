# Prediction Test Examples (User Provided)

This folder is for your own test images.

## Instructions
1. Find 5 images corresponding to the types below.
2. Upload them to this folder (`/workspace/MH6812_Project/test_examples/`).
3. **Important**: Rename your images to include the **Keyword** so the system knows which instruction to apply.

## Style Keywords & Instructions (Updated)

| ID | Image Type | Keyword to use in Filename | Instruction that will be applied |
| :--- | :--- | :--- | :--- |
| **01** | Underexposed/Dark | **`dark`** | "Brighten the image significantly and add a subtle cooling filter to correct the color tone, making it look fresh and natural." |
| **01b**| Forest Dark | **`forest dark`** | "Change the time of day from night to bright noon. Maximize exposure to make the dark forest extremely bright and clear." |
| **02** | Flat/Low Contrast | **`flat`** | "Increase global contrast and saturation to make the image look vivid and three-dimensional, removing the gray haze." |
| **03** | Flat Variant | **`flat_1`** | "Apply a vibrant high-contrast look, enhancing colors and details while keeping the image bright and sharp." |
| **04** | Cold Skin Tone/Snow | **`cold`** | "Warm up the image to give it a golden hour glow, increasing brightness to make it look sunny and inviting." |
| **05** | Overexposed/Too Bright | **`bright`** | "Slightly reduce the overexposed highlights to recover sky details, but maintain the overall bright and airy atmosphere." |
| **06** | Sunset Vibe | **`sunset`** | "Enhance the sunset colors by boosting orange and red tones, increasing contrast for a dramatic and breathtaking view." |

### Naming Examples
- `01_forest_dark.jpg`
- `02_mountains_flat.png`
- `03_portrait_cold.jpg`
- `04_city_bright.jpg`
- `05_girl_sunset.jpg`

## How to Run
After uploading and renaming your files, run:
```bash
python3 ../generate_test_examples.py
```
