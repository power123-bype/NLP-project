# Prediction Test Examples (User Provided)

This folder is for your own test images.

## Instructions
1. Find 5 images corresponding to the types below.
2. Upload them to this folder (`/workspace/MH6812_Project/test_examples/`).
3. **Important**: Rename your images to include the **Keyword** so the system knows which instruction to apply.

## Style Keywords & Instructions (Updated)

| ID | Image Type | Keyword to use in Filename | Instruction that will be applied |
| :--- | :--- | :--- | :--- |
| **01** | Underexposed/Dark | **`dark`** | "Brighten the image significantly but add a strong blue cooling filter to correct the color tone and remove yellowness." |
| **01b**| Forest Dark | **`forest dark`** | "Brighten the dark forest significantly to reveal details, but keep the color temperature cool and natural, avoiding any red tint." |
| **02** | Flat/Low Contrast | **`flat`** | "Apply a strong high dynamic range effect, making shadows darker and highlights much brighter for maximum contrast." |
| **03** | Flat Variant | **`flat_1`** | "Apply a strong high dynamic range effect with vivid colors, but ensure the image remains bright and clear." |
| **04** | Cold Skin Tone/Snow | **`cold`** | "Increase the brightness significantly and make the image look sunny and warm." |
| **05** | Overexposed/Too Bright | **`bright`** | "Reduce the highlights slightly to recover details, but keep the image bright and airy, do not darken the shadows." |
| **06** | Sunset Vibe | **`sunset`** | "Make the sunset extremely intense, fiery and dramatic, boosting saturation and warmth to the maximum." |

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
