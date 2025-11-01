import os
from PIL import Image

# === Configuration ===
input_folder = "/Users/aryanmanchanda/Desktop/Div2K High Resolution Images/DIV2K_train_HR/DIV2K_train_HR"
output_folder = "/Users/aryanmanchanda/Desktop/LowRes_256"
target_size = (256, 256)  # desired low-res size
image_extensions = (".png", ".jpg", ".jpeg")

# === Ensure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

def crop_to_multiple(img, factor=4):
    """Crop image so that width & height are divisible by factor."""
    w, h = img.size
    new_w = w - (w % factor)
    new_h = h - (h % factor)
    return img.crop((0, 0, new_w, new_h))

# === Process all images ===
count = 0
for filename in os.listdir(input_folder):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            img = Image.open(input_path).convert("RGB")
            img = crop_to_multiple(img, 4)
            img = img.resize(target_size, Image.BICUBIC)  # downscale to 256×256
            img.save(output_path, "PNG")
            count += 1
            print(f"[OK] {filename} → saved to {output_path}")
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

print(f"\n✅ Done! {count} images converted and saved to: {output_folder}")