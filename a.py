import os
from glob import glob
from PIL import Image
from torchvision import transforms

# === Configuration ===
hr_dir = "/Users/aryanmanchanda/Desktop/Div2K High Resolution Images/DIV2K_train_HR/DIV2K_train_HR"
lr_dir = "/Users/aryanmanchanda/Desktop/LowRes_256"
upscale_factor = 4
patch_size = 96  # same idea as dataset patches
save_size = 256  # final saved LR image size (optional)

os.makedirs(lr_dir, exist_ok=True)

# === Transforms ===
hr_transform = transforms.Compose([
    transforms.RandomCrop(patch_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

# === Generate Low-Res Images ===
hr_images = sorted(glob(os.path.join(hr_dir, "*.*")))
count = 0

for path in hr_images:
    try:
        img_name = os.path.basename(path)
        hr_img = Image.open(path).convert("RGB")

        # --- Random crop and flip to simulate dataset behavior ---
        hr_tensor = hr_transform(hr_img)
        hr_pil = to_pil(hr_tensor)

        # --- Create low-res by bicubic downsampling ---
        c, h, w = hr_tensor.shape
        lr_size = (h // upscale_factor, w // upscale_factor)
        lr_pil = hr_pil.resize((lr_size[1], lr_size[0]), Image.BICUBIC)

        # --- Optionally upscale back to save size ---
        if save_size:
            lr_pil = lr_pil.resize((save_size, save_size), Image.BICUBIC)

        # --- Save to disk ---
        save_path = os.path.join(lr_dir, img_name)
        lr_pil.save(save_path, "PNG")
        count += 1
        print(f"[OK] {img_name} → saved low-res version")

    except Exception as e:
        print(f"[ERROR] {path}: {e}")

print(f"\n✅ Done! {count} low-resolution images saved in {lr_dir}")