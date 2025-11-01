import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import time

# -------- Residual Block --------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection

# -------- Generator --------
class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16, upscale_factor=4):
        super(Generator, self).__init__()

        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.PReLU()

        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # After residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Upsampling with PixelShuffle
        upsampling = []
        for _ in range(int(upscale_factor/2)):
            upsampling += [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*upsampling)

        # Final output
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out = self.res_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out1 + out  # skip connection
        out = self.upsample(out)
        out = self.conv3(out)
        return out
    

def crop_to_multiple(img, multiple=4):
    w, h = img.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    if (new_w, new_h) != (w, h):
        print(f"[WARN] Cropping image from ({w}x{h}) → ({new_w}x{new_h}) for safe upscaling")
        img = img.crop((0, 0, new_w, new_h))
    return img

def enhance_image(model_path="model.pth", input_image="0001.png", output_image="enhanced_output.jpg", generator=None):
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using device: {device.upper()}")
    print(f"[INFO] Loading model from: {model_path}")

    # Load model
    model = generator

    print("[INFO] Model loaded successfully ✅")

    # Load and preprocess input image
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"❌ Input image not found: {input_image}")

    img = Image.open(input_image).convert("RGB")
    img = crop_to_multiple(img, 4)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    print(f"[INFO] Input image loaded ({img.width}x{img.height})")

    # Run inference
    print("[INFO] Enhancing image... this may take a while on CPU (~10-60s depending on size)")

    with torch.no_grad():
        start_inf = time.time()
        try:
            output_tensor = model(img_tensor)
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            exit(1)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_inf = time.time()

    print(f"[INFO] Inference completed in {end_inf - start_inf:.3f} seconds")

    # Postprocess and save
    output_img = transforms.ToPILImage()(output_tensor.squeeze().cpu())
    output_img.save(output_image)

    total_time = time.time() - start_time
    print(f"[SUCCESS] Enhanced image saved as '{output_image}'")
    print(f"[TIME] Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    enhance_image()