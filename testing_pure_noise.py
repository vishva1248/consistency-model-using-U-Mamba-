import os
import torch
from PIL import Image
from UMamba.umamba import Umamba
CHECKPOINT_PATH = "batch_wise.pth"
OUTPUT_DIR = "generated_images"
IMAGE_SIZE = 64
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
SEED = 42
def save_generated_image(image, output_path):
    image = image.detach().cpu().clamp(-1, 1)
    image = ((image + 1.0) / 2.0 * 255.0)
    image = image.permute(1, 2, 0).byte().numpy()
    Image.fromarray(image).save(output_path)
def Testing_pure_noise():
    print("\n===== U-Mamba Consistency Model Testing =====")
    print("1. Single-step generation")
    print("2. Four-step generation")
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice == "1":
        sampling_mode = "one_step"
        num_steps = 1
    elif choice == "2":
        sampling_mode = "multi_step"
        num_steps = 4
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Mode: {sampling_mode}")
    print(f"Maximum noise scale: {SIGMA_MAX}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = Umamba(device).to(device)
    dummy_x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    dummy_sigma = torch.full((1,), 0.5, device=device)
    with torch.no_grad():
        model(dummy_x, dummy_sigma)
    checkpoint = torch.load(CHECKPOINT_PATH,map_location=device,weights_only=False)
    if "target_model_state_dict" in checkpoint:
        state_dict = checkpoint["target_model_state_dict"]
    elif "main_model_state_dict" in checkpoint:
        state_dict = checkpoint["main_model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint loaded successfully.")
    noise = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    x = SIGMA_MAX * noise
    with torch.no_grad():
        if sampling_mode == "one_step":
            sigma = torch.tensor([SIGMA_MAX], device=device)
            generated = model(x, sigma)
        else:
            ramp = torch.linspace(0, 1, num_steps + 1, device=device)
            sigmas = (SIGMA_MAX ** (1 / 7)+ ramp * (SIGMA_MIN ** (1 / 7) - SIGMA_MAX ** (1 / 7))) ** 7
            for step in range(num_steps):
                current_sigma = sigmas[step].expand(1)
                next_sigma = sigmas[step + 1]
                prediction = model(x, current_sigma)
                if step < num_steps - 1:
                    x = prediction + next_sigma * torch.randn_like(prediction)
                else:
                    generated = prediction
    output_path = os.path.join(OUTPUT_DIR,f"generated_{sampling_mode}.png")
    save_generated_image(generated[0], output_path)
    print(f"\nGeneration completed.")
    print(f"Final image saved to: {output_path}")

