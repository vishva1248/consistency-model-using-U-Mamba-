import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from UMamba.umamba import Umamba
def convert_image(x):
    x = x[0].detach().cpu()
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0)
    return x.numpy()
def Testing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"------ Testing phase running on {device} ------")
    test_model = Umamba(device).to(device)
    x = torch.randn(1, 3, 64, 64).to(device)
    t = torch.tensor([0.7], device=device)
    with torch.no_grad():
        test_model(x, t)
    test_model.load_state_dict(torch.load("batch_wise.pth", map_location=device))
    test_model.eval()
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    image_path = "/home/mqcd/vishva/consistency2.0/dataset_2/img_align_celeba/000001.jpg"
    original = Image.open(image_path).convert("RGB")
    x0 = transform(original).unsqueeze(0).to(device)

    # Use the same noise for every diagnostic level
    torch.manual_seed(42)
    noise = torch.randn_like(x0)

    noise_levels = [0.002, 0.05, 0.2, 1.0, 10.0, 80.0]

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8)
    )
    axes = axes.flatten()

    axes[0].imshow(convert_image(x0))
    axes[0].set_title("Original")
    axes[0].axis("off")

    test_model.eval()

    with torch.no_grad():
        for i, sigma in enumerate(noise_levels, start=1):
            t = torch.tensor([sigma], device=device)
            x_noisy = x0 + sigma * noise
            output = test_model(x_noisy, t)

            mse = torch.mean((output - x0) ** 2).item()

            axes[i].imshow(convert_image(output))
            axes[i].set_title(
                f"sigma={sigma}\nMSE={mse:.4f}"
            )
            axes[i].axis("off")

            print(
                f"sigma={sigma:7.3f} | "
                f"MSE={mse:.6f} | "
                f"output mean={output.mean().item():.4f} | "
                f"output std={output.std().item():.4f}"
            )

    plt.tight_layout()
    plt.savefig("diagnostic_noise_levels.png", dpi=150)
    plt.show()
