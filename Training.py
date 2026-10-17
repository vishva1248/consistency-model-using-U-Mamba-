#change files location and the cuda
#import matplotlib.pyplot as plt
import os
import threading
from PIL import Image
import torch
import torch.nn.functional as F
import math
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from UMamba.umamba import Umamba
transform = transforms.Compose([
transforms.Resize((64,64)),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])
class Training(Dataset):
    def __init__(self,data_path,transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = [file for file in os.listdir(data_path)if
        file.lower().endswith(".jpg")]
        self.data.sort()
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        image_name = self.data[index]
        image_path = os.path.join(self.data_path,image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image
def add_noise(x0, t, noise):
    t = t.view(-1, 1, 1, 1)
    xt = x0 + t * noise
    return xt
def ssim_loss_function(img1, img2):
    # Convert image values from [-1, 1] to [0, 1]
    img1 = ((img1 + 1.0) / 2.0).clamp(0, 1)
    img2 = ((img2 + 1.0) / 2.0).clamp(0, 1)
    window_size = 11
    mu1 = F.avg_pool2d(img1, window_size, stride=1)
    mu2 = F.avg_pool2d(img2, window_size, stride=1)
    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu12 = mu1 * mu2
    sigma1_sq = (F.avg_pool2d(img1 * img1, window_size, stride=1) - mu1_sq)
    sigma2_sq = (F.avg_pool2d(img2 * img2, window_size, stride=1)- mu2_sq)
    sigma12 = (F.avg_pool2d(img1 * img2, window_size, stride=1) - mu12)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = ((2 * mu12 + c1) * (2 * sigma12 + c2))
    denominator = ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim = numerator / denominator.clamp_min(1e-8)
    return 1.0 - ssim.mean()
def make_sigmas(num_scales, sigma_min, sigma_max, rho, device):
    ramp = torch.linspace(0, 1, num_scales, device=device)
    lo = sigma_min ** (1.0 / rho)
    hi = sigma_max ** (1.0 / rho)
    return (lo + ramp * (hi - lo)) ** rho
def progressive_scales(step, total_steps, start_scales, end_scales):
    progress = min(max(step / max(total_steps - 1, 1), 0.0),1.0)
    value = math.sqrt(progress * ((end_scales + 1) ** 2 - start_scales ** 2) + start_scales ** 2)
    return max(start_scales,min(end_scales, math.ceil(value - 1.0) + 1))
def adaptive_target_ema(num_scales, start_scales, start_ema=0.95):
    return math.exp(math.log(start_ema) * start_scales / max(num_scales, 1))
load_data = input("Enter yes if you want to resume from the epoch you have left\nand\nno if you want to start the fresh training:-")
if load_data.lower().strip() == "yes":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #training process start from here 
    dataset = Training(data_path = "dataset_20_image/img_align_celeba",transform=transform)
    train_dataset = DataLoader(dataset,batch_size = 32, shuffle=True)
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0
    start_scales = 2
    end_scales = 50
    max_epochs = int(input("Enter number of epochs: "))
    loss = 100
    ema_decay = 0.999
    history_loss = []
    i = 0
    save_requested = False
    main_model = Umamba(device).to(device)
    target_model = Umamba(device).to(device)
    x0 = next(iter(train_dataset)).to(device)
    noise = torch.randn_like(x0).to(device)
    t1 = torch.full((x0.shape[0],), 0.2, device=device)
    x1 = add_noise(x0, t1 , noise)
    t2 = torch.full((x0.shape[0],), 0.7, device=device)
    x2 = add_noise(x0, t2 , noise)
    main_model(x1,t1)
    target_model(x2,t2)
    target_model.load_state_dict(main_model.state_dict())
    for parameters in target_model.parameters():
        parameters.requires_grad = False
    optimizer = torch.optim.Adam(main_model.parameters(), lr=1e-4)
    checkpoint = torch.load("training_state.pth",map_location=device,weights_only=True)
    main_model.load_state_dict(checkpoint["main_model_state_dict"])
    target_model.load_state_dict(checkpoint["target_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    i = checkpoint["step"]
    history_loss = checkpoint["history_loss"]
    #dataLoader
    print(f"------model training on {device}------")
    print("totat length of dataset:- ",len(dataset))
    #training loop
    def terminal_listener():
        global save_requested
        while True:
            command = input().strip().lower()
            if command == "save":
                save_requested = True
                print("------- Save command received -------")
                break
    listener_thread = threading.Thread(target=terminal_listener,daemon=True)
    listener_thread.start()
    input_folder = "training_output/input"
    output_folder = "training_output/output"
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    total_steps = max_epochs * len(train_dataset)
    for epochs in range(epoch, max_epochs):
        for x0 in train_dataset:
            x0 = x0.to(device)
            num_scales = progressive_scales(
                i, total_steps, start_scales, end_scales
            )
            ema_decay = adaptive_target_ema(num_scales, start_scales)
            noise_level = make_sigmas(
                num_scales, sigma_min, sigma_max, rho, device
            )
            batch_size = x0.shape[0]
            index = torch.randint(0,num_scales - 1,(batch_size,),device=device)
            t1 = noise_level[index]
            t2 = noise_level[index + 1]
            noise = torch.randn_like(x0)
            x1 = add_noise(x0, t1, noise)
            x2 = add_noise(x0, t2, noise)
            output_1 = main_model(x2,t2)
            with   torch.no_grad():
                output_2 = target_model(x1,t1)
            mse_loss = torch.mean((output_1 - output_2) ** 2)
            ssim_loss = ssim_loss_function(output_1, output_2)
            lambda_ssim = 0.1
            loss = mse_loss + lambda_ssim * ssim_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_model.parameters(),max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                for target_param, main_param in zip(
                        target_model.parameters(), main_model.parameters()
                ):
                    target_param.mul_(ema_decay).add_(
                        main_param, alpha=1.0 - ema_decay
                    )

                # Copy BatchNorm buffers from the online model.
                main_buffers = dict(main_model.named_buffers())

                for name, target_buffer in target_model.named_buffers():
                    if name in main_buffers:
                        target_buffer.copy_(main_buffers[name])

            history_loss.append(loss.item())
            i= i+1
            print(
                f"Epoch: {epoch} | "
                f"MSE: {mse_loss.item():.6f} | "
                f"SSIM Loss: {ssim_loss.item():.6f} | "
                f"Total Loss: {loss.item():.6f} | "
                f"Batch: {i}"
            )
            if i % 10 == 0:
                torch.save(main_model.state_dict(),"batch_wise.pth")
                print("------- weights.pth updated -------")
            # if loss <= 1e-4:
            #     print("loss got under 1e-4")
            #     torch.save(
            #         main_model.state_dict(),
            #         "/home/intern/Vishva/Test/consistency2.0/inner_loop_end_weights.pth"
            #     )
            if save_requested:
                print("------- Saving current weights -------")
                torch.save(main_model.state_dict(),"weights.pth")
                print("weights.pth saved successfully.")
                break
        if save_requested:
            break
        epoch += 1
        input_image = x1[0].detach().cpu()
        output_image = output_1[0].detach().cpu()
        input_image = (input_image * 0.5 + 0.5).clamp(0, 1)
        output_image = (output_image * 0.5 + 0.5).clamp(0, 1)
        save_image(input_image, f"{input_folder}/epoch_{i:03d}.png")
        save_image(output_image, f"{output_folder}/epoch_{i:03d}.png")
        torch.save({
            "epoch": epoch,
            "step": i,
            "main_model_state_dict": main_model.state_dict(),
            "target_model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history_loss": history_loss,
        }, "training_state.pth")
        print(f"Training state saved after epoch {epoch}")
    print(f"Epoch :- {epoch} | Consisteny loss :- {loss.item()}")
    model_save_weight_path = "weights.pth"
    torch.save(target_model.state_dict(),model_save_weight_path)
    print(history_loss)
elif load_data.lower().strip() == "no":
    print("no")
    #training process start from here 
    dataset = Training(data_path = "dataset_20_image/img_align_celeba",transform=transform)
    train_dataset = DataLoader(dataset,batch_size = 32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0
    start_scales = 2
    end_scales = 50
    max_epochs = int(input("Enter number of epochs: "))
    loss = 100
    epoch = 0
    ema_decay = 0.999
    history_loss = []
    i = 0
    save_requested = False
    main_model = Umamba(device).to(device)
    target_model = Umamba(device).to(device)
    x0 = next(iter(train_dataset)).to(device)
    noise = torch.randn_like(x0).to(device)
    t1 = torch.full((x0.shape[0],), 0.2, device=device)
    x1 = add_noise(x0, t1 , noise)
    t2 = torch.full((x0.shape[0],), 0.7, device=device)
    x2 = add_noise(x0, t2 , noise)
    main_model(x1,t1)
    target_model(x2,t2)
    target_model.load_state_dict(main_model.state_dict())
    for parameters in target_model.parameters():
        parameters.requires_grad = False
    optimizer = torch.optim.Adam(main_model.parameters(), lr=1e-4)
    #dataLoader
    print(f"------model training on {device}------")
    print("totat length of dataset:- ",len(dataset))
    #training loop
    def terminal_listener():
        global save_requested
        while True:
            command = input().strip().lower()
            if command == "save":
                save_requested = True
                print("------- Save command received -------")
                break
    listener_thread = threading.Thread(target=terminal_listener,daemon=True)
    listener_thread.start()
    input_folder = "training_output/input"
    output_folder = "training_output/output"
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    total_steps = max_epochs * len(train_dataset)
    for epochs in range(max_epochs):
        for x0 in train_dataset:
            x0 = x0.to(device)
            num_scales = progressive_scales(
                i, total_steps, start_scales, end_scales
            )
            ema_decay = adaptive_target_ema(num_scales, start_scales)
            noise_level = make_sigmas(
                num_scales, sigma_min, sigma_max, rho, device
            )
            batch_size = x0.shape[0]
            index = torch.randint(0,num_scales - 1,(batch_size,),device=device)
            t1 = noise_level[index]
            t2 = noise_level[index + 1]
            noise = torch.randn_like(x0)
            x1 = add_noise(x0, t1, noise)
            x2 = add_noise(x0, t2, noise)
            output_1 = main_model(x2,t2)
            with   torch.no_grad():
                output_2 = target_model(x1,t1)
            mse_loss = torch.mean((output_1 - output_2) ** 2)
            ssim_loss = ssim_loss_function(output_1, output_2)
            lambda_ssim = 0.1
            loss = mse_loss + lambda_ssim * ssim_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_model.parameters(),max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                for target_param, main_param in zip(
                        target_model.parameters(), main_model.parameters()
                ):
                    target_param.mul_(ema_decay).add_(
                        main_param, alpha=1.0 - ema_decay
                    )

                # Copy BatchNorm buffers from the online model.
                main_buffers = dict(main_model.named_buffers())

                for name, target_buffer in target_model.named_buffers():
                    if name in main_buffers:
                        target_buffer.copy_(main_buffers[name])

            history_loss.append(loss.item())
            i= i+1
            print(
                f"Epoch: {epoch} | "
                f"MSE: {mse_loss.item():.6f} | "
                f"SSIM Loss: {ssim_loss.item():.6f} | "
                f"Total Loss: {loss.item():.6f} | "
                f"Batch: {i}"
            )
            if i % 10 == 0:
                torch.save(main_model.state_dict(),"batch_wise.pth")
                print("------- weights.pth updated -------")
            # if loss <= 1e-4:
            #     print("loss got under 1e-4")
            #     torch.save(
            #         main_model.state_dict(),
            #         "/home/intern/Vishva/Test/consistency2.0/inner_loop_end_weights.pth"
            #     )
            if save_requested:
                print("------- Saving current weights -------")
                torch.save(main_model.state_dict(),"weights.pth")
                print("weights.pth saved successfully.")
                break
        if save_requested:
            break
        epoch += 1
        input_image = x1[0].detach().cpu()
        output_image = output_1[0].detach().cpu()
        input_image = (input_image * 0.5 + 0.5).clamp(0, 1)
        output_image = (output_image * 0.5 + 0.5).clamp(0, 1)
        save_image(input_image, f"{input_folder}/epoch_{i:03d}.png")
        save_image(output_image, f"{output_folder}/epoch_{i:03d}.png")
        torch.save({
            "epoch": epoch,
            "step": i,
            "main_model_state_dict": main_model.state_dict(),
            "target_model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history_loss": history_loss,
        }, "training_state.pth")
        print(f"Training state saved after epoch {epoch}")
    print(f"Epoch :- {epoch} | Consisteny loss :- {loss.item()}")
    model_save_weight_path = "weights.pth"
    torch.save(target_model.state_dict(),model_save_weight_path)
    print(history_loss)
else:
    print("invalid run main.py again")
