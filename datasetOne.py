#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# ===========================
# 1. Dataset
# ===========================
MAX_GRASPS = 8
IMG_W, IMG_H = 224, 224  # transformed image size

class CornellGraspDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Collect samples, skip backgrounds
        for folder in sorted(os.listdir(root_dir)):
            if folder == 'backgrounds':
                continue
            folder_path = os.path.join(root_dir, folder)
            for rgb_file in glob.glob(os.path.join(folder_path, '*r.png')):
                base_name = rgb_file.replace('r.png', '')
                depth_file = base_name + 'd.tiff'
                pos_file = base_name + 'cpos.txt'
                neg_file = base_name + 'cneg.txt'
                if os.path.exists(depth_file) and os.path.exists(pos_file) and os.path.exists(neg_file):
                    self.samples.append((rgb_file, depth_file, pos_file, neg_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_file, _, pos_file, _ = self.samples[idx]
        rgb = Image.open(rgb_file).convert('RGB')

        # Read positive grasps
        grasps = []
        with open(pos_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                rect = [list(map(float, line.strip().split())) for line in lines[i:i+4]]
                grasps.append(np.array(rect))
        grasps = np.array(grasps, dtype=np.float32)

        if self.transform:
            rgb_transformed = self.transform(rgb)
        else:
            rgb_transformed = transforms.ToTensor()(rgb)

        grasps_tensor = self.process_grasps(grasps)
        return rgb_transformed, grasps_tensor

    def process_grasps(self, grasps, max_grasps=MAX_GRASPS):
        fixed = torch.zeros((max_grasps, 4, 2), dtype=torch.float32)
        if len(grasps) == 0:
            return fixed
        if len(grasps) > max_grasps:
            fixed[:] = torch.tensor(grasps[:max_grasps])
        else:
            fixed[:len(grasps)] = torch.tensor(grasps)
        return fixed

# ===========================
# 2. Transforms
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===========================
# 3. Load dataset
# ===========================
root_dir = 'cornell/archive'
dataset = CornellGraspDataset(root_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Dataset samples:", len(dataset))

# ===========================
# 4. Model
# ===========================

from torchvision.models import resnet18, ResNet18_Weights

class GRConvNet(nn.Module):
    def __init__(self):
        super(GRConvNet, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)  # xc, yc, w, h

    def forward(self, x):
        return self.backbone(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRConvNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# ===========================
# 5. Helper functions
# ===========================

def compute_grasp_target_norm(grasp_tensor):
    """ Convert fixed-size grasp tensor -> normalized [xc, yc, w, h] """
    mask = (grasp_tensor.sum(dim=(1,2)) != 0)
    valid = grasp_tensor[mask]

    if valid.shape[0] == 0:
        return torch.zeros(4)

    rect = valid[0]
    center = rect.mean(dim=0)
    w = rect[:,0].max() - rect[:,0].min()
    h = rect[:,1].max() - rect[:,1].min()

    # Normalize
    xc = center[0] / IMG_W
    yc = center[1] / IMG_H
    w /= IMG_W
    h /= IMG_H
    return torch.tensor([xc, yc, w, h], dtype=torch.float32)

def denormalize_grasp(output):
    """ Convert normalized [xc, yc, w, h] -> pixel coordinates """
    xc = output[0] * IMG_W
    yc = output[1] * IMG_H
    w = output[2] * IMG_W
    h = output[3] * IMG_H

    x0 = xc - w/2
    y0 = yc - h/2
    x1 = xc + w/2
    y1 = yc + h/2
    return np.array([[x0, y0],
                     [x1, y0],
                     [x1, y1],
                     [x0, y1]])

# ===========================
# 6. Visualization
# ===========================
def visualize_batch_side_by_side(dataset, indices):
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 4*len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(indices):
        img_tensor, grasps = dataset[idx]

        # Original image
        orig_path = dataset.samples[idx][0]
        pil_image = Image.open(orig_path).convert('RGB')
        W_orig, H_orig = pil_image.size

        # --- Original ---
        ax_orig = axes[i,0]
        ax_orig.imshow(pil_image)
        ax_orig.set_title("Original")
        ax_orig.axis('off')
        for grasp in grasps:
            if grasp.sum() == 0:
                continue
            rect = grasp.numpy()
            rect = np.vstack([rect, rect[0]])
            ax_orig.plot(rect[:,0], rect[:,1], 'r-', linewidth=2)

        # --- Transformed ---
        img = img_tensor.permute(1,2,0).numpy()
        img = (img*0.229 + 0.485)  # approximate de-normalize for visualization
        img = np.clip(img, 0, 1)

        H, W = img.shape[:2]
        ax_tr = axes[i,1]
        ax_tr.imshow(img)
        ax_tr.set_title("Transformed")
        ax_tr.axis('off')
        for grasp in grasps:
            if grasp.sum() == 0:
                continue
            rect = grasp.numpy()
            rect[:,0] = rect[:,0] * (W / W_orig)
            rect[:,1] = rect[:,1] * (H / H_orig)
            rect = np.vstack([rect, rect[0]])
            ax_tr.plot(rect[:,0], rect[:,1], 'r-', linewidth=2)
    plt.tight_layout()
    plt.show()

# ===========================
# 7. Training loop
# ===========================
weights_path = "grconvnet_weights.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("Loaded pretrained weights. Skipping training.")
else:
    print("No pretrained weights found. Starting training...")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, grasps in train_loader:
            images = images.to(device)
            targets = torch.stack([compute_grasp_target_norm(g) for g in grasps]).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), weights_path)
    print("Training complete. Weights saved.")

# ===========================
# 8. Evaluation
# ===========================
model.eval()
total_loss = 0
with torch.no_grad():
    for images, grasps in test_loader:
        images = images.to(device)
        targets = torch.stack([compute_grasp_target_norm(g) for g in grasps]).to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
print(f"Test Loss: {total_loss/len(test_loader):.4f}")

# ===========================
# 9. Prediction function
# ===========================
def predict_grasp(image_pil):
    model.eval()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)[0].cpu()
    grasp_box = denormalize_grasp(output)
    return grasp_box

# ===========================
# 10. Example visualization
# ===========================
indices_to_show = [0,5,10]
visualize_batch_side_by_side(dataset, indices_to_show)

# Test prediction on a single image
example_img = Image.open(dataset.samples[0][0]).convert('RGB')
pred_box = predict_grasp(example_img)
print("Predicted grasp box (pixel coordinates):")
print(pred_box)

torch.save(model.state_dict(), "grconvnet_weights.pth")
