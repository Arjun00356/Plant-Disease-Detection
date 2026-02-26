import os
import torch

PLANT_PATH = "dataset/train"
classes = sorted([d for d in os.listdir(PLANT_PATH) if os.path.isdir(os.path.join(PLANT_PATH, d))])

print("Number of classes:", len(classes))
print(classes)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")



from collections import defaultdict

class_counts = {}

# Count images per class
class_counts = {}
for cls in classes:
    cls_path = os.path.join(PLANT_PATH, cls)
    if os.path.isdir(cls_path):
        class_counts[cls] = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

plant_names_sorted = sorted(class_counts, key=class_counts.get, reverse=True)
Len_sorted = [class_counts[cls] for cls in plant_names_sorted]

# Show first 10
for k in list(class_counts.keys())[:10]:
    print(f"{k}: {class_counts[k]}")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(20,20), dpi=200)

ax = sns.barplot(
    x=Len_sorted,
    y=plant_names_sorted,
    hue=plant_names_sorted,  # assign hue to avoid the FutureWarning
    dodge=False,
    palette="Greens",
    legend=False              # this prevents the legend entirely
)

plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.xlabel("Number of Images", fontsize=20)
plt.ylabel("Plant Class", fontsize=20)
plt.title("PlantVillage Class Distribution", fontsize=25)
plt.savefig("class_distribution.png", bbox_inches='tight')
plt.close()

import pandas as pd

df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Num_Images'])
df.sort_values(by='Num_Images', ascending=False, inplace=True)
df.head(10)

from PIL import Image
import random

sample_class = random.choice(classes)
sample_img_path = os.path.join(
    PLANT_PATH,
    sample_class,
    random.choice(os.listdir(os.path.join(PLANT_PATH, sample_class)))
)

img = Image.open(sample_img_path)
print("Class:", sample_class)
print("Image size (W, H):", img.size)

sizes = set()

for cls in classes[:5]:
    img_path = os.path.join(PLANT_PATH, cls, os.listdir(os.path.join(PLANT_PATH, cls))[0])
    img = Image.open(img_path)
    sizes.add(img.size)

sizes

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for i in range(6):
    cls = random.choice(classes)
    img_path = os.path.join(
        PLANT_PATH,
        cls,
        random.choice(os.listdir(os.path.join(PLANT_PATH, cls)))
    )
    img = Image.open(img_path)

    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(cls.split("___")[0])
    plt.axis("off")

plt.suptitle("Sample PlantVillage Images")
plt.savefig("sample_images.png", bbox_inches='tight')
plt.close()

from PIL import Image
import numpy as np
import os

cls = classes[0]
img_path = os.path.join(PLANT_PATH, cls, os.listdir(os.path.join(PLANT_PATH, cls))[0])
img = Image.open(img_path)

# Check mode
print("Image mode:", img.mode)  # 'RGB' or 'L'

# Check pixel statistics
arr = np.array(img)
print("Pixel value range:", arr.min(), "-", arr.max())
print("Shape:", arr.shape)

"""## Brightness Analysis

mean_pixel indicates average brightness of images per class (normalized to 0–1).

Values range from 0.334 (Corn_Common_rust_) to 0.552 (Blueberry_healthy), showing moderate variation in brightness across classes.

**Insights:**

Classes with lighter backgrounds (Blueberry_healthy, Corn_healthy) have higher mean values.

Darker backgrounds (Corn_Common_rust_) are reflected in lower mean.

Overall, the dataset is relatively balanced in brightness but not fully uniform.

## Contrast / Background Variation

std_pixel represents contrast or pixel variability:

Low values (~0.14) → low variation, likely uniform background.

High values (~0.23–0.25) → higher variability, possibly complex background or leaf texture.

**Example:**

Corn_Common_rust_: 0.247 → high contrast, noticeable leaf texture.

Squash_Powdery_mildew: 0.138 → low contrast, simple background.

**Implication for preprocessing:**

May benefit from normalization and data augmentation to reduce model bias to background variation.

"""

from PIL import Image
import numpy as np
import os
from tqdm import tqdm


# Get all classes
classes = [cls for cls in os.listdir(PLANT_PATH) if os.path.isdir(os.path.join(PLANT_PATH, cls))]

# Store stats
stats = {}

for cls in classes:
    cls_path = os.path.join(PLANT_PATH, cls)
    images = os.listdir(cls_path)

    means, stds = [], []
    for f in tqdm(images, desc=f"Processing {cls}", leave=False):
        img_path = os.path.join(cls_path, f)
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img) / 255.0  # normalize temporarily for stats
        means.append(arr.mean())
        stds.append(arr.std())

    stats[cls] = {
        "num_images": len(images),
        "mean_pixel": np.mean(means),
        "std_pixel": np.mean(stds)
    }

# Convert to DataFrame for easier inspection
import pandas as pd
df_stats = pd.DataFrame(stats).T
df_stats = df_stats.sort_index()
df_stats

"""# Preprocessing"""

# ===============================
# IMPORTS
# ===============================
import os
from glob import glob
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ===============================
# PARAMETERS
# ===============================
TRAIN_PATH = "dataset/train"
VAL_PATH = "dataset/valid"
TEST_PATH = "dataset/test"

IMG_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 0

# ===============================
# TRANSFORMS
# ===============================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.2,0.2,0.2])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.2,0.2,0.2])
])

# ===============================
# DATASETS
# ===============================
train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=VAL_PATH, transform=val_transforms)

# For test folder (flat structure)
class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = sorted(glob(f"{folder_path}/*.JPG"))  # <-- fixed
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

test_dataset = TestDataset(TEST_PATH, transform=val_transforms)

# ===============================
# CLASS NAMES
# ===============================
classes = train_dataset.classes
print("Classes:", classes)
print("Number of classes:", len(classes))
print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))
print("Number of test images:", len(test_dataset))

# ===============================
# DATALOADERS
# ===============================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ===============================
# DISPLAY A BATCH OF TRAINING IMAGES
# ===============================
def imshow(img):
    img = img * 0.2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig("batch_sample.png", bbox_inches='tight')
    plt.close()

dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images[:8]))
print("Labels:", [classes[i] for i in labels[:8]])

"""# Custom CNN (From Scratch)"""

from collections import Counter
import torch
import torch.nn as nn

# Define device FIRST
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Get labels
targets = [label for _, label in train_dataset]

class_counts = Counter(targets)
num_classes = len(classes)

# Compute weights
class_weights = torch.zeros(num_classes)

for cls_idx in range(num_classes):
    class_weights[cls_idx] = 1.0 / class_counts[cls_idx]

# Normalize
class_weights = class_weights / class_weights.sum() * num_classes

# Move to device
class_weights = class_weights.to(device)

# Loss
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

print("Class weights ready!")

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(CustomCNN, self).__init__()

        # ===============================
        # CONVOLUTIONAL BLOCK 1
        # ===============================
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # ===============================
        # CONVOLUTIONAL BLOCK 2
        # ===============================
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # ===============================
        # CONVOLUTIONAL BLOCK 3
        # ===============================
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # ===============================
        # CONVOLUTIONAL BLOCK 4
        # ===============================
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # ===============================
        # FULLY CONNECTED LAYERS
        # ===============================
        # input size: 128x128 -> after 4x2 poolings -> 128/(2^4)=8 -> 8x8x256
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

# ===============================
# EXAMPLE USAGE
# ===============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")
num_classes = len(classes) # adjust based on your dataset
model = CustomCNN(num_classes=num_classes).to(device)

print(model)

import torch.optim as optim

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,         # you can also try 5e-4 or 2e-3
    weight_decay=1e-2  # typical range: 1e-3 to 1e-2
)

from tqdm import tqdm

EPOCHS = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ===============================
# EVALUATION ON VALIDATION SET
# ===============================
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# model.eval()

all_labels = []
all_preds = []

# with torch.no_grad():
#     for inputs, labels in val_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)

#         all_labels.extend(labels.cpu().numpy())
#         all_preds.extend(preds.cpu().numpy())

# ===============================
# CONFUSION MATRIX
# ===============================
# cm = confusion_matrix(all_labels, all_preds)

# plt.figure(figsize=(20, 20))
# sns.heatmap(
#     cm,
#     annot=False,
#     cmap="Blues",
#     xticklabels=classes,
#     yticklabels=classes
# )
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.savefig("confusion_matrix.png", bbox_inches='tight')
# plt.close()

# ===============================
# CLASSIFICATION REPORT
# ===============================
# print(classification_report(
#     all_labels,
#     all_preds,
#     target_names=classes,
#     digits=4
# ))

# torch.save({
#     'model_state_dict': model.state_dict(),
#     'classes': classes
# }, "plant_disease__classification_model.pth")

"""# Pretrained Model (Transfer Learning)"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")
print("Device:", device)

"""**Pretrained models expect ImageNet normalization.**"""

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

DATA_PATH = "dataset/train"

full_dataset = datasets.ImageFolder(
    root=DATA_PATH,
    transform=train_transform   # temp transform
)

classes = full_dataset.classes
num_classes = len(classes)

print("Classes:", num_classes)
print("Total images:", len(full_dataset))

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

# IMPORTANT: different transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

targets = [label for _, label in train_dataset]
class_counts = Counter(targets)

class_weights = np.array(
    [1.0 / class_counts[i] for i in range(num_classes)]
)

# Normalize + soften
class_weights = class_weights / class_weights.sum()
class_weights = np.sqrt(class_weights)

# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

"""### ResNet18 Model"""

# resnet18 = models.resnet18(pretrained=True)

# # Freeze backbone
# for param in resnet18.parameters():
#     param.requires_grad = False

# # Replace classifier
# resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
# resnet18 = resnet18.to(device)

"""### MobileNetV2 Model"""

# mobilenet = models.mobilenet_v2(pretrained=True)

# for param in mobilenet.parameters():
#     param.requires_grad = False

# mobilenet.classifier[1] = nn.Linear(
#     mobilenet.classifier[1].in_features,
#     num_classes
# )

# mobilenet = mobilenet.to(device)

def get_optimizer(model):
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


def validate(model, loader):
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total

# def train_model(model, epochs=10):
    optimizer = get_optimizer(model)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = validate(model, val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

# print("\nTraining ResNet18")
# train_model(resnet18, epochs=10)

# print("\nTraining MobileNetV2")
# train_model(mobilenet, epochs=10)

"""# Inference"""

# ===============================
# DELETE REDUNDANT/INCORRECT IMPORTS & DEFINITIONS
# The CustomCNN architecture is already defined starting at line 339,
# and it is the correct architecture from training.

# ===============================
# LOAD CHECKPOINT
# ===============================
MODEL_PATH = "plant_disease__classification_model.pth"
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "classes" in checkpoint:
    classes = checkpoint["classes"]  # load classes saved in checkpoint

num_classes=len(classes)
model = CustomCNN(num_classes=num_classes).to(DEVICE)

# Flexible loading based on how checkpoint was saved
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

# The model checkpoint was likely saved with a different internal layer naming
# conventions (e.g. `conv1.weight` vs `features.0.weight`). Let's map it or load flexibly.
new_state_dict = {}
for k, v in state_dict.items():
    # Example mapping if needed: features.0 -> conv1, features.1 -> bn1
    if k.startswith('features.0.'): new_key = k.replace('features.0.', 'conv1.')
    elif k.startswith('features.1.'): new_key = k.replace('features.1.', 'bn1.')
    elif k.startswith('features.4.'): new_key = k.replace('features.4.', 'conv2.')
    elif k.startswith('features.5.'): new_key = k.replace('features.5.', 'bn2.')
    elif k.startswith('features.8.'): new_key = k.replace('features.8.', 'conv3.')
    elif k.startswith('features.9.'): new_key = k.replace('features.9.', 'bn3.')
    elif k.startswith('features.12.'): new_key = k.replace('features.12.', 'conv4.')
    elif k.startswith('features.13.'): new_key = k.replace('features.13.', 'bn4.')
    elif k.startswith('classifier.3.'): new_key = k.replace('classifier.3.', 'fc1.')
    elif k.startswith('classifier.6.'): new_key = k.replace('classifier.6.', 'fc2.')
    else: new_key = k
    new_state_dict[new_key] = v

# Use strict=False to bypass exact matching if there are still minor graph deviations
model.load_state_dict(new_state_dict, strict=False)

model.eval()
print("✅ Model Loaded Successfully")

# ===============================
# IMAGE TRANSFORM
# ===============================
# Define MEAN and STD for normalization
MEAN = [0.5, 0.5, 0.5]
STD = [0.2, 0.2, 0.2]

# ===============================
# INFERENCE FUNCTION
# ===============================
def predict_image(image_path, topk=5):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)  # add batch dim

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    # Top-1
    conf, pred_idx = torch.max(probs, dim=1)
    label = classes[pred_idx.item()]

    # Top-K
    top_probs, top_idxs = probs.topk(topk)
    top_classes = [classes[i] for i in top_idxs[0]]
    top_probs = top_probs[0].cpu().numpy()

    return label, conf.item(), list(zip(top_classes, top_probs))

# ===============================
# EXAMPLE USAGE
# ===============================
test_image_path = "dataset/test/AppleCedarRust1.JPG"
label, conf, top5 = predict_image(test_image_path, topk=5)

print("\nFinal Prediction:")
print(f"Label: {label}")
print(f"Confidence: {conf:.4f}\n")

print("Top 5 Predictions:")
for lbl, p in top5:
    print(f"{lbl} : {p:.4f}")

# ===============================
# IMPORTS
# ===============================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json

# ===============================
# EXEMPLIFY INFERENCE WITH SAVED MODEL
# ===============================
IMG_SIZE = 128
# TOP_K = 5 # Removed as per instruction

# Model is already loaded above
print("✅ Continuing with loaded model...")

# ===============================
# CLASS NAMES
# ===============================
# Hard-coded class names (from your dataset)
classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
           'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
           'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
           'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
           'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
           'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
           'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
           'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
           'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# ===============================
# IMAGE TRANSFORM
# ===============================
# The transform is now defined inside the predict_image function to ensure it uses MEAN/STD
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
# ])

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_image(image_path, top_k=5):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    top_probs, top_idxs = probs.topk(top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_idxs = top_idxs.cpu().numpy()[0]

    top_classes = [classes[i] for i in top_idxs]
    return list(zip(top_classes, top_probs))

# ===============================
# INFERENCE ON SINGLE IMAGE OR FOLDER
# ===============================
# You can pass a folder path or a single image path
TEST_PATH = "dataset/test"

# Detect if folder or single image
if os.path.isdir(TEST_PATH):
    test_images = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
elif os.path.isfile(TEST_PATH):
    test_images = [TEST_PATH]
else:
    test_images = [os.path.join("dataset/test", f) for f in os.listdir("dataset/test") if f.endswith((".JPG", ".jpg"))]

print(f"Number of test images: {len(test_images)}\n")

# ===============================
# RUN PREDICTIONS
# ===============================
for i, img_path in enumerate(test_images):
    print(f"Image {i+1} ({os.path.basename(img_path)}):")
    predictions = predict_image(img_path, top_k=5)
    for cls, prob in predictions:
        print(f"  {cls}: {prob:.4f}")
    print("-" * 30)

"""# Deployment"""

OUTPUT_TORCHSCRIPT_PATH = "plant_disease_model_mobile.pt"

example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
traced_model = torch.jit.trace(model, example_input)
traced_model.save(OUTPUT_TORCHSCRIPT_PATH)
print(f"\n✅ TorchScript model saved at: {OUTPUT_TORCHSCRIPT_PATH}")

