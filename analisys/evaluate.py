import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights
import ssl

from scripts.occlusion import apply_occlusion

ssl._create_default_https_context = ssl._create_unverified_context

data_dir = "../data"
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Transformacje obrazu
#ResNet18_Weights.DEFAULT.transforms() = EfficientNet_B0_Weights.DEFAULT.transforms() = ImageClassification(
#     crop_size=[224]
#     resize_size=[256]
#     mean=[0.485, 0.456, 0.406]
#     std=[0.229, 0.224, 0.225]
#     interpolation=InterpolationMode.BILINEAR
# )
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#Wczytanie modeli
model_resnet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
model_resnet.eval()

model_efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(device)
model_efficientnet.eval()

#Mapowanie klas ImageNet do cat/dog
#https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
cat_indices = list(range(281, 294))
dog_indices = list(range(151, 269))


def predict_cat_dog(outputs):
    probs = torch.nn.functional.softmax(outputs, dim=1)
    top_idx = torch.argmax(probs, dim=1)
    predictions = []
    for idx in top_idx:
        if idx.item() in cat_indices:
            predictions.append("cat")
        elif idx.item() in dog_indices:
            predictions.append("dog")
        else:
            predictions.append("other")
    return predictions


def compute_accuracy(model, dataloader, occlusion=False, occlusion_percent=20):
    correct = 0
    total = 0
    for images, labels in dataloader:
        if occlusion:
            images = torch.stack([apply_occlusion(img, occlusion_percent=occlusion_percent) for img in images])
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        preds = predict_cat_dog(outputs)
        label_names = ["cat" if l == 0 else "dog" for l in labels]

        for p, l in zip(preds, label_names):
            if p == l:
                correct += 1
            total += 1
    return correct / total



#original accuracy
accuracy_resnet_orig = compute_accuracy(model_resnet, dataloader, occlusion=False)
accuracy_eff_orig = compute_accuracy(model_efficientnet, dataloader, occlusion=False)

print(f"ResNet18 Accuracy (original): {accuracy_resnet_orig:.4f}")
print(f"EfficientNet-B0 Accuracy (original): {accuracy_eff_orig:.4f}")
print()


#occluded accuracy (30%)
accuracy_resnet_occ = compute_accuracy(model_resnet, dataloader, occlusion=True, occlusion_percent=30)
accuracy_eff_occ = compute_accuracy(model_efficientnet, dataloader, occlusion=True, occlusion_percent=30)

print(f"ResNet18 Accuracy (occluded 30%): {accuracy_resnet_occ:.4f}")
print(f"EfficientNet-B0 Accuracy (occluded 30%): {accuracy_eff_occ:.4f}")

#occluded accuracy (100%)
accuracy_resnet_occ = compute_accuracy(model_resnet, dataloader, occlusion=True, occlusion_percent=100)
accuracy_eff_occ = compute_accuracy(model_efficientnet, dataloader, occlusion=True, occlusion_percent=100)

print(f"ResNet18 Accuracy (occluded 100%): {accuracy_resnet_occ:.4f}")
print(f"EfficientNet-B0 Accuracy (occluded 100%): {accuracy_eff_occ:.4f}")

# ResNet18 Accuracy (original): 0.8407
# EfficientNet-B0 Accuracy (original): 0.8594
#
# ResNet18 Accuracy (occluded 30%): 0.6643
# EfficientNet-B0 Accuracy (occluded 30%): 0.8149
#
# ResNet18 Accuracy (occluded 100%): 0.0000
# EfficientNet-B0 Accuracy (occluded 100%): 0.0000
