import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("학습 데이터:", len(train_dataset))
print("테스트 데이터:", len(test_dataset))

# 첫 번째 데이터 확인
image, label = train_dataset[0]

print("이미지 크기:", image.shape)
print("라벨:", label)
