# 파이토치 시작하기
<br>

> # **Tensors**
```
import torch
import numpy as np
```
## 텐서 초기화
### 1. 텐서 직접 생성
```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

### 2. NumPy배열로 텐서 생성
```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

### x_data의 속성(크기/구조)유지
```python
# 모든 값은 1
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

# 모든 값은 랜덤
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")
```

### 텐서 생성
```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```
<br>

## 텐서의 속성
```python
tensor = torch.rand(3,4)
```
### 크기
```python
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
```
### 저장된 장치(CPU/GPU)
```python
print(f"Device tensor is stored on: {tensor.device}")
```
<br>

## 텐서 연산
```python
# GPU로 텐서 이동
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```

### 인덱싱 & 슬라이싱 (like numpy)
```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0 # 두 번째 열의 모든 값 0으로 설정
print(tensor)
```

### 텐서 합치기
```python
t1 = torch.cat([tensor, tensor, tensor], dim=1) # 열 방향으로 합치기
print(t1)
```

### 산술 연산
```python
# Matrix Multiplication
# y1=y2=y3, tensor.T는 전치(transpose)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1) # 저장을 위한 텐서를 생성한 것
torch.matmul(tensor, tensor.T, out=y3) # 행렬 곱 결과를 y3에 저장
```

### Element-wise Multiplication
```python
# z1=z2=z3
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

### single-element 텐서의 요소 합계
```python
agg = tensor.sum() # tensor(12.)
agg_item = agg.item() # 12.0
print(agg_item, type(agg_item))
```

### in-place 연산
```python
print(f"{tensor} \n")
tensor.add_(5) # 각 요소에 5 더하기
print(tensor)
```
<br>

## NumPy 변환
### Tensor to NumPy
```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1) # 각 요소에 1 더하기
print(f"t: {t}")
print(f"n: {n}")
```

### NumPy to Tensor
```python
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n) # 각 요소에 1 더하기
print(f"t: {t}")
print(f"n: {n}")
```
<br><br>

> # **Datasets & DataLoaders**
- Dataset: 데이터와 레이블(정답)을 관리
- DataLoader: Dataset의 데이터를 배치 단위로 가져오고, 효율적인 순회를 도움
- Fashion-MNIST dataset(28x28 grayscale img, 60k:10k, 10 labels)

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
```

## 데이터셋 불러오기
- root: 학습/테스트 데이터 저장 경로
- train: 학습/테스트 데이터 구별
- download=True: root에 데이터 없을 시, 인터넷에서 다운로드
- transform: feature & label transformation
```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

## 데이터셋을 순회하고 시각화하기
- Dataset에 list처럼 직접 접근(index) 가능
```python
# mapping label
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

## 사용자 정의 데이터셋 만들기
- **init**, **len**, and **getitem** 3가지 함수 구현하기
```python
import os
import pandas as pd
from torchvision.io import read_image
```
```py
class CustomImageDataset(Dataset):
    # 초기화. Dataset 객체 생성될 때 한 번만 실행.
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # Dataset 샘플 개수 반환
    def __len__(self):
        return len(self.img_labels)

    # idx의 샘플을 Dataset에서 불러오고 반환
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path) # 이미지를 텐서(C,H,W)로 읽어들임
        label = self.img_labels.iloc[idx, 1]
        if self.transform: # 이미지 텐서로 변환환
            image = self.transform(image)
        if self.target_transform: # 레이블을 원핫 인코딩
            label = self.target_transform(label)
        return image, label # (img, lbl)
```

## DataLoader로 학습용 데이터 준비하기
- batch로 샘플 전달, shuffle로 overfit 방지
```py
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## DataLoader를 통해 순회하기
- 데이터를 배치 단위로 순회
```py
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}") # (배치,C,H,W)
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```
<br><br>

> # **Transform**
- 데이터를 학습에 적합한 구조로 만듦
- torchvision은 feature & label 2개를 변환(transform & target_transform)
```py
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    # 위치 y의 값을 1로 설정
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```
<br><br>

> # **Build Model**
- layer & module로 구성
```py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

## 클래스 정의하기
- 모델을 nn.Module의 하위클래스로 정의
```py
class NeuralNetwork(nn.Module):
    # layer 초기화
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        # 모든 클래스에 대한 점수 반환
        return logits

model = NeuralNetwork().to(device)
print(model) # 모델의 전체 구조 출력

X = torch.rand(1, 28, 28, device=device) # 랜덤 입력 데이터 생성
# forward 메서드 실행(model.forward() 직접 호출 안 함)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits) # 점수->확률
y_pred = pred_probab.argmax(1) # logit 가장 높은 클래스의 idx 반환
print(f"Predicted class: {y_pred}")
```

## Layer
```py
input_image = torch.rand(3,28,28) # 3 batch
print(input_image.size())
```

## nn.Flatten
- nn.Flatten 초기화. 2D->1D.
```py
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

## nn.Linear
- weight & bias로써 입력에 linear transformation 적용
```py
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

## nn.ReLU
- 비선형성을 도입하여 신경망이 복잡한 패턴을 학습 가능
```py
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

## nn.Sequential
- 순차적인 layer 연결. 순서가 중요한 신경망 구조를 구현할 때 유용.
```py
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

## nn.Softmax
- 확률값
```py
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## 모델 매개변수
```py
print(f"Model structure: {model}\n\n")

# 모든 layer의 매개변수 이름과 값 반환
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```
<br><br>

> # **Autograd**
## torch.autograd
[연산 그래프](https://tutorials.pytorch.kr/_images/comp-graph.png)
```py
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b # logit
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

## Gradient 계산하기
- 역전파 연산
- requires_grad=True로 설정된 노드에 대해서만 계산 가능
```py
loss.backward()
print(w.grad)
print(b.grad)
```

## Gradient 계산 멈추기
- 순전파 연산
- 모델을 적용만 하는 경우:
1. torch.no_grad() 블럭으로 감싸기
```py
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```
2. detach() 메소드 적용
```py
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```
<br><br>

> # **Optimization**
## 모델 매개변수 최적화하기
- Hyperparameter: learning_rate, batch_size, epochs
- 학습 단계
    1. zero_grad(): gradient 초기화
    2. loss.backward(): gradient 계산
    3. optimizer.step(): hyperparameter 업데이트
```py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # train 모드
    model.train()
    # 데이터 배치 단위로 가져오기
    for batch, (X, y) in enumerate(dataloader):
        # prediction & loss 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 100배치마다 loss & 진행상태 출력
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # evaluation 모드
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # 맞춘 개수
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # 모든 배치의 평균 손실
    test_loss /= num_batches
    # 정확도 계산
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# loss func 초기화
loss_fn = nn.CrossEntropyLoss()
# optimizer 초기화
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```
<br><br>

> # **Save & Load Model**
```py
import torch
import torchvision.models as models
```
## 모델 가중치 저장하고 불러오기
```py
# state_dict에 매개변수 저장
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# 빈 모델 생성
model = models.vgg16()
# 저장된 가중치 불러오기
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```