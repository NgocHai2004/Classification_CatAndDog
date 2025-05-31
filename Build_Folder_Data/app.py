import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Khởi tạo mô hình và tải trọng số đã lưu
model = Model_Cat_Dog().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Định nghĩa các phép biến đổi cho hình ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Tải hình ảnh
img_path = "data/test/dogs/dog800.jpg"
image = Image.open(img_path)

# Chuyển đổi hình ảnh
input_tensor = transform(image).unsqueeze(0).to(device)

# Dự đoán
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    label = "Cat" if predicted.item() == 0 else "Dog"
    print(f"Predicted: {label}")

# Hiển thị hình ảnh và kết quả
plt.imshow(image)
plt.title(f"Predicted: {label}")
plt.show()
