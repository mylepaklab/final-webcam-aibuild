## 🧠 Model Architecture

The model uses a basic Convolutional Neural Network (CNN) architecture:

- **Input:** 256x256 RGB images
- **Layers:**
  - `Rescaling(1./255)`
  - `Conv2D` + `ReLU`
  - `MaxPooling2D`
  - `Conv2D` + `ReLU`
  - `MaxPooling2D`
  - `Conv2D` + `ReLU`
  - `MaxPooling2D`
  - `Flatten`
  - `Dense` (ReLU)
  - `Dropout`
  - `Dense` (Softmax)

**Loss Function:** `sparse_categorical_crossentropy`  
**Optimizer:** `adam`
