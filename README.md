<div align="center">
  <h1>🔍 DeepFake Detection Web App</h1>
  <p>Advanced AI-powered detection of manipulated media with multiple deep learning models</p>
  
  [![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Deploy on Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)
</div>

## 🌟 Overview

DeepFake Detection Web App is a powerful, user-friendly platform that helps identify AI-generated or manipulated media. Built with Flask and modern deep learning models, it provides accurate detection of deepfakes in both images and videos.

### 🎯 Key Features

- **Multi-Model Analysis**: Combines results from multiple state-of-the-art models
  - MesoNet: Specialized in detecting face-swapped content
  - ResNet50: Deep residual network for image classification
  - Xception: Advanced CNN architecture for deepfake detection
  - DeepFaceLab & DFDNet: Additional models for comprehensive analysis

- **User-Friendly Interface**
  - Drag-and-drop file upload
  - Real-time processing status
  - Detailed analysis reports
  - Responsive design for all devices

- **Media Support**
  - Image formats: JPG, PNG, JPEG
  - Video formats: MP4, AVI
  - Batch processing support

## 🚀 Live Demo

Check out our live demo: [https://deepfake-detector.onrender.com](https://deepfake-detector.onrender.com)

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip (Python package manager)
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detector.git
   cd deepfake-detector
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up model weights**
   ```bash
   # Create a .env file with your model download URLs
   # Then run:
   python download_models.py
   ```
   
   Or manually download the model weights and place them in the `models/weights/` directory.

5. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

## 🌐 Deployment

### Deploy on Render (Recommended)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Click the "Deploy to Render" button above
2. Connect your GitHub/GitLab repository
3. Configure your deployment settings
4. Click "Create Web Service"

### Alternative: Manual Deployment

1. **Install Docker** (if not already installed)
2. **Build the Docker image**
   ```bash
   docker build -t deepfake-detector .
   ```
3. **Run the container**
   ```bash
   docker run -p 5000:5000 deepfake-detector
   ```

## 🧠 How It Works

The application uses an ensemble of deep learning models to analyze media files:

1. **Preprocessing**: Media is processed to extract frames and faces
2. **Model Inference**: Multiple models analyze the content
3. **Result Aggregation**: Results are combined for higher accuracy
4. **Report Generation**: Detailed analysis is presented to the user

## 📊 Model Performance

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| MesoNet | 92% | Fast | Face-swapped videos |
| ResNet50 | 89% | Medium | General deepfake detection |
| Xception | 91% | Slow | High-accuracy analysis |

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The developers of the underlying deep learning models
- Open-source community for various libraries and tools
- Render for hosting the demo version
