# Pushkarani Temple Pond Classification System

A comprehensive full-stack AI application for classifying Indian temple ponds (Pushkaranis) into three architectural types, analyzing water quality, and providing educational content through an interactive chatbot.

## System Architecture

**Backend**: Flask 2.3.2 API with 10 deep learning models (DenseNet, EfficientNetV2, ConvNeXt, VGG16, ResNet50, MobileNet, MobileNetV3, Inception, Swin, DINOv2). Features image classification, ensemble voting, water quality analysis, chatbot, caching, and inference optimization.

**Frontend**: React 18.2.0 with interactive image upload, real-time results, water quality visualization, AI chatbot, educational facts, and conservation information.

**Models**: 224x224 image input, ensemble predictions, three classification types (Type-1: Teppakulam, Type-2: Kalyani, Type-3: Kunda).

## Installation

### Requirements
- Python 3.9+, Node.js 16+, Docker & Docker Compose
- 8GB RAM minimum

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Server runs on `http://localhost:5000`

### Frontend
```bash
cd frontend
npm install
npm start
```
Application opens on `http://localhost:3000`

### Docker Deployment
```bash
docker-compose up -d
```

## API Endpoints

- **POST** `/api/predict` - Classify temple pond image (single or ensemble model)
- **POST** `/api/water-quality` - Analyze water clarity, turbidity, and algae presence
- **POST** `/api/chatbot` - Interactive knowledge base queries
- **GET** `/api/models` - List available deep learning models
- **GET** `/api/facts` - Get educational facts about Pushkaranis
- **GET** `/api/conservation` - Conservation and contribution information
- **GET** `/api/health` - Server health check

## Features

**Classification System**: Three temple pond types (Type-1: Teppakulam float tanks, Type-2: Kalyani geometric stepped tanks, Type-3: Kunda rectangular tanks). Confidence scoring and probability distributions with 85-95% accuracy.

**Water Quality Analysis**: Real-time assessment of clarity, turbidity, algae likelihood, and RGB color analysis. Maintenance recommendations based on quality metrics.

**Interactive Chatbot**: Knowledge base for questions about Pushkaranis, architecture, history, water management engineering, and conservation efforts.

**Educational Content**: 100+ facts about temple ponds, historical timelines, engineering achievements, and hydraulic systems.

**Conservation**: Documentation platform for temple ponds in your region with restoration initiative participation.

## Training Models

```bash
cd models
python train1.py  # Configure MODEL_NAME in script
```

Supported models: mobilenet, resnet50, vgg16, inception, densenet, efficientnetv2, mobilenetv3, convnext, swin, dinov2

Test with: `python test.py`

## Project Structure

```
models/
├── backend/               # Flask API
│   ├── app.py
│   └── requirements.txt
├── frontend/              # React application
│   ├── src/App.js
│   ├── components/        # ImageUploader, ResultsDisplay, Chatbot, WaterQualityAnalyzer, etc.
│   └── package.json
├── [model-namefolders]/   # DenseNet, EfficientNetV2, ConvNeXt, ResNet50, etc.
│   └── best_model.keras
├── dataset/               # Training data (Type-1, Type-2, Type-3)
├── train.py, train1.py   # Training scripts
└── test.py               # Testing script
```

## Technical Stack

**Backend**: TensorFlow 2.13.0, Keras, Flask 2.3.2, NumPy, Pillow, Scikit-learn
**Frontend**: React 18.2.0, Axios, Tailwind CSS, React-Icons, React-Toastify
**Deployment**: Docker & Docker Compose

## Performance Metrics

- **Inference Time**: 500-1500ms per image
- **Model Accuracy**: 85-95% depending on model
- **Memory**: ~2-4GB with all models loaded
- **API Response**: <2 seconds average

## Getting Started

```bash
# Backend
cd backend && pip install -r requirements.txt && python app.py

# Frontend (new terminal)
cd frontend && npm install && npm start

# Docker
docker-compose up -d
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not loading | Verify `.keras` files in model folders |
| Port already in use | Change port in `app.py` config |
| API connection errors | Ensure backend running, check frontend `.env` |
| Slow performance | Clear cache, restart services |

## Contributing & Impact

Help preserve 2000+ years of architectural heritage by documenting temple ponds in your region, improving model accuracy with training data, or contributing features. This project supports water security, groundwater recharge, and protection of sacred cultural sites.

## References

- Shilpa Shastra (Architectural manuals)
- Agama Shastras (Temple construction texts)  
- Archaeological Survey of India records
- Academic research on Indian water management systems

---

**Pushing Forward with AI for Sacred Heritage** 🏛️

