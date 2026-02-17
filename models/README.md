# Pushkarani Temple Pond Classification System

A comprehensive full-stack AI application for classifying Indian temple ponds (Pushkaranis) into three architectural types, analyzing water quality, and providing educational content through an interactive chatbot.

## System Architecture

### Backend (Flask API)
- **Framework**: Flask 2.3.2 with CORS support
- **ML Models**: Multiple deep learning models (DenseNet, EfficientNetV2, ConvNeXt, VGG16, ResNet50, MobileNet, MobileNetV3, Inception, Swin, DINOv2)
- **Features**:
  - Image classification with single model or ensemble voting
  - Water quality analysis (clarity, turbidity, algae detection)
  - Interactive chatbot with Pushkarani knowledge base
  - Model serving and inference
  - Caching for performance optimization

### Frontend (React)
- **Framework**: React 18.2.0 with Hooks
- **Features**:
  - Interactive image upload with drag-and-drop
  - Real-time classification results
  - Water quality analysis visualization
  - AI-powered chatbot interface
  - Educational facts and statistics
  - Contribution/conservation information
  - Responsive design with Tailwind CSS

### AI Models
- **Classification**: 10 different pre-trained models available
- **Ensemble Method**: Combines multiple models for robust predictions
- **Image Size**: 224x224 pixels
- **Classes**: Type-1 (Teppakulam), Type-2 (Kalyani), Type-3 (Kunda)

## Installation & Setup

### Backend Setup

1. **Navigate to backend directory**
```bash
cd models/backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure models are trained**
- Place trained model files in respective folders:
  - `../densenet/best_model.keras`
  - `../efficientnetv2/best_model.keras`
  - `../convnext/best_model.keras`
  - etc.

4. **Run backend server**
```bash
python app.py
```

Server will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd models/frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Configure API endpoint** (if needed)
Edit `.env` file:
```
REACT_APP_API_URL=http://localhost:5000/api
```

4. **Start development server**
```bash
npm start
```

Application will open on `http://localhost:3000`

## API Endpoints

### Classification
- **POST** `/api/predict?model=densenet`
  - Single model prediction
  - Body: `FormData` with `image` file
  - Returns: Prediction class, confidence, probabilities

- **POST** `/api/predict/ensemble`
  - Ensemble prediction using all models
  - Body: `FormData` with `image` file
  - Returns: Averaged predictions from all models

### Water Quality
- **POST** `/api/water-quality`
  - Analyze water quality from image
  - Body: `FormData` with `image` file
  - Returns: Clarity, turbidity, algae likelihood, recommendations

### Chatbot
- **POST** `/api/chatbot`
  - Interactive knowledge base
  - Body: `{ "message": "Your question" }`
  - Returns: Contextual response

### Information
- **GET** `/api/models`
  - List available models
  
- **GET** `/api/facts?count=5`
  - Get random Pushkarani facts
  
- **GET** `/api/conservation`
  - Get conservation and contribution information
  
- **GET** `/api/health`
  - Health check endpoint

## Features

### 1. **Temple Pond Classification**
- Upload images of temple ponds
- Get instant AI-powered classification into three types:
  - **Type-1 (Teppakulam)**: Float festival tanks with central pavilions
  - **Type-2 (Kalyani)**: Geometric stepped tanks with complex architecture
  - **Type-3 (Kunda)**: Simple multipurpose rectangular tanks
- View confidence scores and probability distributions
- Get historical and architectural context for identified type

### 2. **Water Quality Analysis**
- Analyze water clarity and turbidity
- Detect algae presence likelihood
- RGB color analysis
- Get maintenance recommendations based on quality metrics
- Comprehensive quality scoring system

### 3. **Interactive Chatbot**
- Ask questions about Pushkaranis
- Learn about architectural types and history
- Get fascinating facts
- Understand water management engineering
- Learn about conservation efforts

### 4. **Educational Content**
- 100+ fascinating facts about temple ponds
- Visual gallery of different Pushkarani types
- Historical timeline and patronage information
- Engineering achievements and hydraulic systems
- Key statistics and records

### 5. **Conservation & Contribution**
- Understand the conservation challenge
- Learn how to contribute
- Upload and document temple ponds in your area
- Join restoration initiatives
- Raise awareness about sacred water heritage

## Training the Models

### Training Command
```bash
cd models
python train1.py
```

Configure in `train1.py`:
```python
MODEL_NAME = "densenet"  # Change to desired model
EPOCHS = 100
BATCH_SIZE = 8
TRAIN_SPLIT = 0.75
```

Available models:
- mobilenet, resnet50, vgg16, inception, densenet
- efficientnetv2, mobilenetv3, convnext, swin, dinov2
- custom (custom CNN)

### Testing Models
```bash
python test.py
```

## Project Structure

```
models/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── requirements.txt        # Python dependencies
│   └── .env                   # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main application
│   │   ├── App.css            # Comprehensive styling
│   │   ├── index.js           # React entry point
│   │   └── components/        # React components
│   │       ├── ImageUploader.js
│   │       ├── ResultsDisplay.js
│   │       ├── WaterQualityAnalyzer.js
│   │       ├── Chatbot.js
│   │       ├── FactsSection.js
│   │       └── ContributionSection.js
│   ├── public/
│   │   └── index.html
│   ├── package.json
│   ├── .env                   # API configuration
│   └── .gitignore
├── densenet/
│   ├── best_model.keras
│   ├── final_model.keras
│   ├── class_indices.json
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── training_summary.json
├── [other model folders]
├── dataset/
│   ├── type-1/
│   ├── type-2/
│   └── type-3/
├── train.py                   # Basic training script
├── train1.py                  # Advanced training script
└── test.py                    # Testing script
```

## Dependencies

### Backend Requirements
- TensorFlow 2.13.0
- Keras 2.13.0
- Flask 2.3.2
- Flask-CORS 4.0.0
- NumPy 1.24.3
- Pillow 10.0.0
- Scikit-learn 1.3.0
- Matplotlib 3.7.2

### Frontend Requirements
- React 18.2.0
- React-DOM 18.2.0
- Axios 1.4.0
- React-Icons 4.10.1
- React-Toastify 9.1.3
- Tailwind CSS 3.3.2

## Performance

- **Inference Time**: ~500-1500ms per image (varies by model)
- **Accuracy**: 85-95% (depending on model and dataset quality)
- **Memory Usage**: ~2-4GB for all models loaded
- **Response Time**: <2 seconds average API response

## Troubleshooting

### Backend Issues
1. **Models not loading**: Ensure model files exist in correct paths
2. **Port already in use**: Change port in `app.py`
3. **CORS errors**: Check frontend `.env` API_URL configuration

### Frontend Issues
1. **API not connecting**: Verify backend is running on correct port
2. **Images not uploading**: Check file size and format
3. **Slow performance**: Clear browser cache and restart

## Contributing

To contribute:
1. Document temple ponds in your region
2. Improve model accuracy with more training data
3. Add new features or visualizations
4. Report bugs and suggest improvements

## Conservation Impact

- Preserves 2000+ years of architectural heritage
- Supports water security and groundwater recharge
- Protects sacred sites and cultural traditions
- Enables evidence-based conservation planning

## License

This project is dedicated to the preservation of India's sacred water heritage.

## Contact

For questions, contributions, or partnership opportunities:
- Project: Pushkarani Heritage Preservation
- Focus: AI for sacred architecture documentation and conservation
- Mission: Digitally preserve and protect temple ponds

## References

- Agama Shastras (Temple construction texts)
- Vastu Shastra (Spatial planning principles)
- Shilpa Shastra (Architectural manuals)
- Archaeological Survey of India records
- Academic research on Indian water management systems

---

**🏛️ Pushing Forward with AI for Sacred Heritage**
