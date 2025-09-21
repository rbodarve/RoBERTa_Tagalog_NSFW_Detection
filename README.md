# RoBERTa Tagalog NSFW Detection

A fine-tuned RoBERTa model specifically designed for detecting NSFW (Not Safe For Work) content in Tagalog text. This project uses the `danjohnvelasco/roberta-tagalog-base-cohfie-v1` model as a base and fine-tunes it for binary classification of safe vs. inappropriate content.

## Features

- **Tagalog-optimized**: Built specifically for Filipino/Tagalog text understanding
- **Binary Classification**: Distinguishes between Safe (0) and NSFW (1) content
- **Multiple Export Formats**: Supports PyTorch, ONNX, and TensorFlow Lite formats
- **Production Ready**: Includes model optimization and mobile deployment support
- **Comprehensive Evaluation**: Detailed metrics and performance analysis

## Model Architecture

- **Base Model**: `danjohnvelasco/roberta-tagalog-base-cohfie-v1`
- **Task**: Sequence Classification (Binary)
- **Labels**: 
  - `0`: Safe content
  - `1`: NSFW content
- **Max Sequence Length**: 256 tokens
- **Framework**: PyTorch with Transformers library

## Installation

### Prerequisites

```bash
python >= 3.7
torch >= 1.9.0
```

### Dependencies

The script will automatically install required dependencies, but you can install them manually:

```bash
pip install transformers[torch]
pip install datasets
pip install torch
pip install pandas
pip install scikit-learn
pip install onnx
pip install onnxruntime
pip install optimum[onnxruntime]
pip install tensorflow
```

## Quick Start

### 1. Prepare Your Dataset

Create a CSV file named `dataset.csv` with the following structure:

```csv
text,label
"Magandang umaga sa lahat",0
"Kumusta kayo ngayong araw",0
"inappropriate content example",1
```

**Required columns:**
- `text`: The text content to classify
- `label`: Binary label (0 for safe, 1 for NSFW)

### 2. Run Training

```bash
python roberta_tagalog.py
```

The script will:
- Load and validate your dataset
- Split data into training/validation sets
- Fine-tune the RoBERTa Tagalog model
- Export to multiple formats (PyTorch, ONNX, TFLite)
- Generate comprehensive evaluation metrics

### 3. Model Outputs

After training, you'll find:

```
models/
├── roberta_tagalog_nsfw/          # Main PyTorch model
├── roberta_tagalog_nsfw_model.onnx    # ONNX format
└── roberta_tagalog_nsfw_model.tflite  # TensorFlow Lite format

metrics/
└── metrics.txt                        # Evaluation results
```

## Usage

### Loading the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("models/roberta_tagalog_nsfw")
tokenizer = AutoTokenizer.from_pretrained("models/roberta_tagalog_nsfw")

# Prepare text
text = "Magandang umaga sa lahat"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()
    
    label = "Safe" if predicted_class == 0 else "NSFW"
    confidence = predictions[0][predicted_class].item()
    
    print(f"Text: '{text}' -> {label} (confidence: {confidence:.4f})")
```

### Batch Processing

```python
def classify_texts(texts, model, tokenizer):
    results = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
            
        results.append({
            'text': text,
            'label': 'Safe' if predicted_class == 0 else 'NSFW',
            'confidence': confidence
        })
    
    return results
```

## Configuration

### Training Parameters

Key hyperparameters can be modified in the `train_model()` function:

```python
training_args_dict = {
    'num_train_epochs': 3,              # Number of training epochs
    'per_device_train_batch_size': 8,   # Batch size for training
    'learning_rate': 3e-5,              # Learning rate
    'weight_decay': 0.01,               # Weight decay for regularization
    'warmup_steps': 100,                # Warmup steps
    'fp16': True,                       # Mixed precision training
}
```

### Model Configuration

```python
MODEL_NAME = "danjohnvelasco/roberta-tagalog-base-cohfie-v1"
OUTPUT_DIR = "models/roberta_tagalog_nsfw"
MAX_LENGTH = 256  # Maximum sequence length
```

## Dataset Requirements

### Format
- **File**: CSV format
- **Required Columns**: `text`, `label`
- **Labels**: Binary (0 = Safe, 1 = NSFW)

### Data Quality Guidelines

1. **Balanced Dataset**: Ensure reasonable balance between safe and NSFW samples
2. **Text Quality**: Clean, properly encoded Tagalog/Filipino text
3. **Label Accuracy**: Carefully annotated labels for reliable training
4. **Sufficient Size**: Minimum 1000+ samples recommended for good performance

### Sample Dataset Structure

```csv
text,label
"Magandang umaga sa lahat",0
"Salamat sa inyong tulong",0
"Masayang pagdating ng bagong taon",0
"[inappropriate content example]",1
"[inappropriate content example]",1
```

## Evaluation Metrics

The model provides comprehensive evaluation including:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples per class

Results are saved in `metrics/metrics.txt`.

## Export Formats

### 1. PyTorch (Default)
- **Location**: `models/roberta_tagalog_nsfw/`
- **Use Case**: Python applications, further fine-tuning
- **Loading**: `AutoModelForSequenceClassification.from_pretrained()`

### 2. ONNX
- **Location**: `models/roberta_tagalog_nsfw_model.onnx`
- **Use Case**: Cross-platform deployment, optimized inference
- **Benefits**: Hardware acceleration, multiple runtime support

### 3. TensorFlow Lite
- **Location**: `models/roberta_tagalog_nsfw_model.tflite`
- **Use Case**: Mobile applications, edge deployment
- **Benefits**: Reduced model size, mobile optimization

## Performance Considerations

### Training Optimization
- **Mixed Precision**: Enabled by default (`fp16=True`)
- **Gradient Checkpointing**: Available for memory-constrained environments
- **Batch Size**: Adjust based on available GPU memory

### Inference Optimization
- **ONNX Runtime**: Faster inference than PyTorch
- **TensorFlow Lite**: Optimized for mobile deployment
- **Batch Processing**: Process multiple texts together for efficiency

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Enable gradient checkpointing
   - Use `fp16=True`

2. **Dataset Loading Errors**
   - Ensure CSV has proper encoding (UTF-8)
   - Check for missing values in required columns
   - Validate label format (0/1 integers)

3. **Model Loading Issues**
   - Verify model directory structure
   - Check for corrupted checkpoint files
   - Ensure consistent tokenizer and model versions

### Debug Mode

Add debug prints to track training progress:

```python
# Add to training_args_dict
'logging_steps': 10,
'eval_steps': 50,
'save_steps': 100,
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is open source. Please check the license file for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{roberta-tagalog-nsfw,
  title={RoBERTa Tagalog NSFW Detection},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/roberta-tagalog-nsfw}
}
```

## Acknowledgments

- **Base Model**: `danjohnvelasco/roberta-tagalog-base-cohfie-v1`
- **Framework**: Hugging Face Transformers
- **Optimization**: ONNX Runtime and TensorFlow Lite teams

## Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Submit a pull request
- Contact: [your-email@example.com]

---

**⚠️ Important Note**: This model is designed for content moderation purposes. Please ensure responsible use and consider the ethical implications of automated content classification systems.