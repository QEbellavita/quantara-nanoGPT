# Quantara nanoGPT - Emotion Training

Train emotion-aware GPT models for the Quantara Neural Ecosystem.

## Quick Start

### 1. Prepare Emotion Data
```bash
cd quantara-nanoGPT
python data/quantara_emotion/prepare.py
```

This loads emotion datasets from `~/Downloads/`:
- `text_emotion.csv` (40K tweets)
- `Emotion_classify_Data.csv` (6K comments)
- `text.csv` (416K samples)
- `archive (4) 3/` (train/test/val)

### 2. Train Model

**Fast training (15-30 min on MacBook):**
```bash
python train.py config/train_quantara_emotion_fast.py
```

**Full training (GPU server):**
```bash
python train.py config/train_quantara_emotion.py --device=cuda
```

### 3. Generate Samples
```bash
# Simple generation
python sample_emotion.py --emotion joy --prompt "I feel"

# Interactive mode
python sample_emotion.py --interactive
```

### 4. Integrate with Quantara

```python
from quantara_integration import QuantaraEmotionGPT

model = QuantaraEmotionGPT(checkpoint_path='out-quantara-emotion/ckpt.pt')

# Generate emotion-aware response
response = model.generate("I'm feeling stressed", emotion="empathy")

# Analyze text emotion
analysis = model.analyze_emotion("This is the best day ever!")

# Get coaching response
coaching = model.get_coaching_response(
    user_message="I've been feeling overwhelmed",
    biometric_data={'heart_rate': 85, 'hrv': 45}
)
```

## Integration Points

| Component | Connection |
|-----------|------------|
| Neural Workflow AI Engine | Emotion-aware workflow triggers |
| AI Conversational Coach | Empathetic response generation |
| Emotion-Aware Training Engine | Real-time emotion detection |
| Psychology Emotion Database | Therapy technique recommendations |
| Biometric Integration | Cross-reference HR/HRV with emotional state |
| Dashboard Data | Real-time sentiment analytics |

## API Endpoints

Start the API server:
```python
from flask import Flask
from quantara_integration import QuantaraEmotionGPT, create_api_routes

app = Flask(__name__)
model = QuantaraEmotionGPT()
create_api_routes(app, model)
app.run(port=5000)
```

Endpoints:
- `POST /api/emotion/generate` - Generate emotion-aware text
- `POST /api/emotion/analyze` - Analyze emotional content
- `POST /api/emotion/coach` - Get coaching response

## Training Configs

| Config | Use Case | Time | Hardware |
|--------|----------|------|----------|
| `train_quantara_emotion_fast.py` | Testing/Debug | 15-30 min | MacBook |
| `train_quantara_emotion.py` | Production | 2-4 hours | GPU |

## Emotion Tags

The model understands these emotion tags:
- `<sadness>`, `<joy>`, `<love>`, `<anger>`, `<fear>`, `<surprise>`
- `<empathy>` - Empathetic responses
- `<therapy>` - Therapy techniques
- `<transition>` - Emotion transitions
