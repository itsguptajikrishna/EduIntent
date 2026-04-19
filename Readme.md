Decoding Pedagogical Intent in Classroom Discourse via Token-Level Contrastive Learning

# Beyond the Words
## AI-Powered Intent Recognition in Hindi-Hinglish Classroom Conversations

> *When a student says "I don't get it" — are they confused, curious, or frustrated? This project is built to know the difference.*

---

## Overview

**EduIntent** is a multimodal intent recognition system trained on a proprietary Hindi/Hinglish dataset collected from real educational classroom scenarios. It implements **TCL-MAP** (Token-Level Contrastive Learning with Modality-Aware Prompting), a state-of-the-art method published at **AAAI 2024**, applied to an entirely new and underexplored domain: the Indian educational classroom.

Unlike existing intent recognition systems built for customer service or smart assistants, this project targets the nuanced, high-context speech patterns of students and teachers — where tone, pace, and language-switching often carry more meaning than the words themselves.

---

## Research Paper

This project implements the following AAAI 2024 paper:

> **Token-Level Contrastive Learning with Modality-Aware Prompting for Multimodal Intent Recognition**  
> Qianrui Zhou, Hua Xu, Hao Li, Hanlei Zhang, Xiaohan Zhang, Yifan Wang, Kai Gao  
> *Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38 No. 15, pp. 17114–17122*  
> DOI: [10.1609/aaai.v38i15.29656](https://doi.org/10.1609/aaai.v38i15.29656)  
> GitHub (original): https://github.com/thuiar/TCL-MAP

---

## What Makes This Project Unique

| Factor | Why It Matters |
|---|---|
| **Education domain** | No large-scale labeled intent dataset exists for classroom discourse in Hindi/Hinglish |
| **Proprietary dataset** | Collected and annotated from scratch — not available anywhere publicly |
| **Hindi/Hinglish language** | Captures real code-switching patterns in Indian academic speech |
| **Token-level fusion** | Intent shifts mid-utterance in educational dialogue — sentence-level models miss this |
| **Modality-aware prompting** | Acoustic cues (pitch, pause, stress) carry distinct meaning in teaching/learning speech |

---

## Dataset

### Overview
A proprietary multimodal dataset collected specifically for the **Education** genre, consisting of classroom dialogue scenarios in Hindi and Hinglish (code-switched Hindi-English).

### Format

| Column | Description |
|---|---|
| `start_time` | Start timestamp of the utterance |
| `end_time` | End timestamp of the utterance |
| `dialogue_hindi` | Utterance transcribed in Hindi script |
| `dialogue_hinglish` | Utterance in Hinglish (romanized code-switched form) |
| `intent` | Intent label from the MintRec taxonomy |

### Intent Labels (MintRec)
Intents are sourced from the **MintRec** benchmark and include categories such as:
`Acknowledge`, `Advise`, `Agree`, `Apologise`, `Comfort`, `Complain`, `Confirm`, `Disagree`, `Explain`, `Inform`, `Joke`, `Oppose`, `Plan`, `Praise`, `Refuse`, `Taunt`, `Thank`, `Warn`, and others.

### Domain
**Genre:** Education — lectures, doubt-clearing sessions, peer explanations, and instructor feedback.

---

## Model Architecture — TCL-MAP

The model consists of two core components:

### 1. Modality-Aware Prompting (MAP)
- Aligns and fuses features from **text**, **video**, and **audio** modalities
- Uses similarity-based modality alignment and a cross-modality attention mechanism
- Generates an optimal multimodal semantic environment for the text modality

### 2. Token-Level Contrastive Learning (TCL)
- Constructs augmented samples using the modality-aware prompt and ground truth labels
- Applies **NT-Xent loss** on the label token
- Uses textual semantic insights from intent labels to guide learning across all modalities

```
[Audio] ──┐
           ├──► MAP Module ──► Fused Prompt ──► TCL Framework ──► Intent Label
[Video] ──┤                                         ▲
           │                                         │
[Text]  ───┘                              NT-Xent on Label Token
```

---

## Project Structure

```
eduintent/
├── data/
│   ├── raw/                  # Raw audio/video recordings
│   ├── annotations/          # Intent-labeled CSV files
│   └── processed/            # Preprocessed features (text, audio, video)
├── models/
│   ├── map_module.py         # Modality-Aware Prompting
│   ├── tcl_framework.py      # Token-Level Contrastive Learning
│   └── tcl_map.py            # Full TCL-MAP model
├── configs/
│   └── config.yaml           # Hyperparameters and paths
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── requirements.txt
└── README.md
```

---

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/eduintent.git
cd eduintent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.12+
- Transformers (HuggingFace)
- librosa (audio feature extraction)
- OpenCV (video feature extraction)
- scikit-learn

---

## Running the Model

### Training
```bash
python train.py --config configs/config.yaml --dataset data/processed/
```

### Evaluation
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --data data/processed/test/
```

---

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Text-only baseline | — | — |
| Multimodal baseline | — | — |
| **TCL-MAP (ours)** | **—** | **—** |

> Results to be updated after training on the EduIntent dataset.

---

## Why the Education Domain?

Most intent recognition research focuses on transactional domains. The classroom is fundamentally different:

- The **same utterance** carries different intent depending on who says it and when
- **Code-switching** (Hindi ↔ English) happens mid-sentence and signals emotional weight
- **Prosodic cues** — a rising tone at the end of a declarative sentence signals a question the text never shows
- No existing public dataset captures **pedagogical intent** in Indian languages

This project is the first attempt to address that gap.

---

## Citation

If you use the TCL-MAP architecture, please cite the original paper:

```bibtex
@article{Zhou_Xu_Li_Zhang_Zhang_Wang_Gao_2024,
  title     = {Token-Level Contrastive Learning with Modality-Aware Prompting for Multimodal Intent Recognition},
  author    = {Zhou, Qianrui and Xu, Hua and Li, Hao and Zhang, Hanlei and Zhang, Xiaohan and Wang, Yifan and Gao, Kai},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {38},
  number    = {15},
  pages     = {17114--17122},
  year      = {2024},
  doi       = {10.1609/aaai.v38i15.29656}
}
```

---

## Acknowledgements

- TCL-MAP architecture by Tsinghua University (THUIAR Lab)
- Intent taxonomy from the MintRec benchmark
- Dataset collected and annotated as part of a proprietary research initiative

---

*EduIntent — Because what's said in a classroom is only half the story.*




