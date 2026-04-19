import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import cv2
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, Wav2Vec2Model
import timm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN = 768
PROMPT_LEN = 10
TEMPERATURE = 0.1
MAX_TEXT_LEN = 64
NUM_LABELS = 10  # change based on dataset

# =========================
# DATASET
# =========================
class MultimodalDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_excel("Dataset.xlsx")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        return waveform.squeeze(0)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < 8:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            frames = [torch.zeros(3, 224, 224)]

        return torch.stack(frames)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["text"])
        audio = self.load_audio(row["audio_path"])
        video = self.load_video(row["video_path"])
        label = int(row["intent"])

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt"
        )

        return {
            "text": {k: v.squeeze(0) for k, v in text_inputs.items()},
            "audio": audio,
            "video": video,
            "label": torch.tensor(label)
        }

# =========================
# FEATURE EXTRACTOR
# =========================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.swin.head = nn.Identity()

    def forward(self, text_inputs, audio, video):
        Z_text = self.bert(**text_inputs).last_hidden_state

        audio = audio.unsqueeze(0)
        F_audio = self.wav2vec(audio).last_hidden_state

        B, T, C, H, W = video.shape
        video = video.view(-1, C, H, W)
        F_video = self.swin(video).view(B, T, -1)

        return Z_text, F_audio, F_video

# =========================
# MAP (SBMA + PROMPT)
# =========================
class MAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_tokens = nn.Parameter(torch.randn(PROMPT_LEN, HIDDEN))
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN)
        )
        self.attn = nn.MultiheadAttention(HIDDEN, 8, batch_first=True)

    def forward(self, Z_text, F_audio, F_video):
        B = Z_text.size(0)
        T = self.prompt_tokens.unsqueeze(0).repeat(B, 1, 1)

        # SBMA
        T_norm = F.normalize(T, dim=-1)
        V_norm = F.normalize(F_video, dim=-1)
        A_norm = F.normalize(F_audio, dim=-1)

        M_TV = torch.softmax(T_norm @ V_norm.transpose(1, 2), dim=-1)
        M_TA = torch.softmax(T_norm @ A_norm.transpose(1, 2), dim=-1)

        V_hat = self.mlp(M_TV @ F_video)
        A_hat = self.mlp(M_TA @ F_audio)

        prompt, _ = self.attn(query=T, key=V_hat, value=A_hat)

        return prompt

# =========================
# FUSION
# =========================
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, Z):
        return self.linear(Z)

# =========================
# CONTRASTIVE LOSS
# =========================
def nt_xent(z1, z2, temperature=TEMPERATURE):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    N = z1.size(0)
    labels = torch.arange(N).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2*N).bool().to(z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

    logits = similarity_matrix / temperature

    return F.cross_entropy(logits, labels)

# =========================
# MODEL
# =========================
class TCLMAP(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.extractor = FeatureExtractor()
        self.map = MAP()
        self.fusion = Fusion()

        self.classifier = nn.Linear(HIDDEN, num_labels)

        self.label_embed = nn.Embedding(num_labels, HIDDEN)
        self.mask_token = nn.Parameter(torch.randn(HIDDEN))

    def forward(self, batch):
        text = {k: v.to(DEVICE) for k, v in batch["text"].items()}
        audio = batch["audio"].to(DEVICE)
        video = batch["video"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        Z_text, F_audio, F_video = self.extractor(text, audio, video)
        Z_prompt = self.map(Z_text, F_audio, F_video)

        label_token = self.label_embed(labels).unsqueeze(1)
        mask_token = self.mask_token.unsqueeze(0).unsqueeze(1).repeat(Z_text.size(0), 1, 1)

        Ze = torch.cat([Z_text, Z_prompt, label_token], dim=1)
        Z = torch.cat([Z_text, Z_prompt, mask_token], dim=1)

        Ze = self.fusion(Ze)
        Z = self.fusion(Z)

        z_label = Ze[:, -1]
        z_mask = Z[:, -1]

        loss_con = nt_xent(z_mask, z_label)

        pooled = Z.mean(dim=1)
        logits = self.classifier(pooled)

        loss_cls = F.cross_entropy(logits, labels)

        return loss_con + loss_cls, logits

# =========================
# EVALUATION
# =========================
def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            _, logits = model(batch)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    print("\n===== Evaluation =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("======================\n")

    model.train()

# =========================
# TRAIN
# =========================
def train():
    dataset = MultimodalDataset("manifest.csv")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = TCLMAP(NUM_LABELS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(5):
        print(f"\n===== Epoch {epoch+1} =====")

        total_loss = 0

        for batch in train_loader:
            loss, _ = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Training Loss: {total_loss/len(train_loader):.4f}")

        evaluate(model, val_loader)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    train()