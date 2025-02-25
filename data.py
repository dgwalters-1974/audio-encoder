import torch
from torch.utils.data import Dataset, DataLoader
import whisper
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn as nn
from datasets import load_dataset
import torchaudio
from torchaudio.datasets import LIBRISPEECH
import os
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from datetime import datetime

BASE_DIR = Path.cwd()
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LibriSpeechWhisperDataset(Dataset):
    def __init__(self, split="test-clean", device=DEVICE, root=CACHE_DIR, name="tiny.en"):
        self.dataset = LIBRISPEECH(
            root=root,
            url=split,  # or the appropriate split you need
            download=True
        )
        self.device = device
        self.model = whisper.load_model(name=name, device=device)
        self.pad_token = 50257  # Whisper's padding token
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        # Get audio, sample rate, and text from dataset
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000  # LibriSpeech should be 16kHz
        
        # Convert audio to mono if stereo and flatten
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0)
        else:
            audio = audio.flatten()
        
        # Process audio with Whisper
        audio = whisper.pad_or_trim(audio).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        # Use the whispertokenizer directly from whisper package
        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)  # Use False for English-only models
        tokens = tokenizer.encode(text)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(self.device)
        
        return (mel, tokens_tensor)

    @staticmethod
    def collate_fn(batch):
        """
        batch: a list of tuples (mel, tokens_tensor)
        Returns a batch of mels (stacked) and a batch of padded token sequences.
        """
        mels, tokens = zip(*batch)
        
        # Stack mels (they should already be the same shape if audio is fixed-length)
        mels = torch.stack(mels)  # shape: [batch_size, 80, T]
        
        # Pad token sequences to the same length
        padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=50257)
        
        return mels, padded_tokens

def trainer(model, data_loader, num_epochs=1, save_dir=CACHE_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")
        
        for counter, (mels, tokens) in enumerate(pbar):
            pred = model(tokens=tokens, mel=mels)
            targets = tokens[:, 1:]
            pred = pred[:, :-1, :]
            loss = criterion(pred.transpose(1, 2), targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if counter % 10 == 0:
                print(f"Batch {counter+1} Loss: {loss.item():.4f}")
                
                # Save checkpoint with timestamp
                checkpoint_path = save_dir / f"whisper_checkpoint_{timestamp}_epoch{epoch+1}_batch{counter+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'batch': counter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'timestamp': timestamp
                }, checkpoint_path)
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(data_loader)
        
        # Save best model with timestamp
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'timestamp': timestamp
            }, save_dir / f"whisper_best_model_{timestamp}.pt")
    
    # Save final model with timestamp
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_loss': avg_epoch_loss,
        'timestamp': timestamp
    }, save_dir / f"whisper_final_model_{timestamp}.pt")
    print(f"Model saved to {save_dir}")
    print(f"Timestamp: {timestamp}")



if __name__ == "__main__":
    # trainer(model, data_loader, num_epochs=10)
    dataset_instance = LibriSpeechWhisperDataset()
    data_loader = DataLoader(dataset_instance, batch_size=12, shuffle=True, collate_fn=LibriSpeechWhisperDataset.collate_fn)
    model = whisper.load_model(
        "tiny.en",
        device=DEVICE
    )
    trainer(model=model, data_loader=data_loader, num_epochs=1)
    