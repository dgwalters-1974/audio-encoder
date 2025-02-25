import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from torchaudio.datasets import LIBRISPEECH
from pathlib import Path
import random
from torch.utils.data import Subset


BASE_DIR = Path.cwd()
CACHE_DIR_TEST = BASE_DIR / ".cache"
CACHE_DIR_TEST.mkdir(parents=True, exist_ok=True)

FINAL_MODEL = "/Users/dgwalters/ML Projects/MLX-5/exp-wisp/whisper_final_model_20250225_132818.pt"
FINETUNED_MODEL = "/Users/dgwalters/ML Projects/MLX-5/exp-wisp/.cache/whisper_checkpoint_20250225_123527_epoch1_batch161.pt"
# Initialize processor and models (original and fine-tuned)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny",
    language="en",
    task="transcribe"
)

# Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the original model (pre-trained version)
ORIGINAL_MODEL = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(DEVICE)

# Load your fine-tuned model
FINETUNED_MODEL_LOADED = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(DEVICE)

# Load the checkpoint and extract just the model weights
checkpoint = torch.load(FINETUNED_MODEL, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Create a new state dict with the correct key names
new_state_dict = {}
for k, v in state_dict.items():
    # Remove the 'model.' prefix if it exists
    if k.startswith('model.'):
        k = k[6:]
    # Map the keys to the HuggingFace format
    if 'encoder.blocks' in k:
        k = k.replace('blocks', 'layers')
    if 'decoder.blocks' in k:
        k = k.replace('blocks', 'layers')
    new_state_dict[k] = v

# Load the remapped state dict
FINETUNED_MODEL_LOADED.load_state_dict(new_state_dict, strict=False)

def transcribe(model, audio):
    # Preprocess the audio; assume audio is a raw waveform sampled at 16kHz
    inputs = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            language="en",
            task="transcribe",
            # Remove forced_decoder_ids to avoid conflict
            forced_decoder_ids=None
        )
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription.strip()

def evaluate_model(model, dataset):
    references = []
    hypotheses = []
    
    for sample in dataset:
        waveform, sample_rate, utterance, *_ = sample
        audio = waveform.squeeze().numpy()
        
        try:
            hypothesis = transcribe(model, audio)
            
            # Skip empty results
            if not hypothesis.strip() or not utterance.strip():
                continue
                
            references.append(utterance.strip())
            hypotheses.append(hypothesis.strip())
            
            # Print for debugging
            print(f"\nReference: {utterance}")
            print(f"Hypothesis: {hypothesis}")
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    if not references or not hypotheses:
        print("No valid transcriptions generated!")
        return 1.0  # Return worst possible WER
        
    # Compute WER using jiwer
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        lambda x: x.split()  # Split into words
    ])
    
    try:
        # Calculate WER for each pair and show examples
        print("\nDetailed WER Examples:")
        total_wer = 0
        valid_pairs = 0
        
        for i in range(len(references)):
            ref = references[i]
            hyp = hypotheses[i]
            
            ref_transformed = " ".join(transformation(ref))
            hyp_transformed = " ".join(transformation(hyp))
            
            try:
                pair_wer = jiwer.wer([ref_transformed], [hyp_transformed])
                total_wer += pair_wer
                valid_pairs += 1
                
                # Print first few examples
                if i < 3:
                    print(f"\nPair {i+1}:")
                    print(f"Reference: '{ref_transformed}'")
                    print(f"Hypothesis: '{hyp_transformed}'")
                    print(f"WER: {pair_wer:.4f}")
            except Exception as e:
                print(f"Error calculating WER for pair {i}: {e}")
                continue
        
        # Calculate average WER
        if valid_pairs == 0:
            print("No valid WER calculations!")
            return 1.0
            
        average_wer = total_wer / valid_pairs
        print(f"\nAverage WER across {valid_pairs} samples: {average_wer:.4f}")
        return average_wer
        
    except Exception as e:
        print(f"Error in WER calculation: {e}")
        return 1.0

def evaluate_models_side_by_side(original_model, finetuned_model, dataset):
    print("\n=== Side by Side Model Comparison ===")
    print("{:<60} | {:<60} | {:<10}".format("Original Model", "Fine-tuned Model", "WER Diff"))
    print("-" * 132)
    
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        lambda x: x.split()
    ])
    
    total_original_wer = 0
    total_finetuned_wer = 0
    valid_pairs = 0
    
    for i, sample in enumerate(dataset):
        waveform, sample_rate, reference, *_ = sample
        audio = waveform.squeeze().numpy()
        
        try:
            # Get transcriptions from both models
            original_hyp = transcribe(original_model, audio)
            finetuned_hyp = transcribe(finetuned_model, audio)
            
            # Skip if any are empty
            if not original_hyp.strip() or not finetuned_hyp.strip() or not reference.strip():
                continue
            
            # Calculate WER for both
            ref_transformed = " ".join(transformation(reference))
            orig_transformed = " ".join(transformation(original_hyp))
            fine_transformed = " ".join(transformation(finetuned_hyp))
            
            original_wer = jiwer.wer([ref_transformed], [orig_transformed])
            finetuned_wer = jiwer.wer([ref_transformed], [fine_transformed])
            wer_diff = original_wer - finetuned_wer
            
            total_original_wer += original_wer
            total_finetuned_wer += finetuned_wer
            valid_pairs += 1
            
            # Print first few examples
            if i < 5:  # Show first 5 examples
                print("\nSample {}:".format(i + 1))
                print("Reference:", reference)
                print("{:<60} | {:<60} | {:<10.4f}".format(
                    original_hyp[:57] + "..." if len(original_hyp) > 57 else original_hyp,
                    finetuned_hyp[:57] + "..." if len(finetuned_hyp) > 57 else finetuned_hyp,
                    wer_diff
                ))
                print("WER: {:.4f} | {:.4f}".format(original_wer, finetuned_wer))
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if valid_pairs == 0:
        print("No valid comparisons made!")
        return
    
    # Calculate and print averages
    avg_original_wer = total_original_wer / valid_pairs
    avg_finetuned_wer = total_finetuned_wer / valid_pairs
    avg_improvement = ((avg_original_wer - avg_finetuned_wer) / avg_original_wer) * 100
    
    print("\n=== Summary Statistics ===")
    print(f"Average Original WER: {avg_original_wer:.4f} ({avg_original_wer*100:.2f}%)")
    print(f"Average Fine-tuned WER: {avg_finetuned_wer:.4f} ({avg_finetuned_wer*100:.2f}%)")
    print(f"Average Improvement: {avg_improvement:.1f}%")
    
    return avg_original_wer, avg_finetuned_wer

dataset = LIBRISPEECH(
    root=CACHE_DIR_TEST,
    url="dev-clean",  # or the appropriate split you need
    download=True
)

# Assume `dataset` is your LIBRISPEECH instance.
# For reproducibility, set a random seed.
random.seed(42)

# Define the desired sample size
sample_size = 100

# Get the total number of samples in the dataset
total_samples = len(dataset)

# Randomly select indices
indices = random.sample(range(total_samples), sample_size)

# Create a subset using the randomly selected indices
test_dataset = Subset(dataset, indices)


# Example: assuming `test_dataset` is your evaluation dataset (list/dataset of dicts with "audio" and "text")
# You can load one using Hugging Face datasets or any other method.
# test_dataset = load_dataset("librispeech_asr", "clean", split="test")  # as an example

# When evaluating models, add more detailed output formatting
print("\n=== Original Model Evaluation ===")
original_wer = evaluate_model(ORIGINAL_MODEL, test_dataset)
print(f"Original model WER: {original_wer:.4f} ({original_wer*100:.2f}%)")

print("\n=== Fine-tuned Model Evaluation ===")
finetuned_wer = evaluate_model(FINETUNED_MODEL_LOADED, test_dataset)
print(f"Fine-tuned model WER: {finetuned_wer:.4f} ({finetuned_wer*100:.2f}%)")

# Print relative improvement
if original_wer > 0:
    relative_improvement = ((original_wer - finetuned_wer) / original_wer) * 100
    print(f"\nRelative WER improvement: {relative_improvement:.1f}%")

print("\nNote: Lower WER is better (0.0 = perfect match, 1.0 = complete mismatch)")

# Run the side-by-side comparison
original_wer, finetuned_wer = evaluate_models_side_by_side(ORIGINAL_MODEL, FINETUNED_MODEL_LOADED, test_dataset)
