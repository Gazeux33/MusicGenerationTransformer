

from torch.utils.data import DataLoader
import torch.nn as nn

from config import *
from src.dataset import MusicDataset
from src.model import MusicModel
from src.trainer import MusicTrainer
from src.utils import load_parsed_files
from src.vectorization import VectorizeLayer

"""
def load_vectorizer():
    notes_vectorizer = VectorizeLayer(special_tokens=["[UNK]",""])
    notes_vectorizer.load(str(os.path.join(VOCAB_PATH, NOTES_TOKENIZER_FILE)))
    durations_vectorizer = VectorizeLayer(special_tokens=["[UNK]",""])
    durations_vectorizer.load(str(os.path.join(VOCAB_PATH, DURATIONS_TOKENIZER_FILE)))
    return notes_vectorizer, durations_vectorizer
"""


def get_dataloader(notes_tokenized, durations_tokenised):
    # Create Custom Dataset
    dataset = MusicDataset(notes_tokenized, durations_tokenised, SEQ_LEN)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def build_model(notes_vectorizer, durations_vectorizer):
    # Build the model
    model = MusicModel(notes_vocab_size=notes_vectorizer.vocab_size,
                       durations_vocab_size=durations_vectorizer.vocab_size,
                       embed_dim=EMBEDDING_DIM,
                       num_heads=N_HEADS,
                       key_dim=KEY_DIM,
                       feed_forward_dim=FEED_FORWARD_DIM,
                       dropout_rate=DROPOUT_RATE)
    return model

def build_trainer(model, dataloader, notes_vectorizer, durations_vectorizer):
    # Define Loss
    loss_fn = nn.CrossEntropyLoss()
    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    # Create the trainer
    trainer = MusicTrainer(model=model,
                           train_loader=dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           device=DEVICE,
                           note_vocab_size=notes_vectorizer.vocab_size,
                           duration_vocab_size=durations_vectorizer.vocab_size,
                           save_freq=TRAINING_SAVING_FREQ,
                           checkpoint_dir=CHECKPOINT_DIR,
                           remove_previous_models=True)
    return trainer



def train():
    sequence_notes, sequence_durations = load_parsed_files(PARSED_SEQUENCES_DATA_PATH)

    # Create and adapt the note vectorizer
    notes_vectorizer = VectorizeLayer(special_tokens=["[UNK]", ""])
    notes_vectorizer.adapt(sequence_notes)

    # Create and adapt the duration vectorizer
    durations_vectorizer = VectorizeLayer(special_tokens=["[UNK]", ""])
    durations_vectorizer.adapt(sequence_durations)

    # Tokenize the notes and durations
    notes_tokenized = notes_vectorizer(sequence_notes)
    durations_tokenised = durations_vectorizer(sequence_durations)

    # Create a dataloader
    dataloader = get_dataloader(notes_tokenized, durations_tokenised)

    # Get the model
    model = build_model(notes_vectorizer, durations_vectorizer)

    # Launch the training
    trainer = build_trainer(model, dataloader, notes_vectorizer, durations_vectorizer)
    trainer.train(5000)