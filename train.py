import os

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from config import *
from src.dataset import MusicDataset
from src.model import MusicModel
from src.trainer import MusicTrainer
from src.utils import load_parsed_files
from src.vectorization import VectorizeLayer


def load_vectorizer():
    notes_vectorizer = VectorizeLayer().load(str(os.path.join(VOCAB_PATH, NOTES_TOKENIZER_FILE)))
    durations_vectorizer = VectorizeLayer().load(str(os.path.join(VOCAB_PATH, DURATIONS_TOKENIZER_FILE)))
    return notes_vectorizer, durations_vectorizer

def get_dataloader(notes_tokenized, durations_tokenised):
    dataset = MusicDataset(notes_tokenized, durations_tokenised, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def build_model(notes_vectorizer, durations_vectorizer):
    model = MusicModel(notes_vectorizer.vocab_size, durations_vectorizer.vocab_size, EMBEDDING_DIM, N_HEADS, KEY_DIM,
                       FEED_FORWARD_DIM, DROPOUT_RATE)
    return model

def build_trainer(model, dataloader, notes_vectorizer, durations_vectorizer):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
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
    notes_vectorizer, durations_vectorizer = load_vectorizer()
    notes_tokenized = notes_vectorizer(sequence_notes)
    durations_tokenised = durations_vectorizer(sequence_durations)

    dataloader = get_dataloader(notes_tokenized, durations_tokenised)


    model = build_model(notes_vectorizer, durations_vectorizer)
    trainer = build_trainer(model, dataloader, notes_vectorizer, durations_vectorizer)
    trainer.train(5000)