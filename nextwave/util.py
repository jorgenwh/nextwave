from typing import List, Tuple
import os

import numpy as np
import librosa

TRAINING_SEGMENT_LENGHT = 200

def get_files(directory: str) -> List[str]:
    filenames = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    return filenames

def read_audio_file(filename: str) -> Tuple[np.ndarray, int]:
    audio_data, sample_rate = librosa.load(filename)
    return audio_data, sample_rate

def audio_to_training_data(
        audio_data: np.ndarray, batch_size=1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    assert isinstance(audio_data, np.ndarray)
    assert audio_data.ndim == 1
    assert audio_data.shape[0] > TRAINING_SEGMENT_LENGHT

    training_examples = []

    for i in range(audio_data.shape[0]-TRAINING_SEGMENT_LENGHT):
        x = audio_data[i:i+TRAINING_SEGMENT_LENGHT]
        y = audio_data[i+1:i+TRAINING_SEGMENT_LENGHT+1]
        training_examples.append((x, y))

    return training_examples


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{round(self.avg, 4)}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
