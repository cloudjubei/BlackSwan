import torch
from common.models import LSTMLinear

def loadCheckpoint(file):
    return torch.load(file)

def loadLSTMLinearFromCheckpoint(checkpoint_file):
    checkpoint = loadCheckpoint(checkpoint_file)

    model = LSTMLinear(**checkpoint['model_args'])

    model.load_state_dict(checkpoint['model_state_dict'])

    return model