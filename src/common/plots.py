import matplotlib.pyplot as plt
import numpy as np
from common.training import getMSEScore, getR2Score

def printScores(dl_test, dl_val, model):
    test_score = getR2Score(dl_test, model)
    val_score = getR2Score(dl_val, model)
    print(f"test R2: {test_score:.4f}, val R2: {val_score:.4f}")

def printScaledScores(dl_test, dl_val, model):
    test_score, _, _ = getMSEScore(dl_test, model, model.scaler)
    test_r2score = getR2Score(dl_test, model)
    val_score, _, _ = getMSEScore(dl_val, model, model.scaler)
    val_r2score = getR2Score(dl_val, model)
    print(f"test MSE: {test_score:.4f} | R2: {test_r2score:.4f}, val MSE: {val_score:.4f} | R2: {val_r2score:.4f}")

def plotTimeseries(y_true_test, y_pred_test, y_true_val, y_pred_val):
    y_true = np.concatenate((y_true_test, y_true_val), axis=0)
    indices = np.arange(len(y_true))
    indices_test = np.arange(len(y_pred_test))
    indices_val = np.arange(len(y_pred_val)) + len(y_pred_test)

    fig, ax = plt.subplots(figsize=(64, 24))
    ax.plot(indices_test, y_pred_test, color='orange')
    ax.plot(indices_val, y_pred_val, color='red')
    ax.plot(indices, y_true, color='blue')

