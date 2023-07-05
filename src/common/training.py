import time
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, precision_score, recall_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
import numpy as np

def getLabelsAndPredictions(dataloader: DataLoader, model: nn.Module):
    y_true = []
    y_pred = []

    for inputs, labels in dataloader:
        # print(f'getLabelsAndPredictions inputs: {inputs.shape}')
        # print(f'getLabelsAndPredictions labels: {labels.shape}')
        outputs = model(inputs)
        # print(f'getLabelsAndPredictions outputs: {outputs.shape}')
        y_true += labels[0].tolist()
        y_pred += outputs.tolist()

    return y_true, y_pred

def getF1(dataloader: DataLoader, model: nn.Module, average: str = 'macro'):
    y_true, y_pred = getLabelsAndPredictions(dataloader, model)
    return f1_score(y_true, y_pred, average='macro')

def getMSEScore(dataloader: DataLoader, model: nn.Module, scaler: StandardScaler=None, print_metrics: bool = False):
    y_true, y_pred = getLabelsAndPredictions(dataloader, model)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # print(f'y_true: {y_true}')
    # print(f'y_pred: {y_pred}')
    score = mean_squared_error(y_true, y_pred)
    if scaler != None:
        y_true = scaler.inverse_transform(np.expand_dims(y_true, 1)).squeeze(1)
        y_pred = scaler.inverse_transform(np.expand_dims(y_pred, 1)).squeeze(1)
    if print_metrics:
        print(f'y_true: {y_true}')
        print(f'y_pred: {y_pred}')

    return np.average(score), y_true, y_pred

def getR2Score(dataloader: DataLoader, model: nn.Module):
    y_true, y_pred = getLabelsAndPredictions(dataloader, model)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = r2_score(y_true, y_pred)

    return np.average(score)

def getAccuracy(dataloader: DataLoader, model: nn.Module):
    y_true, y_pred = getLabelsAndPredictions(dataloader, model)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = (y_pred.round() == y_true) * 1.0
    return np.average(accuracy)

def fit(model: nn.Module, 
        optimizer: optim.Optimizer, 
        loss_fn, 
        train_dl: DataLoader, 
        val_dl: DataLoader, 
        epochs: int,
        patience: int = 0, 
        save_checkpoints: bool = False, checkpoint_path = './data/checkpoints/checkpoint.pkl', model_args = {}, optimizer_args = {},
        checkpoint_f1_path = './data/checkpoints/checkpoint_f1.pkl', 
        print_metrics: bool = True):
    
    infos = []
    lowest_loss = 100000000
    no_improvement = 0
    best_score_train = -100
    best_score_val = -100
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for inputs, labels in train_dl:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels[0])
            # print(f'training outputs: {outputs}')
            # print(f'training labels: {labels}')
            loss.backward()
            
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_dl)

        time_taken = time.time() - start_time

        if (patience > 0 and epoch > 0):
          if average_loss < lowest_loss:
            lowest_loss = average_loss
            no_improvement = 0

            if save_checkpoints:
                torch.save(
                  obj={
                        'epoch': epoch,
                        'loss': lowest_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_args': model_args,
                        'optimizer_args': optimizer_args
                      },
                      f=checkpoint_path
                )  
          else:
            no_improvement += 1
            
          if no_improvement == patience:
            print(f'Best loss: {lowest_loss:4f} best train: {best_score_train:.4f} best val: {best_score_val:.4f}')
            model.eval()
            return infos 
          
        if print_metrics: 
            model.eval()
            with torch.no_grad():
                train_score = getAccuracy(train_dl, model)
                val_score = getAccuracy(val_dl, model)
                if train_score > best_score_train:
                    best_score_train = train_score
                if val_score > best_score_val:
                    best_score_val = val_score
                    torch.save(
                    obj={
                            'epoch': epoch,
                            'loss': lowest_loss,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'model_args': model_args,
                            'optimizer_args': optimizer_args
                        },
                        f=checkpoint_f1_path
                    )  

                infos.append([epoch, average_loss, train_score, val_score])

                print(f"Epoch {epoch+1:4d}/{epochs} = train loss: {average_loss:.4f}, train ACC: {train_score:.4f}, val ACC: {val_score:.4f}, time_taken: {time_taken}s")
    print(f'Best loss: {lowest_loss:4f} best train: {best_score_train:.4f} best val: {best_score_val:.4f}')
    model.eval()
    return infos


def fit_scaled(model: nn.Module, 
        optimizer: optim.Optimizer, 
        loss_fn, 
        train_dl: DataLoader, 
        val_dl: DataLoader, 
        epochs: int,
        patience: int = 0, 
        save_checkpoints: bool = False, checkpoint_path = './checkpoint.pkl', model_args = {}, optimizer_args = {},
        checkpoint_f1_path = './checkpoint_f1.pkl', 
        print_metrics: bool = True):
    
    infos = []
    lowest_loss = 100000000
    no_improvement = 0
    best_score_train = 1000000000
    best_score_val = 1000000000
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for inputs, labels in train_dl:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('training inputs: %s, outputs: %s, labels: %s' % (inputs.shape, outputs.shape, labels.shape))
            loss = loss_fn(outputs, labels[0])
            # print('training loss: %s, outputs: %s, labels: %s' % (loss.item(), outputs, labels))
            loss.backward()
            
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_dl)

        time_taken = time.time() - start_time

        if (patience > 0 and epoch > 0):
          if average_loss < lowest_loss:
            lowest_loss = average_loss
            no_improvement = 0

            if save_checkpoints:
                torch.save(
                  obj={
                        'epoch': epoch,
                        'loss': lowest_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_args': model_args,
                        'optimizer_args': optimizer_args
                      },
                      f=checkpoint_path
                )  
          else:
            no_improvement += 1
            
          if no_improvement == patience:
            print(f'Best loss: {lowest_loss:4f} best train mse: {best_score_train:.4f} best val mse: {best_score_val:.4f}')
            model.eval()
            return infos 
          
        if print_metrics: 
            model.eval()
            with torch.no_grad():
                train_score, _, _ = getMSEScore(train_dl, model, model.scaler)
                train_r2score = getR2Score(train_dl, model)
                val_score, _, _ = getMSEScore(val_dl, model, model.scaler)
                val_r2score = getR2Score(val_dl, model)
                if train_score < best_score_train:
                    best_score_train = train_score
                if val_score < best_score_val:
                    best_score_val = val_score
                    torch.save(
                    obj={
                            'epoch': epoch,
                            'loss': lowest_loss,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'model_args': model_args,
                            'optimizer_args': optimizer_args
                        },
                        f=checkpoint_f1_path
                    )  

                infos.append([epoch, average_loss, train_score, train_r2score, val_score, val_r2score])

                print(f"Epoch {epoch+1:4d}/{epochs} = train loss: {average_loss:.4f}, train MSE: {train_score:.4f} | R2: {train_r2score:.4f}, val MSE: {val_score:.4f} | R2: {val_r2score:.4f}, time_taken: {time_taken}s")
    print(f'Best loss: {lowest_loss:4f} best train: {best_score_train:.4f} best val: {best_score_val:.4f}')
    model.eval()
    return infos
