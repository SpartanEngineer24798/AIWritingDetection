import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def read_data(input_folder):
    data_dict = {}
    for key in os.listdir(input_folder):
        path = os.path.join(input_folder, key)
        data = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data.append(json.load(f))
                    except json.JSONDecodeError:
                        f.seek(0)  # Reset the file cursor
                        raw_content = f.read()
                        data.append(raw_content)
            except IOError as e:
                print(f"Error reading file {file_path}: {e}")
        data_dict[key] = data
    
    return data_dict


def process_dictionary(dictionary):
    keys = list(dictionary.keys())
    features = []
    labels = []

    for key in keys:
        elements = dictionary[key]
        features.extend(elements)
        if key == "human":
            labels.extend([1] * len(elements))
        else:
            labels.extend([0] * len(elements))

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


def train_model(model, features, labels, epochs, batch_size, val_split, alpha, lr, patience):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=val_split, random_state=42)
    print(X_train)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)

    best_val_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(X_train), batch_size):
            inputs = torch.from_numpy(X_train[i:i + batch_size]).float()
            targets = torch.from_numpy(y_train[i:i + batch_size]).float()
            targets = targets.unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)

            loss += alpha * l2_reg

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.round(outputs)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(X_train)
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                inputs = torch.from_numpy(X_val[i:i + batch_size]).float()
                targets = torch.from_numpy(y_val[i:i + batch_size]).float()
                targets = targets.unsqueeze(1)  # Reshape labels

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                predicted = torch.round(outputs)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(X_val)
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
                  .format(epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping! No improvement in validation loss for {} epochs.".format(patience))
                break

    model.load_state_dict(best_model_weights)

    return model.state_dict(), train_losses, val_losses, train_accs, val_accs


def plot_curves(train_losses, val_losses, train_accs, val_accs, results_dir):
    epochs = len(train_losses)
    x = np.arange(1, epochs + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_losses, label='Train')
    plt.plot(x, val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, train_accs, label='Train')
    plt.plot(x, val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save plot to mlp folder
    mlp_dir = os.path.join(results_dir, 'mlp')
    os.makedirs(mlp_dir, exist_ok=True)
    plot_path = os.path.join(mlp_dir, 'plot.png')
    i = 1
    while os.path.exists(plot_path):
        plot_path = os.path.join(mlp_dir, f'plot_{i}.png')
        i += 1
    plt.savefig(plot_path)
    plt.close()


def predict(model, features, results_dir):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features).float()
        outputs = model(inputs)
        predictions = outputs.numpy()

    # Save predictions to mlp folder
    mlp_dir = os.path.join(results_dir, 'mlp')
    os.makedirs(mlp_dir, exist_ok=True)
    pred_path = os.path.join(mlp_dir, 'predictions.npy')
    i = 1
    while os.path.exists(pred_path):
        pred_path = os.path.join(mlp_dir, f'predictions_{i}.npy')
        i += 1
    np.save(pred_path, predictions)

    return predictions


def main(input_folder, results_dir, alpha, lr, patience):
    data = read_data(input_folder)
    features, labels = process_dictionary(data)

    input_size = 9
    hidden_size = 9
    model = MLP(input_size, hidden_size)

    best_checkpoint, train_losses, val_losses, train_accs, val_accs = train_model(
        model, features, labels, epochs=1000, batch_size=32, val_split=0.2, alpha=alpha, lr=lr, patience=patience)

    plot_curves(train_losses, val_losses, train_accs, val_accs, results_dir)
    predictions = predict(model, features, results_dir)
    
    mlp_dir = os.path.join(results_dir, 'mlp')
    os.makedirs(mlp_dir, exist_ok=True)
    checkpoint_path = os.path.join(mlp_dir, 'best_model_checkpoint.pth')
    torch.save(best_checkpoint, checkpoint_path)

    print("Training complete. Best model checkpoint and results saved successfully.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", "--o", required=True, help="Path to the input folder")
    parser.add_argument("--r", required=True, help="Path to the results directory")
    parser.add_argument("--alpha", type=float, default=0.01, help="L2 regularization coefficient (default: 0.01)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 0.0001)")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs with no improvement to wait before stopping (default: 5)")
    args = parser.parse_args()

    print("Beginning MLP training. Note that MLP training may require hyperparameter tuning")

    main(args.i, args.r, args.alpha, args.lr, args.patience)
