import torch
import pandas as pd
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import the required libraries

# Step 2: Prepare the dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.loc[index, 'text'])
        label = self.data.loc[index, 'label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float32)  # Assuming labels are in float format
        }

# Step 3: Install the required libraries
# !pip install torch
# !pip install transformers
# !pip install pandas
# !pip install scikit-learn

# Step 4: Load and split the dataset
train_df = pd.read_csv('data.csv')
val_df = pd.read_csv('validation.csv')

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Step 5: Initialize the ALBERT tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=1)

# Step 6: Create data loaders for training and validation
batch_size = 16
max_length = 128

train_dataset = CustomDataset(train_df, tokenizer, max_length)
val_dataset = CustomDataset(val_df, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 7: Train the ALBERT model and print the loss curve and best model checkpoint accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.MSELoss()

num_epochs = 1000

best_val_loss = float('inf')
best_model_path = 'best_model.pt'

train_losses = []
val_losses = []
val_accuracies = []

print("Training started...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            predictions = outputs.logits.squeeze()
            predicted_labels = torch.round(torch.sigmoid(predictions))
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    val_loss /= len(val_loader)
    accuracy = total_correct / total_samples
    val_losses.append(val_loss)
    val_accuracies.append(accuracy)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
    else:
        # Perform early stopping
        print(f'Early stopping triggered. No improvement in validation loss for {epoch} epochs.')
        break

    print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {accuracy:.4f}')

print("Training completed!")

# Plotting the loss curve
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Printing the best model checkpoint accuracy
best_model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=1)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)

best_model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = best_model(input_ids, attention_mask=attention_mask, labels=labels)

        predictions = outputs.logits.squeeze()
        predicted_labels = torch.round(torch.sigmoid(predictions))
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f'Best Model Checkpoint Accuracy: {accuracy:.4f}')
