import torch

print(torch.__version__)
from MoH import TransformerWithMoE
import torch.nn as nn
import itertools

device = torch.device('cuda')


# Data Preparation
class DNADataset(torch.utils.data.Dataset):
    def __init__(self, text, k=3):
        self.k = k
        self.block_size = 1024
        self.batch_size = 32
        self.vocab = self.build_vocab(text)
        self.data = self.batchify(self.encode(text))

    def batchify(self, data):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = len(data) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data[:nbatch * self.batch_size]
        # Evenly divide the data across the bsz batches.
        data = [data[i * self.batch_size: (i + 1) * self.batch_size] for i in range(nbatch)]
        return data

    def build_vocab(self, text):
        k_mers = set()
        k_mers.update([text[i:i + self.k] for i in range(len(text) - self.k + 1)])
        return {k_mer: i for i, k_mer in enumerate(k_mers)}

    # rest of the class remains the same

    def encode(self, sequence):
        k_mers = [sequence[i:i + self.k] for i in range(len(sequence) - self.k + 1)]
        return [self.vocab[k_mer] for k_mer in k_mers]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_data = sequence[:-1]
        target_data = sequence[1:]
        return torch.tensor(input_data), torch.tensor(target_data)


# Open the file and read the sequences
with open('shakespeare_input.txt', 'r') as f:
    text = f.read()

# Create the dataset and dataloader
dataset = DNADataset(text)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Print the first batch
print("Length of the dataset:", len(dataset.data))
for batch in dataloader:
    input_data, target_data = batch
    print("Input data:", input_data)
    print("Target data:", target_data)
    break

vocab_size = len(dataset.vocab)
print("Vocab size:", vocab_size)


def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_data, target_data = batch
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        output_data = model(input_data)
        loss = loss_fn(output_data.view(-1, vocab_size), target_data.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


# Evaluation Function

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_data, target_data = batch
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            output_data = model(input_data)
            loss = loss_fn(output_data.view(-1, vocab_size), target_data.view(-1))
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


num_epochs = 10
# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Initialize the model with configurations matching Mixtral 8x7B
model = TransformerWithMoE(
    num_layers=16,  # Decreased from 32
    dim=16,  # Decreased from 28
    head_dim=32,  # Decreased from 128
    hidden_dim=32,  # Decreased from 28
    n_heads=4,  # Decreased from 32
    num_experts=4,  # Decreased from 8
    vocab_size=vocab_size,
    num_experts_per_tok=1  # Decreased from 2
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters())

model = model.to(device)
# Training Loop
for epoch in range(num_epochs):
    train_loss = train(model, dataloader, loss_fn, optimizer)
    val_loss = evaluate(model, dataloader, loss_fn)
    print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

# Model Saving
torch.save(model.state_dict(), 'model.pth')
