import torch
import torch.nn as nn
import torch.nn.functional as F


# Hyena Attention Layer
class Hyena(nn.Module):
    def __init__(self):
        super(Hyena, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Define the Expert class
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Define the Gating Network class
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=2)


# Define the Mixture of Experts Layer class
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)

    def forward(self, x, num_experts_per_tok):
        gating_scores = self.gate(x)
        topk_gating_scores, topk_indices = gating_scores.topk(num_experts_per_tok, dim=2, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=2)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.transpose(1, 2)
        output = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs)
        return output


# Define the overall Transformer model with integrated MoE
class TransformerWithMoE(nn.Module):
    def __init__(self, num_layers, dim, head_dim, hidden_dim, n_heads, num_experts, vocab_size, num_experts_per_tok):
        super(TransformerWithMoE, self).__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads) for _ in range(num_layers)])
        self.moe_layer = MoELayer(dim, hidden_dim, dim, num_experts)
        self.output_layer = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.moe_layer(x, self.num_experts_per_tok)
        logits = self.output_layer(x)
        return logits

