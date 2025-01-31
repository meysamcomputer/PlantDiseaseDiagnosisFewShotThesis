import torch.nn as nn

#   تعریف Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=1)
        )

#     def forward(self, x):
#         # x shape: (batch_size, sequence_length, feature_dim)
#         attention_weights = self.attention(x)  # (batch_size, sequence_length, 1)
#         weighted_features = x * attention_weights  # (batch_size, sequence_length, feature_dim)
#         return weighted_features.sum(dim=1)  # (batch_size, feature_dim)
    
    def forward(self, x):
        attention_weights = self.attention(x)  # (batch_size, sequence_length, 1)
        weighted_features = x * attention_weights  # (batch_size, sequence_length, feature_dim)
        return weighted_features.sum(dim=1), attention_weights  # 