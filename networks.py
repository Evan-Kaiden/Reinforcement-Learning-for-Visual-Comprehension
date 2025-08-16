import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionPolicy(nn.Module):
    def __init__(self, action_space, embd_dim):
        super().__init__()
        self.action_space = action_space

        self.fc1 = nn.Linear(embd_dim + 2, 64)
        self.dist_head = nn.Linear(64, action_space)

        nn.init.zeros_(self.fc1.weight[:, -2:]) 
        nn.init.zeros_(self.fc1.bias)
    
    def forward(self, current_context, loc):
        """
        Inputs:
            current_context: [B, embd_D] -- context for current step
        Returns:
            dist: [B, Action Space]
            stop_logits: [B, 1]
        """
        x = torch.cat([current_context, loc], dim=1)
        x = F.relu(self.fc1(x))      

        logits = self.dist_head(x)
        return logits
        

class PerceptionEncoder(nn.Module):
    def __init__(self, in_channels, embd_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.location_encoder = nn.Linear(2, 128)
        self.proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, embd_dim)
        )

    def forward(self, img_patch, loc):
        """
        Inputs:
            img_patch: [B, C, H, W]
        Returns:        
            encoding: [B, embd_D]
        """
        x = self.encoder(img_patch)
        x = x.view(x.size(0), -1) 

        loc = self.location_encoder(loc) 
        x = x + loc # add patch location information

        encoding = self.proj(x)

        return encoding

class gate(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embd_dim, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, embedding):
        """
        Inputs: 
            embedding: [B, embd_dim]
        Returns:
            output: [B, 1]
        """
        return self.gate(embedding)
    
class SequenceSummarizer(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.hidden_size = embd_dim
        self.lstm = nn.LSTM(embd_dim, embd_dim, num_layers=1, batch_first=True)
        self.atn = nn.MultiheadAttention(embd_dim, num_heads=1, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(embd_dim)
    def forward(self, x):
        """
        Inputs: 
            x : [B, T, embd_D] full sequence of events 
        Returns:
            out : [B, T, embd_D] full sequence after LSTM
        """
        atn_out, _ = self.atn(x, x, x)
        x = F.relu(self.norm(atn_out + x))
        return self.lstm(x)
    

class ContextMemory(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.embd_dim = embd_dim
        self.hidden_size = embd_dim
        self.lstm = nn.LSTMCell(embd_dim, embd_dim)

    def forward(self, current_context):
        """
        Inputs:
            current_context: [B, T, embd_D] -- 1 timestep input
            prev_state: tuple (h, c), T = 1
                h: [B, T, embd_dim]
                c: [B, T, embd_dim]
        Returns:
            output: [B, T, embd_dim]
            next_state: (h, c)
        """
        return self.lstm(*current_context)

class Classifier(nn.Module):
    def __init__(self, embd_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embd_dim, n_classes)

    def forward(self, embeding):
        """
        Inputs:
            embedding: [B, embd_dim]
        Returns:
            output: [B, n_classes]
        """

        return self.fc1(embeding)