import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionPolicy(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.fc1 = nn.Linear(embd_dim, 16)
        self.loc_mean_head = nn.Linear(16, 2)
        self.loc_log_std = nn.Parameter(torch.zeros(2))
        self.stop_head = nn.Linear(16, 1)
        self.baseline_head = nn.Linear(16, 1)
        
    
    def forward(self, current_context):
        """
        Inputs:
            current_context: [B, embd_D] -- context for current step
        Returns:
            location: [B, 2] --> [B, (x,y)]
            dist: Normal distribution
            stop_prob: [B, 1]
        """
        x = F.relu(self.fc1(current_context))      

        mean = F.tanh(self.loc_mean_head(x))
        std = torch.exp(torch.clamp(self.loc_log_std, -1.5, 0.5)).expand_as(mean)

        dist = torch.distributions.Normal(mean, std)
        raw_location = dist.rsample()
        location = torch.clamp(raw_location, -1, 1)

        stop_prob = torch.sigmoid(self.stop_head(x))
        baseline = self.baseline_head(x)

        return location, raw_location, stop_prob, dist, baseline
        

    
class PerceptionEncoder(nn.Module):
    def __init__(self, in_channels, input_shape, embd_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3),   nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> [B,64,1,1]
        )
        self.proj = nn.Linear(64, embd_dim)

    def forward(self, img_patch):
        """
        Inputs:
            img_patch: [B, C, H, W]
        Returns:        
            encoding: [B, embd_D]
        """
        x = self.encoder(img_patch)
        x = x.view(x.size(0), -1) 
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
    


class ContextMemory(nn.Module):
    def __init__(self, embd_dim, n_layers=1):
        super().__init__()
        self.embd_dim = embd_dim
        self.rnn = nn.LSTM(input_size=embd_dim, hidden_size=embd_dim, num_layers=n_layers, batch_first=True)

    def forward(self, current_context):
        """
        Inputs:
            current_context: [B, T, embd_D] -- 1 timestep input
            prev_state: tuple (h, c), T = 1
            - h: [B, T, embd_dim]
            - c: [B, T, embd_dim]
        Returns:
            - output: [B, T, embd_dim]
            - next_state: (h, c)
        """
        output, next_state = self.rnn(current_context)
        return output.squeeze(0), next_state  

class Classifier(nn.Module):
    def __init__(self, embd_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embd_dim, n_classes)
    
    def forward(self, embeding):
        """
        embedding: [B, embd_dim]
        Returns:
        - output: [B, n_classes]
        """
        return self.fc1(embeding)