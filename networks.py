import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionPolicy(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.fc1 = nn.Linear(embd_dim, 64)
        self.location_head = nn.Linear(64, 2)
        self.stop_head = nn.Linear(64, 1)
        
    
    def forward(self, current_context):
        """
        current_context: [B, embd_D] -- context for current step
        --------------------------------------------------------
        location: [B, 2] --> [B, (x,y)]
        stop_prob: [B, 1]
        """
        x = F.relu(self.fc1(current_context))               
        location = F.tanh(self.location_head(x))
        stop_prob = torch.sigmoid(self.stop_head(x))
        return location, stop_prob
    

    
class PerceptionEncoder(nn.Module):
    def __init__(self, in_channels, input_shape, embd_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3,3), stride=1)
        shape = input_shape - 3 + 1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1)
        shape = shape - 3 + 1

        self.confidence_head = nn.Linear(shape * shape * 128, 1)
        self.encoding_head = nn.Linear(shape * shape * 128, embd_size)

    def forward(self, img_patch):
        """
        img_patch: [B, C, H, W]
        --------------------------------------------------------        
        confidence: [B, 1]
        encoding: [B, embd_D]
        """
        x = F.relu(self.conv1(img_patch))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)

        confidence = torch.sigmoid(self.confidence_head(x))
        encoding = self.encoding_head(x)

        return confidence, encoding


class ContextMemory(nn.Module):
    def __init__(self, embd_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_size=embd_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
    
    def forward(self, current_context, add_state):
        """
        current_context: [B, 1, embd_D] -- current memory
        add_state: [h, c]               -- state to update memory with
         - h: [1, B, hidden_dim]
         - c: [1, B, hidden_dim]
        --------------------------------------------------------
        output: [1, B, hidden_dim]
        next_state: [h, c]
         - h: [1, B, hidden_dim]
         - c: [1, B, hidden_dim]
        """
        x = current_context.unsqueeze(0)
        output, next_state = self.rnn(x, add_state)

        return output.squeeze(0), next_state

        