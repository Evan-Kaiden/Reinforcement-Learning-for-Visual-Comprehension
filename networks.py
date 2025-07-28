import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_glimpse_grid

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
    def __init__(self, in_channels, input_shape, embd_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3,3), stride=1)
        shape = input_shape - 3 + 1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1)
        shape = shape - 3 + 1

        self.confidence_head = nn.Linear(shape * shape * 128, 1)
        self.encoding_head = nn.Linear(shape * shape * 128, embd_dim)

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
        self.rnn = nn.LSTM(input_size=embd_dim, hidden_size=hidden_dim, num_layers=1, batch_first=False)

    def forward(self, current_context, prev_state):
        """
        current_context: [1, B, embd_D] -- 1 timestep input
        prev_state: tuple (h, c)
         - h: [1, B, hidden_dim]
         - c: [1, B, hidden_dim]
        Returns:
        - output: [1, B, hidden_dim]
        - next_state: (h, c)
        """
        output, next_state = self.rnn(current_context, prev_state)
        return output.squeeze(0), next_state  


def policytest():
    embd_dim = 256
    batch_size = 1

    policy = PerceptionPolicy(embd_dim)

    state = torch.randn((batch_size, embd_dim))

    location, stop_prob = policy(state)

    print("location shape:", location.shape)
    print("stop prob shape:", stop_prob.shape)

def perceptiontest():
    in_channels = 3
    input_shape = 12
    image_size = 64
    embd_dim = 256

    perception_module = PerceptionEncoder(in_channels, input_shape, embd_dim)

    image = torch.randn((1, in_channels, image_size, image_size))
    center = torch.tensor([[0.0, 0.0]])
    grid = make_glimpse_grid(center,input_shape, 64)
    img_patch = torch.nn.functional.grid_sample(image, grid, align_corners=True)

    confidence, encoding = perception_module(img_patch)

    print("Patch shape:", img_patch.shape)
    print("Encoding shape:", encoding.shape)
    print("Confidence shape:", confidence.shape)

def memorytest():
    embd_dim = 256
    batch_size = 1

    memory_module = ContextMemory(embd_dim=embd_dim, hidden_dim=embd_dim)

    # Simulate one timestep of input: [seq_len=1, B, D]
    current_context = torch.randn((1, batch_size, embd_dim))

    # Initialize LSTM state (h, c): both [1, B, H]
    h0 = torch.zeros(1, batch_size, embd_dim)
    c0 = torch.zeros(1, batch_size, embd_dim)
    prev_state = (h0, c0)

    out, (current_context, cell_state) = memory_module(current_context, prev_state)

    print("output shape:", out.shape)               # [B, hidden_dim]
    print("h shape:", current_context.shape)        # [1, B, hidden_dim]
    print("c shape:", cell_state.shape)             # [1, B, hidden_dim]

policytest()
perceptiontest()
memorytest()

