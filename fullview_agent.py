import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from dataset import trainloader, testloader
from utils import make_glimpse_grid, tiled_centers

# -------- model --------
class fullglimpseAgent(nn.Module):
    def __init__(self, encoder, memory, classifier, gate, image_size=28, patch_size=14, stride=None, n_classes=10, embd_dim=256, in_ch=1, device=None):
        super().__init__()
        self.device = device
        self.S = image_size
        self.g = patch_size
        self.stride = stride if stride is not None else patch_size  # non-overlap default

        self.encoder = encoder.to(device).float()
        self.rnn = memory.to(device).float()
        self.classifier = classifier.to(device).float()
        self.gate = gate.to(device).float()

        # start near average pooling (stability)
        nn.init.zeros_(self.gate.gate[-1].weight)
        nn.init.zeros_(self.gate.gate[-1].bias)

        self.attn_tau = 0.5  # temperature; 0.3–1.0 works
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def _retina(self, x, centers):
        """
        x: [B,C,S,S], centers: [T,2] in [-1,1]
        return: [B,T,C*len(scales), g, g]
        """
        B, C, S, _ = x.shape
        T = centers.size(0)
        patches = []
    
        g_s = min(int(round(self.g)), S)
        grid = make_glimpse_grid(
            centers.unsqueeze(0).expand(B, -1, -1).reshape(-1, 2),
            g_s, S
        )                                                      # [B*T,g_s,g_s,2]
        x_rep = x.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B*T, C, S, S)
        p = F.grid_sample(x_rep, grid, align_corners=True)     # [B*T,C,g_s,g_s]
        if g_s != self.g:
            p = F.interpolate(p, size=(self.g, self.g), mode='bilinear', align_corners=True)
        patches.append(p.reshape(B, T, C, self.g, self.g))

        return torch.cat(patches, dim=2)  # [B,T,C*,g,g]

    def forward(self, x):
        """
        x: [B,C,S,S]
        returns: logits [B,n_classes], attn weights [B,T,1]
        """
        B, C, S, _ = x.shape
        centers = tiled_centers(self.S, self.g, self.stride, x.device, x.dtype)  # [T,2]
        T = centers.size(0)

        # retina patches -> features per tile
        patches = self._retina(x, centers)                 # [B,T,C*,g,g]
        B_, T_, C_, g, _ = patches.shape
        feats = self.encoder(patches.reshape(B*T, C_, g, g)).view(B, T, -1)    # [B,T,D]


        # LSTM over time
        rnn_out, _ = self.rnn(feats)                       # [B,T,D]
        if len(rnn_out.shape) < 3:
            rnn_out = rnn_out.unsqueeze(0)
        # learned temporal gate (soft attention) over LSTM outputs
        scores = self.gate(rnn_out)                        # [B,T,1]
        alpha  = torch.softmax(scores / self.attn_tau, dim=1)  # [B,T,1]

        pooled = (rnn_out * alpha).sum(dim=1)              # [B,D]
        logits = self.classifier(pooled)                   # [B,n_classes]
        return logits, alpha
    
    def train_agent(self, epochs, trainloader, testloader=None):
        self.train()
        for _ in range(epochs):
            for imgs, targets in trainloader:
                if self.device is not None:
                    imgs, targets = imgs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                logits, _ = self.forward(imgs)
                loss = self.criterion(logits, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
            if testloader is not None:
                self.eval_agent(testloader)

    def eval_agent(self, testloader):
        self.eval()

        total = 0
        correct = 0
        loss_sum = 0.0

        with torch.no_grad():
            for x, y in testloader:
                if self.device is not None:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                out = self.forward(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out  

                if self.criterion is not None:
                    loss_sum += self.criterion(logits, y).item() * y.size(0)

                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total   += y.size(0)

        acc = correct / max(total, 1)
        if self.criterion is not None and total > 0:
            avg_loss = loss_sum / total
            print(f"Test accuracy: {acc:.4f} | Test loss: {avg_loss:.4f} | N={total}")
            return acc, avg_loss
        else:
            print(f"Test accuracy: {acc:.4f} | N={total}")
            return acc, None
        
    def plot(self, x):
        if self.device is not None:
            x.to(self.device, non_blocking=True)
        _, alpha = self.forward(x)
        h = w = alpha[0].shape[0] // sqrt(alpha[0].shape[0])
        alpha = alpha[0].reshape(1, h, w)
        
        import matplotlib.pyplot as plt
        plt.imshow(alpha.squeeze().detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Attention Weights')
        plt.show()



# agent = fullglimpseAgent(image_size=28, patch_size=6, stride=9,
#                          n_classes=10, embd_dim=256, in_ch=1, device='mps')

# agent.train_agent(epochs=10, trainloader=trainloader, testloader=testloader)