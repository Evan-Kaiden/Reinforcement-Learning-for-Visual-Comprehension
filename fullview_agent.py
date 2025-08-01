import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from dataset import trainloader, testloader
from utils import make_glimpse_grid, tiled_centers

from tqdm import tqdm

# -------- model --------
class fullglimpseAgent(nn.Module):
    def __init__(self, encoder, memory, classifier, gate, image_size=28, patch_size=14, stride=None, n_classes=10, embd_dim=256, in_ch=1, device=None):
        super().__init__()
        self.device = device
        self.S = image_size
        self.p = patch_size
        self.stride = stride if stride is not None else patch_size  # non-overlap default

        self.encoder = encoder.to(device).float()
        self.rnn = nn.LSTMCell(embd_dim, embd_dim).to(device).float()
        self.classifier = classifier.to(device).float()
        self.gate = gate.to(device).float()

        # start near average pooling (stability)
        nn.init.zeros_(self.gate.gate[-1].weight)
        nn.init.zeros_(self.gate.gate[-1].bias)

        self.attn_tau = 0.5 
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def _retina_step(self, x, center):
        """
        x : [B,C,S,S]      center : [B,2] tensor in [-1,1]
        returns one patch : [B,C,g,g]
        """
        _, _, S, _ = x.shape
        grid = make_glimpse_grid(center, min(int(round(self.p)), S), S)  # [B,g',g',2]
        patch = F.grid_sample(x, grid, align_corners=True)               # [B,C,g',g']
        if patch.size(-1) != self.p:
            patch = F.interpolate(patch, size=(self.p, self.p),
                                mode='bilinear', align_corners=True)
        return patch

    def forward(self, x):
        """
        Online version: feeds one patch per step into LSTMCell.
        Returns logits [B,n_classes] and attention weights [B,T,1].
        """
        B, C, S, _ = x.shape
        device = x.device
        centers = tiled_centers(self.S, self.p, self.stride, device, x.dtype)  # [T,2]
        T = centers.size(0)

        # hidden state init (h, c)  – you can learn an init instead if you like
        h_t = x.new_zeros(B, self.rnn.hidden_size)
        c_t = x.new_zeros(B, self.rnn.hidden_size)

        # containers for all time-step outputs
        rnn_outputs = []

        for t in range(T):
            # 1) choose / (later) sample next location
            center_t = centers[t].unsqueeze(0).expand(B, -1)                   # [B,2]

            # 2) glimpse
            patch_t = self._retina_step(x, center_t)                           # [B,C,g,g]

            # 3) encode
            feat_t = self.encoder(patch_t).view(B, -1)                         # [B,D_enc]

            # 4) one RNN step
            h_t, c_t = self.rnn(feat_t, (h_t, c_t))                            # both [B,D]

            rnn_outputs.append(h_t.unsqueeze(1))       # keep time dimension

        rnn_out = torch.cat(rnn_outputs, dim=1)        # [B,T,D]

        # ---- identical to old code from here down ----
        scores = self.gate(rnn_out)                    # [B,T,1]
        alpha  = torch.softmax(scores / self.attn_tau, dim=1)
        pooled = (rnn_out * alpha).sum(dim=1)          # [B,D]
        logits = self.classifier(pooled)               # [B,n_classes]
        return logits, alpha
    
    def train_agent(self, epochs, trainloader, testloader=None):
        self.train()
        for _ in range(epochs):
            with tqdm(total=len(trainloader)) as pbar:
                for imgs, targets in trainloader:
                    if self.device is not None:
                        imgs, targets = imgs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    logits, _ = self.forward(imgs)
                    loss = self.criterion(logits, targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
        
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
        h = w = alpha[0].shape[0] // int(sqrt(alpha[0].shape[0]))
        alpha = alpha[0].reshape(1, h, w)
        
        import matplotlib.pyplot as plt
        plt.imshow(alpha.squeeze().detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Attention Weights')
        plt.show()
