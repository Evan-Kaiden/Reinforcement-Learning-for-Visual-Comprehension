import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
import os

class preTrainer(nn.Module):
    def __init__(self, encoder, classifier, gate, seq_summarizer, lr, patch_size, stride, in_channels, img_size, device='mps'):
        super().__init__()

        self.device = device

        # Action Space Sampling Parameters -----------------------
        self.in_channels = in_channels
        self.S = img_size
        self.p = patch_size
        self.stride = stride
        self.attn_tau = 0.5
        
        # Models -------------------------------------------------
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.gate = gate.to(device)
        self.seq_summarizer = seq_summarizer.to(device)


        # Optimisers ----------------------------------------------
        self.optimizer = optim.Adam(chain(self.encoder.parameters(),
                                          self.classifier.parameters(),
                                          self.gate.parameters(),
                                          self.seq_summarizer.parameters()),
                                        lr=lr,
                                        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Generate A Constant Sampling Spcae ------------------------
        self._create_centers(self.S, self.p, self.stride)

        # Track For Model Check Points ------------------------------
        self.best_acc = 0.0

    def _create_centers(self, img_size, patch_size, stride):
        """creates a matrix x,y values normalized to range [-1, 1]"""
        coords = torch.arange(patch_size // 2,
                          img_size - patch_size // 2 + 1,
                          stride, dtype=torch.float32)

        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        norm_x = 2.0 * xx / (img_size - 1) - 1.0
        norm_y = 2.0 * yy / (img_size - 1) - 1.0

        self.centers = torch.stack((norm_x, norm_y), dim=-1).to(self.device)

    def get_patches(self, x):
        """Return all patches of an image given a patch size and a stride"""
        B = x.size(0)
        patches = F.unfold(x, kernel_size=self.p, stride=self.stride).permute(0, 2, 1).reshape(-1, self.in_channels, self.p, self.p)
        T = patches.size(0) // B

        return patches.contiguous(), T
    
    def forward(self, x):
        B = x.size(0)

        # Get Patches and Centers -----------------------------------------
        patches, T = self.get_patches(x)
        centers = self.centers.reshape(-1, 2)
        centers = centers.repeat(B, 1)
        
        # Encoder Output --------------------------------------------------
        seq_input = self.encoder(patches, centers)              #  [B*T, D]
        seq_input = seq_input.view(B, T, -1)                    # [B, T, D]

        # RNN Output ------------------------------------------------------
        seq_summarizer_out, _ = self.seq_summarizer(seq_input)    # [B,T,D]

        # Attention -------------------------------------------------------
        scores = self.gate(seq_summarizer_out)                    # [B,T,1]
        alpha = torch.softmax(scores / self.attn_tau, dim=1)
        pooled = (seq_summarizer_out * alpha).sum(dim=1)           #  [B,D]

        # Class Prediction ------------------------------------------------
        logits = self.classifier(pooled)                    # [B,n_classes]

        return logits
    
    def train_models(self, epochs, trainloader, testloader=None):
        self.train()

        # Learning Rate Scaler -----------------------------------------------------
        lr_schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        prev_save_epoch = 0
        for epoch in range(epochs):
            total_classification_loss, total = 0.0, 0.0
            
            for imgs, targets in trainloader:
                imgs, targets = imgs.to(self.device), targets.to(self.device)

                logits = self.forward(imgs)

                # Supervised loss --------------------------------------------------
                cls_loss = self.criterion(logits, targets)
                total_classification_loss += cls_loss.item()

                # Back‑prop --------------------------------------------------------
                self.optimizer.zero_grad()
                cls_loss.backward()
                self.optimizer.step()

                total += 1
                
            lr_schedule.step()
               
            print(f'Epoch {epoch + 1} |  Classification Loss : {total_classification_loss / total}')

            if testloader is not None and (epoch + 1) % 5 == 0:
                acc = self.test(testloader)
                self.train()
                if acc > self.best_acc:
                    prev_save_epoch = epoch + 1
                    self.best_acc = acc
                    self.save_models('pretrainer_ckpt/')
                    print("Saving Checkpoint...")

        if prev_save_epoch < epochs:
            acc = self.test(testloader)
            if acc > self.best_acc:
                    prev_save_epoch = epoch + 1
                    self.best_acc = acc
                    self.save_models('pretrainer_ckpt/')
                    print("Saving Checkpoint...")
        
                
    @torch.no_grad()
    def test(self, testloader):
        self.eval()
        correct, total_classification_loss, total = 0.0, 0.0, 0.0
        for imgs, targets in testloader:
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            
            logits = self.forward(imgs)
            
            # Accumulate -------------------------------------------------------
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

            # Supervised loss --------------------------------------------------
            cls_loss = self.criterion(logits, targets)
            total_classification_loss += cls_loss.item()
        
        print(f'Accuracy {correct / total} |  Classification Loss : {total_classification_loss / total}')

        return correct / total

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        torch.save(self.encoder.state_dict(), os.path.join(save_dir, 'encoder.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, 'classifier.pth'))
        torch.save(self.gate.state_dict(), os.path.join(save_dir, 'gate.pth'))
        torch.save(self.seq_summarizer.state_dict(), os.path.join(save_dir, 'seq_summarizer.pth'))
    


    def get_models(self, from_saved=True, save_dir=None):
        models = {"encoder.pth" : self.encoder, "classifier.pth" : self.classifier, "gate.pth" : self.gate, "seq_summarizer.pth" : self.seq_summarizer}
        if from_saved and save_dir is not None:
            assert os.path.exists(save_dir), "Invalid Directory"
            for path, model in models.items():
                if not os.path.exists(os.path.join(save_dir, path)):
                    print(f"Model {path} does not exist using current model")
                else:
                    state_dict = torch.load(os.path.join(save_dir, path), map_location=self.device)
                    model.load_state_dict(state_dict)
        
        return self.encoder, self.classifier, self.gate, self.seq_summarizer

