import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

import os
from itertools import chain

from utils import make_glimpse_grid, entropy_weight_t
from plotter import plot_centers, plot_attentions, make_gif

# -----------------------------------------------------------------------------
#  Glimpse‑based agent with REINFORCE + entropy bonus + baseline
# -----------------------------------------------------------------------------

class GlimpseAgent(nn.Module):
    def __init__(self, policy, encoder, classifier, gate, seq_summarizer, context_memory, 
                 action_space, stride, init_entropy_weight = 1e-1, gamma = 0.96, image_size = 28, 
                 patch_size = 14, embd_dim = 128, device = None):
        super().__init__()

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Action Space Sampling Parameters ---------------------------------------
        self.S = image_size   # input image resolution (assume square)
        self.p = patch_size   # size of extracted glimpse
        self.action_space = action_space
        self.stride = stride
        self.first = patch_size // 2          
        self.last = image_size - patch_size//2 

        # Models -----------------------------------------------------------------
        self.encoder = encoder.to(self.device)
        self.gate = gate.to(self.device)
        self.classifier = classifier.to(self.device)
        self.policy = policy.to(self.device)
        self.memory = context_memory.to(self.device)
        self.rnn = seq_summarizer.to(self.device)
        #  value head
        self.value_head = nn.Sequential(nn.Linear(embd_dim, 64),
                                        nn.ReLU(), 
                                        nn.Linear(64, 1)).to(self.device)

        #  Hyper‑params ----------------------------------------------------------
        self.attn_tau = 0.5                  # temperature for soft attention
        self.gamma = gamma
        self.entropy_weight = init_entropy_weight  # encourage exploration
        self.init_entropy = init_entropy_weight

        # Optimisers --------------------------------------------------------------
        self.classification_criterion = nn.CrossEntropyLoss()
        self.classification_optimizer = torch.optim.Adam(
            chain(
                   self.encoder.parameters(),
                   self.classifier.parameters(),
                   #self.gate.parameters(),
                   self.rnn.parameters(),
                   #self.memory.parameters(),
                ),
                lr=1e-4
               )  
        
        self.reinforce_optimizer = torch.optim.Adam(chain(self.policy.parameters(), self.memory.parameters()), lr=3e-4)
        self.value_head_optimizer = torch.optim.Adam(self.value_head.parameters(), lr=1e-3)

        # Track For Model Checkpoints -----------------------------------------------
        self.best_acc = 0.0

    def _retina_step(self, x, center):
        """Extract pxp patch around center (in [-1,1] coords)"""
        _, _, S, _ = x.shape
        grid = make_glimpse_grid(center, min(int(round(self.p)), S), S)
        patch = F.grid_sample(x, grid, align_corners=True)
        if patch.size(-1) != self.p:
            patch = F.interpolate(patch, size=(self.p, self.p), mode="bilinear", align_corners=True)
        return patch

    def _idx_to_coord(self, idx):
        """Map discrete action index to (x,y) in [-1,1]"""
        row = idx // self.action_space       
        col = idx %  self.action_space     

        patch_x = self.first + col * self.stride      
        patch_y = self.first + row * self.stride

        # normalise to [-1,1]
        x_c = 2.0 * patch_x / (self.S - 1) - 1.0
        y_c = 2.0 * patch_y / (self.S - 1) - 1.0

        return torch.stack([x_c, y_c], dim=-1)

    @staticmethod
    def _discounted(rewards, gamma):
        """Calculation of discounted returns"""
        R = torch.zeros_like(rewards[-1])
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.stack(returns)  # [T,B]
    
    @staticmethod
    def _confidence_reward(logits_t0, logits_t1, targets):
        """Per step reward based on confidence improvement"""
        ce0 = F.cross_entropy(logits_t0, targets, reduction='none')
        ce1 = F.cross_entropy(logits_t1, targets, reduction='none')
        return ce0 - ce1

    def _forward_seq(self, seq_feats):
        """forward pass on a sequence of features"""
        
        # Pool sequence with LSTM + attention gate --------------------------------
        seq_input = torch.cat(seq_feats, dim=1)
        rnn_out, _ = self.rnn(seq_input)                            # [B,T,D]

        scores = self.gate(rnn_out)                             # [B,T,1]
        alpha = torch.softmax(scores / self.attn_tau, dim=1)
        pooled = (rnn_out * alpha).sum(dim=1)                   # [B,D]
 
        logits = self.classifier(pooled)                        # [B,n_classes]
        return logits

    # ---------------------------------------------------------------------
    #  Forward rollout
    # ---------------------------------------------------------------------

    def forward(self, x, targets=None, max_steps=8):
        """
        Returns
        -------
        logits :    [B, n_classes]
        logps  :    [T,B]   combined log-prob per step 
        advantages: [T, B]  normalized advantages per step
        returns:    [T,B]   discounted rewards
        entropies:  [T,B]   step-wise entropy 
        values :    [T,B]   value baseline per step
        """
        B, _, _, _ = x.shape
        device = x.device

        # episode state
        n = self.action_space
        A = n * n               # number of locations

        # start at the grid center 
        center_idx = (n // 2) * n + (n // 2)
        prev_idx = x.new_full((B,), center_idx, dtype=torch.long, device=device)
        prev_loc = self._idx_to_coord(prev_idx)

        # memory state
        h_t = x.new_zeros(B, self.memory.hidden_size)
        c_t = x.new_zeros(B, self.memory.hidden_size)
        prev_ctx = h_t

        # visited mask 
        visited = torch.zeros(B, A, device=device, dtype=torch.bool)
        upd0 = F.one_hot(prev_idx, num_classes=A).to(device=device, dtype=torch.bool)
        visited |= upd0

        # logs
        logps, entropies, rewards, seq_feats, values = [], [], [], [], []

        prev_logits = None

        for t in range(max_steps):
            # read current patch, update memory
            patch = self._retina_step(x, prev_loc)
            feat_t = self.encoder(patch, prev_loc).view(B, -1)
            h_t, c_t = self.memory((feat_t, (h_t, c_t)))
            prev_ctx = h_t

            # value baseline for variance reduction
            baseline_t = self.value_head(h_t.detach()).squeeze(-1)

            values.append(baseline_t)
            seq_feats.append(feat_t.unsqueeze(1))

            # step reward delta CE using classification head on accumulated features
            if targets is not None:
                with torch.no_grad():
                    logits_t = self._forward_seq(seq_feats)  # [B, n_classes]
                    if prev_logits is None:
                        # first step: no cost
                        step_reward = torch.zeros(B, dtype=torch.float32, device=device)
                    else:
                        delta = self._confidence_reward(logits_t, prev_logits, targets)
                        step_reward = delta.clamp(-0.1, 0.1)
                    rewards.append(step_reward)
                    prev_logits = logits_t.detach()

            # pick next location from updated state 
            action_logits = self.policy(prev_ctx, prev_loc)

            # per-step masked logits
            step_mask = visited.detach().clone()
            masked_logits = action_logits.masked_fill(step_mask, -1e9)
            act_dist = Categorical(logits=masked_logits)
            idx_next = act_dist.sample()

            # log-prob and entropy
            logp_t = act_dist.log_prob(idx_next)
            entr_t = act_dist.entropy()
            logps.append(logp_t)
            entropies.append(entr_t)

            # update visited with the chosen next index
            upd = F.one_hot(idx_next, num_classes=A).to(device=device, dtype=torch.bool)
            visited |= upd

            # advance to next step
            prev_idx = idx_next
            prev_loc = self._idx_to_coord(prev_idx)

        # final class prediction & terminal reward
        logits = self._forward_seq(seq_feats)  # [B,n_classes]

        # ---------- Update Final Reward ----------
        if targets is not None:
            preds = logits.argmax(dim=1)
            final_r = torch.where(
                preds == targets,
                torch.tensor( 2.0, device=device),
                torch.tensor(-2.0, device=device),
            )
            rewards[-1] = rewards[-1] + final_r   # add to last step


        # ---------- stack & advantages ----------
        values = torch.stack(values)                    # [T,B]
        logps = torch.stack(logps)                     # [T,B]
        entropies = torch.stack(entropies)                 # [T,B]
        returns = self._discounted(rewards, self.gamma)  # [T,B]

        advantages = (returns - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8))

        return logits, logps, advantages, returns, values, entropies
        
        
    @torch.no_grad()
    def greedy_forward(self, x, steps=8):
        """
        Returns
        -------
        logits : [B, n_classes]
        centers: list of coords        (when B==1) for visualisation
        patches: list of patches       (when B==1) for visualisation
        dists:   list of distributions (when B==1) for visualisation
        """
        B, _, _, _ = x.shape
        device = x.device

        n = self.action_space
        A = n * n

        # ---------- Initialize Episoide Parameters ----------
        center_idx = (n // 2) * n + (n // 2)    # start at grid center
        prev_idx = x.new_full((B,), center_idx, dtype=torch.long, device=device)
        prev_loc = self._idx_to_coord(prev_idx)

        h_t = x.new_zeros(B, self.memory.hidden_size)
        c_t = x.new_zeros(B, self.memory.hidden_size)
        prev_ctx = h_t

        visited = torch.zeros(B, A, device=device, dtype=torch.bool)
        upd0 = F.one_hot(prev_idx, num_classes=A).to(device=device, dtype=torch.bool)
        visited |= upd0

        # logs
        seq_feats, centers, patches, dists = [], [], [], []

        for t in range(steps):
            # sense
            patch = self._retina_step(x, prev_loc)
            feat_t = self.encoder(patch, prev_loc).view(B, -1)
        
            if B == 1:
                centers.append(prev_loc[0])
                patches.append(patch)

            h_t, c_t = self.memory((feat_t, (h_t, c_t)))
            prev_ctx = h_t
            seq_feats.append(feat_t.unsqueeze(1))

            # ---------- Mask for Visited & Take Greedy Action ----------
            action_logits = self.policy(prev_ctx, prev_loc)

            masked_logits = action_logits.masked_fill(visited, -1e9)
            all_done = visited.all(dim=1)
            masked_logits = torch.where(all_done.unsqueeze(1),
                                        torch.zeros_like(masked_logits),
                                        masked_logits)

            idx_next = masked_logits.argmax(dim=1)

            if B == 1: dists.append(masked_logits)

            # ---------- Update Visited Mask ----------
            upd = F.one_hot(idx_next, num_classes=A).to(device=device, dtype=torch.bool)
            visited |= upd

            # Advance
            prev_idx = idx_next
            prev_loc = self._idx_to_coord(prev_idx)

        logits = self._forward_seq(seq_feats)
        return logits, centers, patches, dists
        

    # ------------------------------------------------------------------
    #  Training helpers
    # ------------------------------------------------------------------

    def train_agent(self, epochs, trainloader, testloader=None, steps=6):
        self.train()
        for epoch in range(epochs):
            
            # ---------- Entropy schedule ----------
            self.entropy_weight = max(0.01, entropy_weight_t(epoch, epochs, self.init_entropy))

            total_policy_loss, total_value_loss, total_classification_loss, total_returns, total = 0.0, 0.0, 0.0, 0.0, 0.0

            for imgs, targets in trainloader:
                
                total += 1
                imgs, targets = imgs.to(self.device), targets.to(self.device)

                logits, logps, advantages, returns, values, entropies = self.forward(imgs, targets, max_steps=steps)

                total_returns += returns[0].mean().item()

                # ---------- Supervised loss ----------
                cls_loss = self.classification_criterion(logits, targets)
                total_classification_loss += cls_loss.item()

                # ---------- Policy loss ----------
                policy_loss = -(logps * advantages.detach()).mean()
                entropy_loss = -(self.entropy_weight * entropies.mean())
                rl_loss = policy_loss + entropy_loss
                total_policy_loss += rl_loss.item()

                # ---------- Value loss ----------
                value_loss = 0.5 * (values - returns.detach()).pow(2).mean()
                total_value_loss += value_loss.item()

                # ---------- Back‑prop RL branch ----------
                self.value_head_optimizer.zero_grad()
                self.reinforce_optimizer.zero_grad()
                rl_loss.backward(retain_graph=True)
                value_loss.backward()

                # Clip Gradients For Stability
                torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.reinforce_optimizer.step()
                self.value_head_optimizer.step()

                # ---------- Back‑prop supervised branch ----------
                self.classification_optimizer.zero_grad()
                cls_loss.backward()
                self.classification_optimizer.step()
               
            print(f'Epoch {epoch + 1} | Policy Loss : {total_policy_loss / total} | Classification Loss : {total_classification_loss / total} | Value Loss : {total_value_loss / total} | Max Steps : {steps} | Avg Reward : {total_returns / total} | Entropy Weight {self.entropy_weight}')

            if testloader is not None:
                acc, _ = self.eval_agent(testloader, max_steps=steps)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_models("agent_ckpt")
                    print("Saving Checkpoint...")
                # Visualize 10 Examples Every 10 epochs
                if (epoch + 1) % 10 == 0:
                    for i in range(10):
                        self.viz_glimpses(next(iter(trainloader))[0][2:3, :], epoch=epoch + 1, idx=i, max_steps=steps)
                
                self.train()
                
    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_agent(self, testloader, max_steps):
        self.eval()
        total, correct, loss_sum = 0, 0, 0.0
        
        # ---------- Eval Taking Greedy Actions ----------
        for imgs, targets in testloader:
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            logits, _, _, _ = self.greedy_forward(imgs, steps=max_steps)
            loss_sum += self.classification_criterion(logits, targets).item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

        acc = correct / max(total, 1)
        avg_loss = loss_sum / max(total, 1)
        
        print(f'Test Accuracy : {acc} | Test Loss : {avg_loss}\n')
        return acc, avg_loss
    
    @torch.no_grad()
    def viz_glimpses(self, x, epoch, idx, max_steps=6):
        """Save Visuals of Agents Vision Path and Action Probabilities"""
        self.eval()
        x = x.to(self.device)
        _, centers, _, dists = self.greedy_forward(x, steps=max_steps)

        plot_centers(x.squeeze(0).detach().cpu(), epoch, idx, centers)
        plot_attentions(dists, epoch, idx, self.action_space)
    
    @torch.no_grad()
    def make_viz(self, x, max_steps=6, filepath=None):
        self.eval()
        x = x.to(self.device)
        _, centers, _, _ = self.greedy_forward(x, steps=max_steps)

        if filepath is not None:
            make_gif(x.squeeze(0).detach().cpu(), centers, self.p, filepath)
        else:
            make_gif(x.squeeze(0).detach().cpu(), centers, self.p)

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        torch.save(self.encoder.state_dict(), os.path.join(save_dir, 'encoder.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, 'classifier.pth'))
        torch.save(self.policy.state_dict(), os.path.join(save_dir, 'policy.pth'))
        torch.save(self.memory.state_dict(), os.path.join(save_dir, 'memory.pth'))
        torch.save(self.rnn.state_dict(), os.path.join(save_dir, 'rnn.pth'))
        torch.save(self.value_head.state_dict(), os.path.join(save_dir, 'value_head.pth'))
    
    def set_models(self, save_dir=None):
        models = {
            "encoder.pth" : self.encoder, 
            "classifier.pth" : self.classifier, 
            "gate.pth" : self.gate, 
            "rnn.pth" : self.rnn,
            "memory.pth" : self.memory,
            "value_head.pth" : self.value_head,
            "policy.pth" : self.policy
            }
        
        if save_dir is not None:
            assert os.path.exists(save_dir), "Directory does not exist"
            for path, model in models.items():
                if not os.path.exists(os.path.join(save_dir, path)):
                    print(f"Model {path} does not exist using current model")
                else:
                    state_dict = torch.load(os.path.join(save_dir, path), map_location=self.device)
                    model.load_state_dict(state_dict)
