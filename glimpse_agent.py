import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
from itertools import chain
from math import sqrt
from utils import make_glimpse_grid
from plotter import plot_centers

# -----------------------------------------------------------------------------
#  Glimpse‑based agent with REINFORCE + entropy bonus + baseline
# -----------------------------------------------------------------------------

class GlimpseAgent(nn.Module):
    """Recurrent glimpse policy-gradient classifier.

    Forward returns logits and all tensors needed for REINFORCE so the training
    loop can compute the losses.
    """

    def __init__(self, policy, encoder, classifier, gate, step_cost = 0.000001, gamma = 0.96, image_size = 28, patch_size = 14, embd_dim = 128, device = None):
        super().__init__()

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.S = image_size   # input image resolution (assume square)
        self.p = patch_size   # size of extracted glimpse
        self.gamma = gamma
        self.step_cost = step_cost

        #  Feature extractor, attention gate, and classifier
        self.encoder = encoder.to(self.device)
        self.gate = gate.to(self.device)
        self.classifier = classifier.to(self.device)

        #  Policy network (spatial + stop heads)
        self.policy = policy.to(self.device)

        #  One‑step memory (for glimpse sequence)
        self.memory = nn.LSTMCell(embd_dim, embd_dim).to(self.device)
        self.rnn = nn.LSTM(embd_dim, embd_dim, batch_first=True).to(self.device)

        #  value head  (decrease variance of REINFORCE)
        self.cricit = nn.Sequential(nn.Linear(embd_dim, 64),
                                    nn.ReLU(), 
                                    nn.Linear(64, 1)).to(self.device)

        #  Small stability tweaks ------------------------------------------------
        if hasattr(self.gate, "gate"):
            nn.init.zeros_(self.gate.gate[-1].weight)
            nn.init.zeros_(self.gate.gate[-1].bias)
        nn.init.zeros_(self.policy.dist_head.weight)
        nn.init.zeros_(self.policy.dist_head.bias)
        # bias the stop head negative so the agent tends to continue at start
        nn.init.constant_(self.policy.stop_head.bias, -2.0)

        #  Hyper‑params ----------------------------------------------------------
        self.attn_tau = 0.5  # temperature for soft attention
        self.entropy_weight = 1e-1  # encourage exploration

        #  Optimisers ------------------------------------------------------------
        self.classification_criterion = nn.CrossEntropyLoss()
        self.classification_optimizer = torch.optim.Adam(
            chain(
                self.encoder.parameters(),
                self.classifier.parameters(),
                self.gate.parameters(),
                self.rnn.parameters(),
                self.memory.parameters(),
            ),
            lr=1e-3,
        )
        self.reinforce_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.cricit_optimizer = torch.optim.Adam(self.cricit.parameters(), lr=1e-3)

    def _retina_step(self, x, center):
        """Extract pxp patch around center (in [-1,1] coords)."""
        _, _, S, _ = x.shape
        grid = make_glimpse_grid(center, min(int(round(self.p)), S), S)
        patch = F.grid_sample(x, grid, align_corners=True)
        if patch.size(-1) != self.p:
            patch = F.interpolate(patch, size=(self.p, self.p), mode="bilinear", align_corners=True)
        return patch

    def _idx_to_coord(self, idx):
        """Map discrete action index normalised (x,y) in [-1,1]."""
        idx = idx.to(torch.float32)
        row = torch.floor(idx / self.S)
        col = idx % self.S
        x_center = 2.0 * col / (self.S - 1) - 1.0
        y_center = 2.0 * row / (self.S - 1) - 1.0
        return torch.stack([x_center, y_center], dim=-1)

    @staticmethod
    def _discounted(rewards, gamma):
        """List[T] tensor [T,B] of discounted returns."""
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
        return F.tanh(ce0 - ce1)

    def _forward_seq(self, seq_feats):
        """forward pass on a sequence of features"""
        
        # Pool sequence with another LSTM + attention gate ------------------
        seq_input = torch.cat(seq_feats, dim=1) if len(seq_feats) > 1 else seq_feats[0]
        rnn_out, _ = self.rnn(seq_input)  # [B,T,D]

        scores = self.gate(rnn_out)           # [B,T,1]
        alpha = torch.softmax(scores / self.attn_tau, dim=1)
        pooled = (rnn_out * alpha).sum(dim=1)  # [B,D]
        logits = self.classifier(pooled)       # [B,n_classes]

        return logits

    # ---------------------------------------------------------------------
    #  Forward rollout
    # ---------------------------------------------------------------------

    def forward(self, x, targets = None, max_steps = 8, min_steps = 3):
        """Run an episode.

        Returns
        -------
        logits : [B, n_classes]
        logps  : [T,B]   combined log-prob per step
        returns: [T,B]   discounted rewards  (zeros if *targets* is None)
        entropies: [T,B] step-wise entropy (for regularisation)
        """
        B, _, _, _ = x.shape
        device = x.device

        # initial hidden state
        h_t = x.new_zeros(B, self.memory.hidden_size)
        c_t = x.new_zeros(B, self.memory.hidden_size)
        prev_ctx = h_t.detach()

        logps, entropies, rewards, seq_feats, values = [], [], [], [], []

        for t in range(max_steps):
            action_logits, stop_logits = self.policy(prev_ctx.detach())

            act_dist = Categorical(logits=action_logits)
            stop_dist = Bernoulli(logits=stop_logits.squeeze(-1))

            idx = act_dist.sample()
            stop = stop_dist.sample()

            logps.append(act_dist.log_prob(idx) + stop_dist.log_prob(stop))
            entropies.append(act_dist.entropy() + stop_dist.entropy())

            patch = self._retina_step(x, self._idx_to_coord(idx))
            feat_t = self.encoder(patch, self._idx_to_coord(idx)).view(B, -1)
            
            h_t, c_t = self.memory(feat_t, (h_t, c_t))
            baseline_t = self.cricit(h_t.detach()).squeeze(-1)
            prev_ctx = h_t.detach()  # next‑step policy context

            values.append(baseline_t)
            seq_feats.append(feat_t.unsqueeze(1))

            if t == 0 and targets is not None:
                with torch.no_grad():
                    rewards.append(torch.zeros(B, device=device) - self.step_cost) 
                    prev_logits = self._forward_seq(seq_feats) # next-step reward context

            elif targets is not None:
                with torch.no_grad():
                    logits = self._forward_seq(seq_feats) # next-step reward context
                    step_reward = self._confidence_reward(logits, prev_logits, targets) - self.step_cost
                    prev_logits = logits.detach()
                    rewards.append(step_reward)

            if stop.bool().all():
                break
                
        # final class prediction 
        logits = self._forward_seq(seq_feats)

        # final reward: 1 if prediction correct else 0 ----------------------
        if targets is not None:
            preds = logits.argmax(dim=1)
            final_r = -5.0 if not (preds == targets).bool() else 5.0  # [B]
            rewards[-1] += final_r  # last reward
        else:
            # Evaluation mode... no RL signal
            rewards = [torch.zeros(B, device=device)] * len(seq_feats)

        returns = self._discounted(rewards, self.gamma)  # [T,B]
        advantages = returns - torch.stack(values).detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8))

        # stack lists
        values = torch.stack(values)        # [T,B]
        logps = torch.stack(logps)          # [T,B]
        entropies = torch.stack(entropies)  # [T,B]

        return logits, logps, advantages, returns, values, entropies
    
    
    @torch.no_grad()
    def greedy_forward(self, x, max_steps=32, min_steps=3):
        """Run a greedy episode.

        Returns
        -------
        logits : [B, n_classes]
        steps : int
        """
        B, _, _, _ = x.shape

        # initial hidden state
        h_t = x.new_zeros(B, self.memory.hidden_size)
        c_t = x.new_zeros(B, self.memory.hidden_size)
        prev_ctx = h_t

        seq_feats, centers, patches = [], [], []
        steps = 0

        for t in range(max_steps):
            action_logits, stop_logits = self.policy(prev_ctx.detach())

            # greedy action
            idx = torch.argmax(action_logits, dim=1)
            s = F.sigmoid(stop_logits.squeeze(-1))          # [B]
            stop = (s > 0.5).bool()        


            patch = self._retina_step(x, self._idx_to_coord(idx))
            feat_t = self.encoder(patch, self._idx_to_coord(idx)).view(B, -1)

            centers.append(self._idx_to_coord(idx)[0])
            patches.append(patch)

            h_t, c_t = self.memory(feat_t, (h_t, c_t))
            prev_ctx = h_t  # next‑step policy context

            seq_feats.append(feat_t.unsqueeze(1))
            steps += 1

            if stop:
                break

        # final class prediction 
        logits = self._forward_seq(seq_feats)

        return logits, centers, patches, steps
    

    # ------------------------------------------------------------------
    #  Training helpers
    # ------------------------------------------------------------------

    def train_agent(self, epochs, trainloader, testloader=None, start_steps=10, max_steps=64):
        self.train()
        for epoch in range(epochs):
            # Max Step schedule -----------------------------------------------------
            steps = min((start_steps + (2 * epoch)), max_steps)
            total_policy_loss, total_value_loss, total_classification_loss, total_returns, total = 0.0, 0.0, 0.0, 0.0, 0.0
            for imgs, targets in trainloader:
                
                total += 1
                imgs, targets = imgs.to(self.device), targets.to(self.device)

                logits, logps, advantages, returns, values, entropies = self.forward(imgs, targets, max_steps=steps)

                total_returns += returns.mean(dim=0).flatten().item()
                # Supervised loss --------------------------------------------------
                cls_loss = self.classification_criterion(logits, targets)
                total_classification_loss += cls_loss.item()

                # Policy loss ------------------------------------------------------
                policy_loss = -(logps * advantages.detach()).mean()
                entropy_loss = -(self.entropy_weight * entropies.mean())
                rl_loss = policy_loss + entropy_loss
                total_policy_loss += rl_loss.item()

                # Value loss -------------------------------------------------------
                value_loss = 0.5 * (values - returns.detach()).pow(2).mean()
                total_value_loss += value_loss.item()

                #  Back‑prop RL branch first ---------------------------------------
                self.cricit_optimizer.zero_grad()
                self.reinforce_optimizer.zero_grad()
                rl_loss.backward(retain_graph=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cricit.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                self.reinforce_optimizer.step()
                self.cricit_optimizer.step()

                #  Back‑prop supervised branch --------------------------------------
                self.classification_optimizer.zero_grad()
                cls_loss.backward()
                self.classification_optimizer.step()
               
            print(f'Epoch {epoch + 1} | Policy Loss : {total_policy_loss / total} | Classification Loss : {total_classification_loss / total} | Value Loss : {total_value_loss / total} | Max Steps : {steps} | Avg Reward : {total_returns / total}')
        
            if testloader is not None:
                self.eval_agent(testloader, max_steps=max_steps)

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_agent(self, testloader, max_steps=32):
        self.eval()
        total, correct, loss_sum, step_sum = 0, 0, 0.0, 0.0
        
        for imgs, targets in testloader:
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            logits, _, _, steps = self.greedy_forward(imgs, max_steps=max_steps)
            loss_sum += self.classification_criterion(logits, targets).item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            step_sum += steps
            total += targets.size(0)

        acc = correct / max(total, 1)
        avg_loss = loss_sum / max(total, 1)
        avg_steps = step_sum / max(total, 1)
        
        print(f'Test Accuracy : {acc} | Test Loss : {avg_loss} | Avg Steps : {avg_steps}')
        return acc, avg_loss
    
    @torch.no_grad()
    def viz_glimpses(self, x, max_steps=32):
        """displays what the agent is seeing per time step"""
        x = x.to(self.device)
        logits, centers, patches, steps = self.greedy_forward(x, max_steps=max_steps)

        plot_centers(x.squeeze(0).detach().cpu().numpy(), centers)

        

