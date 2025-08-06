import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from math import sqrt
from itertools import chain

from utils import make_glimpse_grid, step_cost_t, entropy_weight_t
from plotter import plot_centers

# -----------------------------------------------------------------------------
#  Glimpse‑based agent with REINFORCE + entropy bonus + baseline
# -----------------------------------------------------------------------------

class GlimpseAgent(nn.Module):
    """Recurrent glimpse policy-gradient classifier.

    Forward returns logits and all tensors needed for REINFORCE so the training
    loop can compute the losses.
    """

    def __init__(self, policy, encoder, classifier, gate, seq_summarizer, context_memory, 
                 action_space, stride, init_entropy_weight = 1e-1, step_cost = 1e-5, gamma = 0.96, image_size = 28, 
                 patch_size = 14, embd_dim = 128, device = None):
        super().__init__()

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.S = image_size   # input image resolution (assume square)
        self.p = patch_size   # size of extracted glimpse
        self.action_space = action_space
        self.gamma = gamma
        self.max_step_cost = step_cost
        self.stride = stride
        self.first = patch_size // 2          
        self.last = image_size - patch_size//2 

        #  Feature extractor, attention gate, and classifier
        self.encoder = encoder.to(self.device)
        self.gate = gate.to(self.device)
        self.classifier = classifier.to(self.device)

        #  Policy network (spatial + stop heads)
        self.policy = policy.to(self.device)

        #  One‑step memory (for glimpse sequence)
        self.memory = context_memory.to(self.device)
        self.rnn = seq_summarizer.to(self.device)

        #  value head  (decrease variance of REINFORCE)
        self.cricit = nn.Sequential(nn.Linear(embd_dim, 64),
                                    nn.ReLU(), 
                                    nn.Linear(64, 1)).to(self.device)

        #  Small stability tweaks ------------------------------------------------
        nn.init.zeros_(self.policy.dist_head.weight)
        nn.init.zeros_(self.policy.dist_head.bias)
        # bias the stop head negative so the agent tends to continue at start
        nn.init.constant_(self.policy.stop_head.bias, -2.0)

        #  Hyper‑params ----------------------------------------------------------
        self.attn_tau = 0.5  # temperature for soft attention
        self.entropy_weight = init_entropy_weight  # encourage exploration
        self.init_entropy = init_entropy_weight

        #  Optimisers ------------------------------------------------------------
        self.classification_criterion = nn.CrossEntropyLoss()
        self.classification_optimizer = torch.optim.Adam(
            chain(
                self.encoder.parameters(),
                self.classifier.parameters(),
                # self.gate.parameters(),
                self.rnn.parameters(),
                self.memory.parameters(),
            ),
            lr=3e-5,
        )
        

        self.reinforce_optimizer = torch.optim.Adam(chain(self.policy.parameters(), self.memory.parameters()), lr=3e-4)
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
        row  = idx // self.action_space       
        col  = idx %  self.action_space     

        patch_x = self.first + col * self.stride      
        patch_y = self.first + row * self.stride

        # normalise to [-1,1]
        x_c  = 2.0 * patch_x / (self.S - 1) - 1.0
        y_c  = 2.0 * patch_y / (self.S - 1) - 1.0

        return torch.stack([x_c, y_c], dim=-1)

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
        return ce0 - ce1

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

    def forward(self, x, targets=None, max_steps=8):
        """
        Returns
        -------
        logits : [B, n_classes]
        logps  : [T,B]   combined log-prob per step (move+stop)
        returns: [T,B]   discounted rewards
        entropies: [T,B] step-wise entropy (move+stop)
        values : [T,B]   critic baseline per step
        """
        B, _, _, _ = x.shape
        device = x.device

        # ----- episode state -----
        n = self.action_space
        A = n * n                                # number of discrete locations

        # start at the grid center (for even n this picks one of the 4 middles)
        center_idx = (n // 2) * n + (n // 2)
        prev_idx = x.new_full((B,), center_idx, dtype=torch.long, device=device)
        prev_loc = self._idx_to_coord(prev_idx)

        # memory state
        h_t = x.new_zeros(B, self.memory.hidden_size)
        c_t = x.new_zeros(B, self.memory.hidden_size)
        prev_ctx = h_t

        # visited mask (mark starting cell as visited)
        visited = torch.zeros(B, A, device=device, dtype=torch.bool)
        upd0 = F.one_hot(prev_idx, num_classes=A).to(device=device, dtype=torch.bool)
        visited |= upd0

        alive = torch.ones(B, dtype=torch.bool, device=device)

        # logs
        logps, entropies, rewards, seq_feats, values = [], [], [], [], []

        prev_logits = None  # for ΔCE reward

        for t in range(max_steps):
            # ---------- SENSE: read current patch, update memory ----------
            patch  = self._retina_step(x, prev_loc)
            feat_t = self.encoder(patch, prev_loc).view(B, -1)
            h_t, c_t = self.memory((feat_t, (h_t, c_t)))
            prev_ctx = h_t

            # critic baseline for variance reduction (detach to avoid critic->memory grads)
            baseline_t = self.cricit(h_t.detach()).squeeze(-1) * alive
            values.append(baseline_t)

            seq_feats.append(feat_t.unsqueeze(1))

            # step reward (ΔCE) using classification head on accumulated features
            if targets is not None:
                with torch.no_grad():
                    logits_t = self._forward_seq(seq_feats)  # [B, n_classes]
                    if prev_logits is None:
                        # first step: only pay the step cost
                        step_reward = -self.step_cost * torch.ones(B, device=device)
                    else:
                        # ΔCE = ce(prev) - ce(curr) = Δ log p_true ; clip for stability
                        delta = self._confidence_reward(logits_t, prev_logits, targets)
                        step_reward = delta.clamp(-0.25, 0.25) - self.step_cost
                    rewards.append(step_reward * alive)
                    prev_logits = logits_t.detach()
            else:
                # evaluation without RL signal
                rewards.append(torch.zeros(B, device=device))

            # ---------- ACT: pick NEXT location + stop from updated state ----------
            action_logits, stop_logits = self.policy(prev_ctx, prev_loc)

            # per-step mask must be detached copy (avoid in-place version issues)
            step_mask = visited.detach().clone()
            masked_logits = action_logits.masked_fill(step_mask, -1e9)

            # if a row has all actions visited, fall back to uniform logits
            all_done = step_mask.all(dim=1)
            if all_done.any():
                masked_logits = torch.where(
                    all_done.unsqueeze(1), torch.zeros_like(masked_logits), masked_logits
                )

            act_dist  = Categorical(logits=masked_logits)
            stop_dist = Bernoulli(logits=stop_logits.squeeze(-1))

            idx_next = act_dist.sample()
            stop     = stop_dist.sample().bool()

            # respect 'alive' mask
            idx_next = torch.where(alive, idx_next, torch.zeros_like(idx_next))
            stop     = torch.where(alive,   stop,     torch.zeros_like(stop))

            # log-prob & entropy (sum of move + stop)
            logp_t = (act_dist.log_prob(idx_next) + stop_dist.log_prob(stop.float())) * alive
            entr_t = (act_dist.entropy()         + stop_dist.entropy())               * alive
            logps.append(logp_t)
            entropies.append(entr_t)

            # update visited with the chosen NEXT index (MPS-safe, no boolean indexing)
            upd = F.one_hot(idx_next, num_classes=A).to(device=device, dtype=torch.bool)
            upd = upd & alive.unsqueeze(1)
            visited |= upd

            # advance to next step
            prev_idx = idx_next
            prev_loc = self._idx_to_coord(prev_idx)

            # update alive
            alive = alive & (~stop)
            if not alive.any():
                break

        # ---------- final class prediction & terminal reward ----------
        logits = self._forward_seq(seq_feats)  # [B,n_classes]

        if targets is not None:
            preds = logits.argmax(dim=1)
            final_r = torch.where(
                preds == targets,
                torch.tensor( 2.0, device=device),
                torch.tensor(-2.0, device=device),
            )
            rewards[-1] = rewards[-1] + final_r  # add to last step
        else:
            # already appended zeros above; nothing to add
            pass

        # ---------- stack & advantages ----------
        values     = torch.stack(values)        # [T,B]
        logps      = torch.stack(logps)         # [T,B]
        entropies  = torch.stack(entropies)     # [T,B]
        returns    = self._discounted(rewards, self.gamma)  # [T,B]

        advantages = (returns - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8))

        return logits, logps, advantages, returns, values, entropies
        
        
    @torch.no_grad()
    def greedy_forward(self, x, max_steps=32):
        """
        Returns
        -------
        logits : [B, n_classes]
        centers: list of coords (when B==1) for visualisation
        patches: list of patches (when B==1)
        steps  : total number of steps taken (sum over batch)
        """
        B, _, _, _ = x.shape
        device = x.device

        n = self.action_space
        A = n * n

        # start at grid center
        center_idx = (n // 2) * n + (n // 2)
        prev_idx = x.new_full((B,), center_idx, dtype=torch.long, device=device)
        prev_loc = self._idx_to_coord(prev_idx)

        h_t = x.new_zeros(B, self.memory.hidden_size)
        c_t = x.new_zeros(B, self.memory.hidden_size)
        prev_ctx = h_t

        visited = torch.zeros(B, A, device=device, dtype=torch.bool)
        upd0 = F.one_hot(prev_idx, num_classes=A).to(device=device, dtype=torch.bool)
        visited |= upd0

        alive = torch.ones(B, dtype=torch.bool, device=device)

        seq_feats, centers, patches = [], [], []
        steps_per_item = torch.zeros(B, device=device, dtype=torch.float32)

        for t in range(max_steps):
            # SENSE
            patch  = self._retina_step(x, prev_loc)
            feat_t = self.encoder(patch, prev_loc).view(B, -1)
            if B == 1:
                centers.append(prev_loc[0]); patches.append(patch)

            h_t, c_t = self.memory((feat_t, (h_t, c_t)))
            prev_ctx = h_t
            seq_feats.append(feat_t.unsqueeze(1))
            steps_per_item += alive.to(steps_per_item.dtype)

            # ACT (greedy)
            action_logits, stop_logits = self.policy(prev_ctx, prev_loc)

            masked_logits = action_logits.masked_fill(visited, -1e9)
            all_done = visited.all(dim=1)
            masked_logits = torch.where(all_done.unsqueeze(1),
                                        torch.zeros_like(masked_logits),
                                        masked_logits)

            idx_next = masked_logits.argmax(dim=1)
            stop     = (stop_logits.squeeze(-1) > 0) | all_done

            idx_next = torch.where(alive, idx_next, torch.zeros_like(idx_next))
            stop     = torch.where(alive,   stop,     torch.zeros_like(stop))

            # mark visited
            upd = F.one_hot(idx_next, num_classes=A).to(device=device, dtype=torch.bool)
            upd = upd & alive.unsqueeze(1)
            visited |= upd

            # advance
            prev_idx = idx_next
            prev_loc = self._idx_to_coord(prev_idx)

            alive = alive & (~stop)
            if not alive.any():
                break

        logits = self._forward_seq(seq_feats)
        return logits, centers, patches, steps_per_item.sum()
        

    # ------------------------------------------------------------------
    #  Training helpers
    # ------------------------------------------------------------------

    def train_agent(self, epochs, trainloader, testloader=None, start_steps=4, max_steps=8):
        self.train()
        for epoch in range(epochs):
            # Max Step schedule -----------------------------------------------------
            steps = min((start_steps + (2 * epoch)), max_steps)
            self.step_cost = step_cost_t(epoch, epochs, self.max_step_cost, self.max_step_cost // 2)
            self.entropy_weight = max(0.01, entropy_weight_t(epoch, epochs, self.init_entropy))
            total_policy_loss, total_value_loss, total_classification_loss, total_returns, total = 0.0, 0.0, 0.0, 0.0, 0.0
            for imgs, targets in trainloader:
                
                total += 1
                imgs, targets = imgs.to(self.device), targets.to(self.device)

                logits, logps, advantages, returns, values, entropies = self.forward(imgs, targets, max_steps=steps)

                total_returns += returns[0].mean().item()

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
                torch.nn.utils.clip_grad_norm_(self.cricit.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.reinforce_optimizer.step()
                self.cricit_optimizer.step()

                #  Back‑prop supervised branch --------------------------------------
                self.classification_optimizer.zero_grad()
                cls_loss.backward()
                self.classification_optimizer.step()
               
            print(f'Epoch {epoch + 1} | Policy Loss : {total_policy_loss / total} | Classification Loss : {total_classification_loss / total} | Value Loss : {total_value_loss / total} | Max Steps : {steps} | Avg Reward : {total_returns / total} | Entropy Weight {self.entropy_weight}')

            if testloader is not None:
                self.eval_agent(testloader, max_steps=steps)
                for i in range(10):
                    self.viz_glimpses(next(iter(trainloader))[0][2:3, :], epoch=epoch + 1, idx=i, max_steps=steps)

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_agent(self, testloader, max_steps):
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
        
        print(f'Test Accuracy : {acc} | Test Loss : {avg_loss} | Avg Steps : {avg_steps}\n')
        return acc, avg_loss
    
    @torch.no_grad()
    def viz_glimpses(self, x, epoch, idx, max_steps=6):
        """displays what the agent is seeing per time step"""
        x = x.to(self.device)
        logits, centers, patches, steps = self.greedy_forward(x, max_steps=max_steps)

        plot_centers(x.squeeze(0).detach().cpu().numpy(), epoch, idx, centers)

        

