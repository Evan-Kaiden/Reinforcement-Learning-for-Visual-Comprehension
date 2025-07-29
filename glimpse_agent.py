import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_glimpse_grid

class glimpseAgent():
    def __init__(self, policy_model, perception_model, memory_model, classifier_model, image_size, patch_size, policy_lr, supervised_lr):
        self.policy_model = policy_model
        self.perception_model = perception_model
        self.memory_model = memory_model
        self.classifier_model = classifier_model

        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=policy_lr)
        self.supervised_optimizer = torch.optim.Adam(
            list(self.perception_model.parameters()) +
            list(self.memory_model.parameters()) +
            list(self.classifier_model.parameters()),
            lr=supervised_lr
        )
        
        self.image_size = image_size
        self.patch_size = patch_size

    def _sample(self, image, grid):
        return F.grid_sample(image, grid, align_corners=True)


    def train(self, train_loader, eps_per_image=10, max_steps=5, lr=1e-3, gamma=0.99, stop_thresh=0.75):

        self.policy_model.train()
        self.perception_model.train()
        self.memory_model.train()
        self.classifier_model.train()

        criterion = nn.CrossEntropyLoss()
        for inputs, targets in train_loader:
            for episoide in range(eps_per_image):
                step = 0
                current_context, cell_state = (torch.zeros(1, self.memory_model.embd_dim), torch.zeros(1, self.memory_model.embd_dim))
                encodings, confidences, log_probs = [], [], []

                while step < max_steps:
                    # Run current context through policy
                    location, raw_location, stop_prob, dist = self.policy_model(current_context.squeeze(0))
                    raw_location = raw_location.clone()

                    if stop_prob > stop_thresh:
                        break

                    log_prob = dist.log_prob(raw_location).sum(dim=-1) 
                    log_probs.append(log_prob)

                    # sample a patch based on policy output
                    location = location.unsqueeze(0)
                    grid = make_glimpse_grid(location, self.patch_size, self.image_size)
                    patch = self._sample(inputs, grid)

                    # generate encoding of patch
                    confidence, encoding = self.perception_model(patch)

                    encodings.append(encoding)
                    confidences.append(confidence.unsqueeze(0))
                    # pass encoding through memory model to update memory
                    _, (current_context, cell_state) = self.memory_model(encoding, (current_context, cell_state))

                    step += 1


                encodings = torch.stack(encodings).view(step, self.memory_model.embd_dim)
                confidences = torch.stack(confidences).view(step, 1)

                h0 = torch.zeros(1, self.memory_model.embd_dim)
                c0 = torch.zeros(1, self.memory_model.embd_dim)
                prev_state = (h0, c0)
  
                vals = encodings * confidences 

                _, (current_context, cell_state) = self.memory_model(vals, prev_state)
                current_context = current_context.detach()

                outputs = self.classifier_model(current_context)
                _, predicted = outputs.max(1)

                with torch.no_grad():
                    reward = (predicted == targets).float()

                log_probs = torch.stack(log_probs)
                rewards = reward.expand_as(log_probs)

                policy_loss = -(log_probs * rewards.detach()).mean()  
                
                with torch.no_grad():
                    _, _, _, entropy_dist = self.policy_model(current_context.squeeze(0))
                    entropy = entropy_dist.entropy().sum(dim=-1).mean()

                policy_loss = policy_loss - 0.01 * entropy
                classification_loss = criterion(outputs, targets)

                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.policy_optimizer.step()

                self.supervised_optimizer.zero_grad()
                classification_loss.backward()
                self.supervised_optimizer.step()
                

