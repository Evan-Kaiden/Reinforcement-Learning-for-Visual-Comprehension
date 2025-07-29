from dataset import trainloader, testloader

from glimpse_agent import glimpseAgent
from networks import PerceptionPolicy, PerceptionEncoder, ContextMemory, Classifier
import torch
def main():
    patch = 12
    embd = 128
    classes = 10
    img_size = 32
    img_channels = 3
    policy = PerceptionPolicy(embd)
    encoder = PerceptionEncoder(img_channels, patch, embd)
    memory = ContextMemory(embd)
    classifier = Classifier(embd, classes)
    torch.autograd.set_detect_anomaly(True)
    agent = glimpseAgent(policy, encoder, memory, classifier, img_size, patch, 1e-3, 1e-3)
    agent.train(trainloader)

if __name__ == '__main__':
    main()