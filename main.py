from dataset import trainloader, testloader

from glimpse_agent import GlimpseAgent
from networks import PerceptionPolicy, PerceptionEncoder, ContextMemory, Classifier, gate

"""Reward. at each step pass the current context to the classifier. if the currrent context has pused the prob distribition
towards closer towards the target class, then reward is positive, else negative. final reward is the sum of all rewards plus the final classification reward."""

def main():
    patch = 6
    embd = 256
    classes = 10
    img_size = 28
    img_channels = 1

    policy = PerceptionPolicy(img_size*img_size, embd)
    encoder = PerceptionEncoder(input_shape=patch, embd_dim=embd, in_channels=img_channels)
    classifier = Classifier(embd_dim=embd, n_classes=classes)
    attention_gate = gate(embd_dim=embd)

    agent = GlimpseAgent(policy=policy, encoder=encoder,  classifier=classifier, gate=attention_gate,
                         image_size=img_size, patch_size=patch, embd_dim=embd,  device=None)

    agent.train_agent(epochs=20, trainloader=trainloader, testloader=testloader)

if __name__ == '__main__':
    main()
