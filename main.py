import torch
import argparse

import data_loader
from glimpse_agent import GlimpseAgent
from forward_seq_pretrainer import preTrainer
from networks import PerceptionPolicy, PerceptionEncoder, ContextMemory, Classifier, gate, SequenceSummarizer


# Argparse Setup
parser = argparse.ArgumentParser()

# Dataset Arguments
parser.add_argument('--img_size', type=int, default=28)
parser.add_argument('--clutter_count', type=int, default=None)
parser.add_argument('--clutter_size', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--random_seed', type=int, default=32)

# Pretrainer Arguments
parser.add_argument('--pretrain', type=lambda x: x.lower() == 'true', default=None)
parser.add_argument('--cls_lr', type=float, default=1e-3)
parser.add_argument('--cls_epochs', type=int, default=10)

# RL Arguments
parser.add_argument('--agent_epochs', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--entropy_weight', type=float, default=0.03)
parser.add_argument('--steps', type=int, default=6)

# Image Arguments
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--embd_dim', type=int, default=64)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--img_channels', type=int, default=1)
parser.add_argument('--stride', type=int, default=4)

# checkpoint & viz Arguments
parser.add_argument('--from_checkpoint', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--viz', type=lambda x: x.lower() == 'true', default=False)

args = parser.parse_args()

torch.manual_seed(args.random_seed)
if not args.from_checkpoint:
    assert args.pretrain is not None, 'Must Specify --pretrain True/False'

if args.img_size != 28:
    assert args.clutter_count is not None, "Must Specificy Clutter Count when using cluttered MNIST, --clutter_count int"
    trainloader, testloader = data_loader.get_cluttered_data('MNIST', args.img_size, args.clutter_count, args.clutter_size, args.batch_size, './data')
    
else:
    trainloader, testloader = data_loader.get_data('MNIST', args.batch_size, './data')

# Action Space Calculation
action_space = int((args.img_size - args.patch_size) / args.stride) + 1

# Model Setup
policy = PerceptionPolicy(action_space=action_space ** 2, embd_dim=args.embd_dim)
policy_memory = ContextMemory(embd_dim=args.embd_dim)
encoder = PerceptionEncoder(embd_dim=args.embd_dim, in_channels=args.img_channels)
classifier = Classifier(embd_dim=args.embd_dim, n_classes=args.n_classes)
attention_gate = gate(embd_dim=args.embd_dim)
seq_summarizer = SequenceSummarizer(embd_dim=args.embd_dim)

if args.pretrain and not args.from_checkpoint:
    # Classifiacation Pre-Trainer Setup
    pretrainer = preTrainer(encoder, classifier, attention_gate, seq_summarizer, 
                            lr=args.cls_lr, patch_size=args.patch_size, stride=args.stride, 
                            in_channels=args.img_channels, img_size=args.img_size, device='mps')

    # Classifiacation Models Training
    pretrainer.train_models(epochs=args.cls_epochs, trainloader=trainloader, testloader=testloader)

    # Extract Trained Models
    encoder, classifier, attention_gate, seq_summarizer = pretrainer.get_models(from_saved=True, save_dir='pretrainer_ckpt')

# Agent Setup
agent = GlimpseAgent(policy=policy, encoder=encoder, classifier=classifier, gate=attention_gate, seq_summarizer=seq_summarizer, context_memory=policy_memory,
                        image_size=args.img_size, stride=args.stride, patch_size=args.patch_size, embd_dim=args.embd_dim, 
                        device='mps', action_space=action_space, gamma=args.gamma, init_entropy_weight=args.entropy_weight)

if not args.from_checkpoint:
    # Agent Training
    agent.train_agent(epochs=args.agent_epochs, steps=args.steps, trainloader=trainloader, testloader=testloader)
else:
    agent.set_models('agent_ckpt')

# Make Gifs
if args.viz == True:
    loader = iter(testloader)
    for i in range(10):
        x = next(loader)[0][:1, :]
        agent.make_viz(x=x, filepath=f'attention{i}.gif')