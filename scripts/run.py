import torch
from ssm.cli import TrainingCLI
torch.autograd.set_detect_anomaly(True)
cli = TrainingCLI()
cli()
