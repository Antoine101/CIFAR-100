import torch

def compute_normstats(train_set):
    red_channels = torch.stack([train_set[i][0][0, :, :] for i in range(len(train_set))], dim=0)
    green_channels = torch.stack([train_set[i][0][1, :, :] for i in range(len(train_set))], dim=0)
    blue_channels = torch.stack([train_set[i][0][2, :, :] for i in range(len(train_set))], dim=0)
    train_set_mean = (red_channels.mean().item(), green_channels.mean().item(), blue_channels.mean().item())
    train_set_std = (red_channels.std().item(), green_channels.std().item(), blue_channels.std().item())
    return train_set_mean, train_set_std