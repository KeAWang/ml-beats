import torch

torch.manual_seed(1)


def generate_bin_data(num_points, length, prob=0.5):
    d = torch.distributions.Bernoulli(probs=torch.Tensor([prob]))
    x = d.sample((num_points, length))


N = 100
L = 1000

data = generate_bin_data(N, L)
torch.save(data, open('train_bin_data.pt', 'wb'))
