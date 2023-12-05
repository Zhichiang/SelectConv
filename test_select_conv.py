import torch
from select_conv import DeformConv2d


if __name__ == "__main__":
    x = torch.ones((2, 1, 7, 7)).to(torch.device("cuda"))
    conv = DeformConv2d(1, 1, 5, padding=2).to(torch.device("cuda"))
    y = conv(x)

    print(x)
    print(conv)
    print(y)



