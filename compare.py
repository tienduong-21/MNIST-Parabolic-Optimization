import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from mnist_original.main import Net as NetOriginal, train as train_original, test as test_original
from mnist_parabal.main import Net as NetParabal, train as train_parabal, test as test_parabal

if __name__ == '__main__':
     # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()
    print(f"Accelerator available: {torch.accelerator.is_available()}")

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_kwargs = {'batch_size': args.batch_size}
    print(f"Train kwargs: {train_kwargs}")
    test_kwargs = {'batch_size': args.test_batch_size}
    print(f"Test kwargs: {test_kwargs}")

    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    loss_original = []
    loss_parabol = []

    accuracy_original = []
    accuracy_parabol = []

    ### Train and test the original model
    model_original = NetOriginal().to(device)
    optimizer_original = optim.SGD(model_original.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_original(args, model_original, device, train_loader, optimizer_original, epoch)
        test_loss, test_accuracy = test_original(model_original, device, test_loader)

        loss_original.append(test_loss)
        accuracy_original.append(test_accuracy)

    ### Train and test the parabolic optimization model
    torch.manual_seed(args.seed)
    model_parabal = NetParabal().to(device)
    for epoch in range(1, args.epochs + 1):
        train_parabal(args, model_parabal, device, train_loader, epoch)
        test_loss, test_accuracy = test_parabal(model_parabal, device, test_loader)
        loss_parabol.append(test_loss)
        accuracy_parabol.append(test_accuracy)


    plt.figure(figsize=(12, 5))
    plt.plot(range(1, args.epochs + 1), loss_original, label='SGD')
    plt.plot(range(1, args.epochs + 1), loss_parabol, label='Parabolic Optimization')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.savefig('loss_comparison.png')

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, args.epochs + 1), accuracy_original, label='SGD')
    plt.plot(range(1, args.epochs + 1), accuracy_parabol, label='Parabolic Optimization')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.savefig('accuracy_comparison.png')

