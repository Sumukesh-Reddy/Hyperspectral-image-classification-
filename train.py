import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import grouper, sliding_window, count_sliding_window


def train(network, optimizer, criterion, train_loader, val_loader, epoch, saving_path, device, scheduler=None):
    """Train the network and return history of losses and val accuracies per epoch."""
    best_acc = -0.1
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    for e in tqdm(range(1, epoch + 1), desc="Training"):
        network.train()
        epoch_losses = []
        correct = 0
        total = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = network(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            # Track training accuracy
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        epoch_loss = np.mean(epoch_losses)
        train_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)

        # Validation
        val_acc, val_loss = validation(network, val_loader, device, criterion)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if e % 10 == 0 or e == 1:
            tqdm.write(f"Epoch {e}/{epoch} — loss: {epoch_loss:.6f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.6f}, val_acc: {val_acc:.4f}")

        if scheduler is not None:
            scheduler.step()

        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(network, is_best, saving_path, epoch=e, acc=best_acc)

    return history


def validation(network, val_loader, device, criterion=None):
    num_correct = 0.
    total_num = 0.
    val_losses = []
    network.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = network(images)
            if criterion is not None:
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())
            _, predicted = torch.max(outputs, dim=1)
            num_correct += (predicted == targets).sum().item()
            total_num += targets.size(0)
    overall_acc = num_correct / total_num
    avg_val_loss = np.mean(val_losses) if val_losses else 0.0
    return overall_acc, avg_val_loss


def test(network, model_dir, image, patch_size, n_classes, device):
    network.load_state_dict(torch.load(model_dir + "/model_best.pth", map_location=device))
    network.eval()

    batch_size = 64
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    # pad the image
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image.shape[:2] + (n_classes,))

    iterations = count_sliding_window(image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations,
                      desc="Inference on HSI"):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = network(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out
    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]


def save_checkpoint(network, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write("epoch = {epoch}: best OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best.pth'))
    else:
        if kwargs['epoch'] % 10 == 0:
            torch.save(network.state_dict(), os.path.join(saving_path, 'model.pth'))
