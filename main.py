from math import ceil
from typing import Union, Optional, Tuple

from sacred import Experiment, observers
import torch as th
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from data import make_dataset
from model import TransformerDecoder


grok_experiment = Experiment('grok')
observer = observers.FileStorageObserver('results/grok')
grok_experiment.observers.append(observer)


@grok_experiment.config
def config():
    # Dataset parameters
    base: int = 97
    train_frac: float = 0.5

    # Transformer parameters
    embedding_dim: int = 128
    dropout: float = 0.1
    decoder_layers: int = 2
    nhead: int = 4
    feedforward_dim: int = 512  # This induces 409315 ~= 4e5 parameters (97 base, 99 embedding tokens)

    # Training parameters
    batch_size: int = 512
    epochs: int = 10000
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 1.0

    # Misc
    device: Union[str, th.device] = 'cuda' if th.cuda.is_available() else 'cpu'
    seed: Optional[int] = 4  # chosen by fair dice roll, guaranteed random
    verbose: bool = True


def process_sequence(sequence, model, device):
    sequence = sequence.to(device)
    input_seq = sequence[:, :-1]
    target_values = sequence[:, -1]

    output_probs = model(input_seq)
    last_token_probs = output_probs[:, -1, :]
    return target_values, last_token_probs


def get_num_correct(target_values, last_token_probs):
    output_predictions = th.argmax(last_token_probs, dim=1)
    return th.sum(target_values == output_predictions).item()


def train_epoch(model, train_loader, loss_function, optimizer, scheduler, device):
    model.train()
    losses = 0

    for sequence in train_loader:
        target_values, last_token_probs = process_sequence(sequence, model, device)

        optimizer.zero_grad()
        loss = loss_function(last_token_probs, target_values)
        losses += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    return losses / len(train_loader.dataset)


def evaluate(model, train_loader, test_loader, loss_function, device):
    model.eval()
    losses = 0
    train_correct, test_correct = 0, 0

    for sequence in train_loader:
        target_values, last_token_probs = process_sequence(sequence, model, device)
        train_correct += get_num_correct(target_values, last_token_probs)

    for sequence in test_loader:
        target_values, last_token_probs = process_sequence(sequence, model, device)
        test_correct += get_num_correct(target_values, last_token_probs)

        loss = loss_function(last_token_probs, target_values)
        losses += loss.item()

    return (train_correct / len(train_loader.dataset),
            losses / len(test_loader.dataset),
            test_correct / len(test_loader.dataset)
            )


@grok_experiment.automain
def main(base: int,
         train_frac: float,
         embedding_dim: int,
         dropout: float,
         decoder_layers: int,
         nhead: int,
         feedforward_dim: int,
         batch_size: int,
         epochs: int,
         lr: float,
         betas: Tuple[float, float],
         weight_decay: float,
         device: Union[str, th.device],
         seed: Optional[int] = None,
         verbose: bool = False,
         ):
    if seed is not None:
        th.manual_seed(seed)

    train_loader, test_loader = make_dataset(base, train_frac=train_frac, batch_size=batch_size)
    gradient_updates_per_epoch = ceil(len(train_loader.dataset) / batch_size)

    num_classes = base + 2
    model = TransformerDecoder(embed_dim=embedding_dim,
                               nhead=nhead,
                               decoder_layers=decoder_layers,
                               feedforward_dim=feedforward_dim,
                               dropout=dropout,
                               n_classes=num_classes)

    model = model.to(device)
    optimizer = AdamW(model.parameters(),
                      lr=lr,
                      betas=betas,
                      weight_decay=weight_decay)
    scheduler = LinearLR(optimizer,
                         start_factor=0.1,
                         total_iters=10)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, scheduler, device)
        train_acc, test_loss, test_acc = evaluate(model, train_loader, test_loader, loss_function, device)
        grok_experiment.log_scalar(name='train_loss', value=train_loss, step=(epoch+1)*gradient_updates_per_epoch)
        grok_experiment.log_scalar(name='train_acc',  value=train_acc,  step=(epoch+1)*gradient_updates_per_epoch)
        grok_experiment.log_scalar(name='test_loss',  value=test_loss,  step=(epoch+1)*gradient_updates_per_epoch)
        grok_experiment.log_scalar(name='test_acc',   value=test_acc,   step=(epoch+1)*gradient_updates_per_epoch)
        if verbose:
            print(f" === EPOCH {epoch} === ")
            print(f"Train loss: {train_loss:.4f}; Train accuracy: {train_acc:.4f}")
            print(f"Test loss:  {test_loss:.4f}; Test accuracy:  {test_acc:.4f}")
