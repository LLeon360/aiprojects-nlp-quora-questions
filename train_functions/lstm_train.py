import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from sklearn.metrics import recall_score


def lstm_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    step = 0
    
    writer = SummaryWriter()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            
            features = batch["input_ids"]
            labels = batch["labels"]
            # features = features.to(device)
            # labels = labels.to(device)

            # Forward propagate

            # outputs = model(features)
            # outputs = torch.flatten(outputs)

            outputs = forward_prop(batch, model, device)

            # Backpropagation and gradient descent            

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # Compute training loss and metrics
                # Log the results to Tensorboard.
                preds = torch.round(outputs).detach()
                acc = compute_accuracy(preds, labels)
                writer.add_scalar('Loss/train', loss, epoch)
                # writer.add_scalar('Accuracy/train', acc, epoch)
                writer.add_scalar('Specificity/train', compute_specificity(preds, labels), epoch)
                writer.add_scalar('Recall/train', compute_recall(preds, labels), epoch)

                # Compute validation loss and metrics
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!

                results = evaluate(val_loader, model, loss_fn, device) 
                writer.add_scalar('Loss/val', results["loss"], epoch)
                # writer.add_scalar('Accuracy/val', results["acc"], epoch)
                writer.add_scalar('Specificity/val', results["spec"], epoch)
                writer.add_scalar('Recall/val', results["recall"], epoch)

                #turn on training, evaluate turns off training
                model.train()

                print(f'Loss/train {loss} Loss/val {results["loss"]}')

            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total

def compute_specificity(outputs, labels):
    """
    Computes the specificity of a model's predictions. Uses recall but treats negative as positive
    """
    return recall_score(y_true=labels, y_pred=outputs, pos_label=0)

def compute_recall(outputs, labels):
    """
    Computes the recall of a model's predictions.
    """

    return recall_score(y_true=labels, y_pred=outputs)

def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    """
    #turn off training
    model.eval()

    batch = next(iter(val_loader))
    labels = batch["labels"]
    
    outputs = forward_prop(batch, model, device)
    
    loss = loss_fn(outputs, labels)
    preds = torch.round(outputs).detach()
    return { 
        "loss":loss, 
        "acc":compute_accuracy(preds, labels), 
        "spec": compute_specificity(preds, labels),
        "recall": compute_recall(preds, labels)
    }

def forward_prop(batch, model, device): 
    
    # features = features.to(device)
    # labels = labels.to(device)

    outputs = model(batch)
    outputs = torch.flatten(outputs)
    return outputs