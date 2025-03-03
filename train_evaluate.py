from sklearn.metrics import f1_score
from tqdm import tqdm

def train_and_evaluate(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "best_model.pth",
    device=None
):
    """
    Trains the model, evaluates it on validation data, saves the best model, and tests it.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        checkpoint_path (str): Path to save the best model weights.
        device: PyTorch device ('cuda' or 'cpu'). If None, it will be auto-detected.

    Returns:
        Tuple: Test accuracy and F1-score.
    """

    # Automatically detect device if not provided
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Move model to the appropriate device
    # model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, 
    #                                               steps_per_epoch=len(train_loader), 
    #                                               epochs=100)

    best_val_loss = float('inf')  # Initialize the best validation loss

    for epoch in range(num_epochs):
        #print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for data, labels in tqdm(train_loader, desc="Training", leave=False):
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100.0 * correct_train / total_train
        #print(f"Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validating", leave=False):
                data, labels = data.to(device), labels.to(device)

                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, labels)

                # Accumulate metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = 100.0 * correct_val / total_val
        #print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            #print(f"Best model saved with Val Loss: {best_val_loss / len(val_loader):.4f}")

    # Load the best model for testing
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Testing phase
    test_loss = 0
    correct_test = 0
    total_test = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Testing", leave=False):
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Accumulate metrics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_test += predicted.eq(labels).sum().item()
            total_test += labels.size(0)

            # Collect labels and predictions for F1-score
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_accuracy = 100.0 * correct_test / total_test
    test_f1 = f1_score(all_labels, all_predictions, average="weighted")
    print(f"Test Loss: {test_loss / len(test_loader):.2f}, Test Accuracy: {test_accuracy:.2f}%, F1-Score: {test_f1:.4f}")

    return test_accuracy, test_f1
