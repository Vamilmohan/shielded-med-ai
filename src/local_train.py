import torch
import torch.nn as nn
import torch.optim as optim
import os
# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Limit CPU threads
torch.set_num_threads(1)
def local_train(model, dataloader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for _ in range(epochs):
        for images, labels in dataloader:
            labels = labels.view(-1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)

    return model.state_dict(), accuracy, avg_loss