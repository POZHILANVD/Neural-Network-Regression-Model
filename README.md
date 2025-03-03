# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

![image](https://github.com/user-attachments/assets/81fc6451-8e15-4581-9f90-9dd7c073c191)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:POZHILAN V D
### Register Number:212223240118
```python
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1, 8)
    self.fc2 = nn.Linear(8, 10)
    self.fc3 = nn.Linear(10, 1)
    self.relu = nn.ReLU()
    self.history = {'loss': []} 

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)  
    return x
# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
  
    output = ai_brain(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)


```
## Dataset Information

![image](https://github.com/user-attachments/assets/e1e96450-8f19-4a6f-a900-c65a50da45bf)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/82b00306-c933-4227-91ab-8a7a62433b1d)

### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/79ae32d7-a9ea-426c-a2d6-30071f2f90cb)

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
