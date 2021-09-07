import torch
import torchvision 
from torchvision import transforms, utils
from torch.utils.data import DataLoader 
import cv2 as cv 
import recognitionModel as nn
from dataloader import MyDataset
import dataloader as data 
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import os
DATATPATH__ = os.path.join(os.getcwd(), 'artisan_dataset')

print("Start")
#Labels and classes
listData = ['fist','L','nothing','okay','palm','Peace']
labels = {"fist":0,"L":1,"nothing":2,"okay":3,"palm":4,"Peace":5,"peace":5}


#splitting and extracting the data
train_x, train_y = data.get_data(DATATPATH__,listData, labels)
x_train, x_validation, y_train, y_validation = train_test_split(train_x, train_y, test_size = 0.2, shuffle=True)

#declaring the transformer for normalizing the data
transform = transforms.Compose([transforms.ToTensor(), #data normalization
                                transforms.Normalize(
                                    (0.5,), 
                                    (0.5,))])

#creatind data objects as MyDataset instances
trainig_set = MyDataset(x_train, y_train, transform=transform)
validation_set = MyDataset(x_validation, y_validation, transform=transform)

#use gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataLoader = DataLoader(trainig_set,
                              batch_size=64,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=torch.cuda.is_available())

validation_dataLoader = DataLoader(validation_set,
                              batch_size=64,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=torch.cuda.is_available())



#import the model from our model class and define the optimizer and the loss
model = nn.Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


  #Define the loss and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

#Training
for epoch in range(5):
  print("Starting epoch:", epoch)
  runing_loss = 0
  for i, data in enumerate(train_dataLoader,0):
    #get the inputs 
    inputs, labels = data

    #zero the parameter gradients
    optimizer.zero_grad()

    #forward + backward + optimize
    output = model(inputs)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    #print statistics
    runing_loss += loss.item()
       
  print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, runing_loss / (len(x_train)/64)))
  #Validation
  val_loss = 0
  with torch.no_grad():
    for data in validation_dataLoader:
      images, labels = data
      output = model(images)
      loss_val = criterion(output, labels)
      val_loss +=loss_val.item()

    print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, val_loss / (len(x_validation)/64)))  

print("Training process finished")


#testing on single image:


