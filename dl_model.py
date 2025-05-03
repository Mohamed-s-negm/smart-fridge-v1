import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from PIL import Image
import copy
import os

class DL_ImgClass:

    #1. We setup the main information
    def __init__(self, device=None, num_classes=None, model_path=None, num_epochs=150, batch_size=32, lr=1e-4, early_stopping_patience=10):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.dataloaders = {}
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'best_model.pth')
        self.model = None
        self.criterion = None
        self.optimizer = None


    #2. Data Transformations
    def prepare_data(self, data_dir, input_size=(224, 224)):
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        #3. Load the Data
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
        self.class_names = train_dataset.classes

        train_len = int(0.8 * len(train_dataset))
        val_len = len(train_dataset) - train_len

        train_data, val_data = random_split(train_dataset, [train_len, val_len])

        self.dataloaders['train'] = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.dataloaders['val'] = DataLoader(val_data, batch_size=self.batch_size, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self._init_model()


    #4. initialize VGG16
    def _init_model(self):
        assert self.num_classes is not None, "num_classes must be given a value"
        vgg16 = models.vgg16(pretrained=True)

        # Unfreeze all layers for full fine-tuning
        for param in vgg16.parameters():
            param.requires_grad = True


        # Replace the last layer of the classifier
        vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, self.num_classes)

        self.model = vgg16.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        #We add a scheduler to motitor the validation rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5)



    #5. Training Function
    def train(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        epoch_no_improve = 0

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print('_'* 20)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0


                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = 100 * running_corrects.double() / len(self.dataloaders[phase].dataset)

                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == 'val':
                    self.scheduler.step(epoch_acc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        self.save_model()
                        epoch_no_improve = 0
                    else:
                        epoch_no_improve += 1

            print()

            if epoch_no_improve >= self.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs.")

        print(f"Best val Acc: {best_acc:.4f}")
        self.model.load_state_dict(best_model_wts)
        self.test()
        
   
    #6. Test the Model
    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")


    #7. Save the new model
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")


    #8. load an existing model
    def load_model(self, path):
        self._init_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")


    #9. Predict method to be used
    def predict(self, image_path):
        self.model.eval()

        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        index = predicted.item()
        return self.class_names[index]


