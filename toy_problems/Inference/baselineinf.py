import torchvision
import torchvision.datasets as datasets
import logging
import torch
import torchvision.transforms as transforms
import time as timers

from datetime import datetime
from tqdm import tqdm

mean_CIFAR10 = [0.49139968, 0.48215841, 0.44653091]
std_CIFAR10 = [0.24703223, 0.24348513, 0.26158784]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_CIFAR10, std_CIFAR10),
])

testset = datasets.CIFAR10(root='/home/dbreen/Documents/tddl/bigdata/cifar10', train=True, download=False, transform=transform)

batch_size = 128
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Setup the inference logger
logging.basicConfig(level=logging.INFO)
inference_logger = logging.getLogger('Inferencefin')

if not any(isinstance(handler, logging.FileHandler) for handler in inference_logger.handlers):
    fh = logging.FileHandler('inferencefin.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    inference_logger.addHandler(fh)

# Setup the loading logger
loading_logger = logging.getLogger('Loadingfin')

if not any(isinstance(handler, logging.FileHandler) for handler in loading_logger.handlers):
    fh_loading = logging.FileHandler('loadingfin.log')
    fh_loading.setLevel(logging.INFO)
    formatter_loading = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh_loading.setFormatter(formatter_loading)
    loading_logger.addHandler(fh_loading)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

error_files = []  # List to store the paths of files that cause errors

def load_model(path):
    try:
        with open(path, 'rb') as f:
            model = torch.load(f, map_location=device)
        loading_logger.info(f'Successfully loaded model from {path}')
        return model
    except Exception as e:
        loading_logger.error(f'Error loading model from {path}: {e}')
        return None


for runnr in ['runnr1', 'runnr2', 'runnr3', 'runnr4', 'runnr5']:
    path="/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/baselines/baseline-rn18-cifar10-b128/runnr1/rn18_18_dNone_128_adam_l1e-05_g0.0_w0.0_sFalse/cnn_final.pth"
    loading_logger.info(f'Trying to load model from {path}')
    
    model = load_model(path)
    if model is not None:
        break
    else:
        error_files.append(path)
        continue  # Skip to the next iteration if no model could be loaded

model.to(device)
correct = 0
total = 0

with torch.no_grad():
    for i in [1, 2, 3]:
        timers.sleep(60)
        now = datetime.now()
        sec_wait = 60 - now.second
        timers.sleep(sec_wait)

        inference_logger.info(f'start-inf-base-cif-ind{i}' )
        for s in range(60):
            t = tqdm(testloader, total=int(len(testloader)))
            for s, data in enumerate(t):
                images, labels = data
                images = images.to(device)  # Move input data to the same device as the model
                labels = labels.to(device)  # Move labels to the same device as the model
                outputs = model(images)  # calculate outputs by running images through the network
                _, predicted = torch.max(outputs.data, 1)  # the class with the highest energy is what we choose as prediction
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        inference_logger.info(f'end-inf-base-cif-ind{i}')

# Clear memory after each iteration
del model
torch.cuda.empty_cache()

# Save the list of error files to a text file
with open('error_files.txt', 'w') as f:
    for item in error_files:
        f.write(f"{item}\n")

print("List of files that caused errors has been saved to 'error_files.txt'.")


