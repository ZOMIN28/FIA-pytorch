import torch
from utils.readData import read_dataset
from torchvision.utils import save_image
from utils.utils import denorm
from net.resnet import ResNet18
from attack import FIAAttack
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 1
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')

# load model
model = ResNet18()
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(device)
model.eval()


for n,(data, target) in enumerate(test_loader):
    data = data.to(device)
    target = target.to(device)
    attack = FIAAttack(model=model, device=device)
    X = attack.perturb(data)

    # save adv img
    save_image(denorm(X[0].data.cpu()),"save/adv_img/"+str(n)+".png")