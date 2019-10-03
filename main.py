import utils
import torch
import sys, os
from torchvision import transforms
from artnet import ARTNet
from PIL import Image

labels = ['nonporn', 'porn']
assert len(sys.argv) == 3, 'Insufficient number of argument'

v = utils.extract_frames(sys.argv[2], 'samples')
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomCrop((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

frames = [Image.open(os.path.join(v, f)) for f in os.listdir(v)]
frames = [transform(f) for f in frames]
tensors = []
for i in range(0, len(frames),16):
    tensors.append(torch.stack(frames[i:i+16]))

model = ARTNet()
model.load_state_dict(torch.load(sys.argv[1]))
model = model.to('cuda')

for tensor in tensors:
    tensor = tensor.to('cuda')
    tensor = tensor.unsqueeze(0)
    result = model(tensor)
    print(result)
    label = result.argmax(1)

