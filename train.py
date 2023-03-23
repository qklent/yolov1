import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    intersection_over_union,
    nms,
    mean_average_precision,
    plot_image_with_boxes,
    load_checkpoint,
    get_bboxes,
    save_checkpoint,
    cellboxes_to_boxes
)
from loss import YoloLoss


torch.manual_seed(123)


LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 #64
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2#2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

#normalization to improve
#torch.normalize in trancforms.Compose()
#torchvision.transforms.functional.normalize
#self.darknet to imagenet?
#torch.nn.optim �������� ��� ����������� lr https://www.youtube.com/watch?v=6CvpMOO-DB4&ab_channel=DeepLearningSchool 33:30



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_index, (x,y) in enumerate(loop):
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss)

    print(f"mean loss was: {sum(mean_loss) / len(mean_loss)}")


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE,map_location=torch.device('cpu')), model, optimizer)

    train_dataset = VOCDataset(
        "data/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    test_dataset = VOCDataset(
        "data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )
    
    for epoch in range(EPOCHS):
        if LOAD_MODEL:
            for x, y in train_loader:
                x = x.to(DEVICE)
                for idx in range(8):
                    bboxes = cellboxes_to_boxes(model(x))
                    bboxes = nms(bboxes[idx], iou_threshold=0.5, prob_threshold=0.5, box_format="midpoint")
                    plot_image_with_boxes(x[idx].permute(1,2,0).to("cpu"), bboxes)

            import sys
            sys.exit()
            
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, prob_threshold=0.4,device=DEVICE, box_format="midpoint"
        )
        mAP = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        print(f"train mAP: {mAP}")

        # if mean_average_precision > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)


        train_fn(train_loader, model, optimizer, loss_fn)

    
if __name__ == "__main__":
    main()