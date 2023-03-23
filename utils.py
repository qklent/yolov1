import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#47 ������
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """Implement the intersection over union (IoU) between box1 and box2 
    
    Parameters:
        boxes_preds (torch.tensor) -- (BATCH_SIZE, 4) list of predicted bboxes
        boxes_labels (torch.tensor) -- (BATCH_SIZE, 4) list of predicted bboxes
        box_format (str) -- "midpoint"/"corners" if boxes (x, y, width, height) or (x1, y1, x2, y2)
                                                         x, y -- center coords
                                                         x1, y1 -- up left coords
                                                         x2, y2 -- bottom right coords
    return iou (torch.tensor) -- (BATCH_SIZE) list of intersection over union for all bboxes in batch
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - (intersection) + 1e-6)


def nms(
        bboxes, 
        iou_threshold,
        prob_threshold,
        box_format="corners"
):
    """
    implementation of non-maximum suppression algorithm i.e. takes the bbox with highest probability for each object

    Parameters:
        bboxes (list) -- list of lists containing bboxes specified as [num_class, prob_of_this_class, x1, y1, x2, y2]
        iou_threshold (int) -- intersection over union threshold
        prob_threshold (int) -- probability threshold
        box_format (str) -- "midpoint"/"corners" if boxes (x, y, width, height) or (x1, y1, x2, y2)
                                                         x, y -- center coords
                                                         x1, y1 -- up left coords
                                                         x2, y2 -- bottom right coords

    return list of lists with bboxes

    """
    assert type(bboxes) == list
    # ����������� ��������� ������, ��� �� ����� ������ �����
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=1)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box 
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.Tensor(chosen_box[2:]),
                torch.Tensor(box[2:]),
                box_format=box_format
            ) 
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


def mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=0.5,
        box_format="corners",
        num_classes=20
):
    """
    implementation of mean average precision metric
    
    Parameters:
        pred_boxes (list) list of lists [[train_index, class_pred, prob_score, x1, x2, y1, y2], ...] train_index - number  of image this bbox comes from
        true boxes (list) list of lists with target bboxes
        iou_threshold (int) -- intersection over union threshold
        box_format (str) -- "midpoint"/"corners" if boxes (x, y, width, height) or (x1, y1, x2, y2)
                                                    x, y -- center coords
                                                    x1, y1 -- up left coords
                                                    x2, y2 -- bottom right coords
        num_classes (int) -- number of classes
    return mean_average_precision for certain interesction over union threshold
    """
    average_precision = []

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_index, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for index, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = index

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_index] == 0:
                    TP[detection_index] = 1
                    amount_bboxes[detection[0]][best_gt_index] = 1
                else:
                    FP[detection_index] = 1

            else:
                FP[detection_index] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + 1e-6)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precision.append(torch.trapz(precisions, recalls))
    return sum(average_precision) / len(average_precision)



def get_bboxes(
    loader,
    model,
    iou_threshold,
    prob_threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    """
    return all pred bboxes and all target bboxes
    """
    
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predictions = model(x)
        
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
                box_format=box_format,
            )

            
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    convert cell's x, y, width, height coords to image's x, y, width, height
        Parameters:
            S (int) -- image's split size
            predictions (tensor) -- shape = (number_of_examples, S * S * (C + B * 5)) 
                where C -- number of classes, B -- numbrer of bboxess
    return tensor with shape(batch_size, S, S, 5) where S -- image's split size 
            and last dim -- [predicted_class, confidence, image's x, image's y, image's width, image's height]
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    """

    """
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_index in range(out.shape[0]):
        bboxes = []

        for bbox_index in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_index, bbox_index, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def plot_image_with_boxes(image, boxes):
    """
    plots predicted bboxes on the image
    """
    image = np.array(image)
    height, width, _ = image.shape

    fig, ax = plt.subplots(1)

    ax.imshow(image)

    for box in boxes:
        box = box[2:]
        assert len(box) == 4
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])