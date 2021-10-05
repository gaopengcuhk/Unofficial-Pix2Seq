import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw,ImageFont

from models import build_model

import datasets.transforms as T
from util import box_ops

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")


    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    #not used
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")


    #image input path
    parser.add_argument('--img_path', default="", type=str, help='path for the image to be detected')
    return parser


LABEL_MAP = {
    0: "unlabeled",1: "person",2: "bicycle",3: "car",4: "motorcycle",5: "airplane",6: "bus",7: "train",8: "truck",9: "boat",
    10: "traffic",11: "fire",12: "street",13: "stop",14: "parking",15: "bench",16: "bird",17: "cat",18: "dog",19: "horse",
    20: "sheep",21: "cow",22: "elephant",23: "bear",24: "zebra",25: "giraffe",26: "hat",27: "backpack",28: "umbrella",29: "shoe",
    30: "eye",31: "handbag",32: "tie",33: "suitcase",34: "frisbee",35: "skis",36: "snowboard",37: "sports",38: "kite",39: "baseball",
    40: "baseball",41: "skateboard",42: "surfboard",43: "tennis",44: "bottle",45: "plate",46: "wine",47: "cup",48: "fork",49: "knife",
    50: "spoon",51: "bowl",52: "banana",53: "apple",54: "sandwich",55: "orange",56: "broccoli",57: "carrot",58: "hot",59: "pizza",
    60: "donut",61: "cake",62: "chair",63: "couch",64: "potted",65: "bed",66: "mirror",67: "dining",68: "window",69: "desk",
    70: "toilet",71: "door",72: "tv",73: "laptop",74: "mouse",75: "remote",76: "keyboard",77: "cell",78: "microwave",79: "oven",
    80: "toaster",81: "sink",82: "refrigerator",83: "blender",84: "book",85: "clock",86: "vase",87: "scissors",88: "teddy",89: "hair",
    90: "toothbrush",91: "hair",92: "banner"
}
def main(args):


    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    #read image
    image = Image.open(args.img_path)
    image = image.convert('RGB')

    w_ori,h_ori = image.size
    # image = np.array(image).astype(np.uint8)

    #transform
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform = T.Compose([
        T.RandomResize([800], max_size=512),
        normalize,
    ])

    image_new = transform(image,None)

    c,h,w = image_new[0].shape
    image_new = image_new[0].view(1,c,h,w).to(device)
    seq = torch.ones(1, 1).to(device,dtype=torch.long) * 2001
    model.eval()

    # get predictions
    output = model(image_new,seq)

    #decode bbox
    output, value = output
    output = output[0]
    value = value[0]
    for i in range(101):
        if output[i] == 2000:
            break
    if output[1:i].shape[0] == 99:
        i = 101
    output = output[1:i].reshape(-1, 5)
    box = output[:, :4].clip(0, 999).float() / (1000 - 1)
    box = box_ops.box_cxcywh_to_xyxy(box)
    label = output[:, 4].unsqueeze(-1) - 1500

    scale_fct = torch.tensor([w_ori, h_ori, w_ori, h_ori],dtype=torch.float).unsqueeze(0).to(device)
    box = scale_fct * box
    value = value[:(i - 1)].reshape(-1, 5)[:, -1]
    result ={'scores': value.detach().cpu().numpy(),
             'labels': label.squeeze(-1).detach().cpu().numpy(),
             'boxes': box.detach().cpu().numpy()}
    print(result,flush=True)

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)
    for id in range(result['labels'].shape[0]):
        x_min,y_min,x_max,y_max = result['boxes'][id]
        draw.line([(x_min, y_min), (x_min, y_max), (x_max, y_max),
                   (x_max, y_min),(x_min, y_min)], width=2, fill=(0,0,255))

        label = LABEL_MAP[result['labels'][id]]
        draw.text((x_min, y_min), label, (255,255,0), font=font)

    image.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
