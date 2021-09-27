# Unofficial-Pix2seq: A Language Modeling Framework for Object Detection
Unofficial implementation of Pix2SEQ. Please use this code with causion. Many implemtation details are not following original paper and significantly simplified. 

# Aim
This project aims for a step by step replication of Pix2Seq starting from DETR codebase. 

# Step 1
Starting from DETR, we add bounding box quantization over normalized coordinate, sequence generator from normalized coordinate, auto-regressive decoder and training code for Pix2SEQ.

## How to use?
Install packages following original DETR and command line is same as DETR.

By setting image size to 512, each epoch takes 3 minutes on 8 A100 GPU.

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path ../../data/coco/
```

## Released at 8pm, 26th, Seq
Problem to be solved : 1) better logging 2) correct padding, end of sentence, start of sentence token 3) efficient padding 4) better organization of code 5) fixed order of bounding box 6) shared dictionary between position and category

## Released at 10pm, 26th, Seq
Problem to be solved: 1) better organization of code 2) fixed order of bounding box

# Step 2
Finish inference code of pix2seq and report performance on object detection benchmark. Note that we are going to write an inefficent greedy decoding. The progress can be significantly accelerated by following cache previous state in Fairseq. The quality can be improved by nucleus sampling and beam search. We leave these complex but engineering tricks for future implementation and keep the project as simple as possible for understanding language modeling object detection.

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path ../../data/coco/  --eval --resume checkpoint.pth --batch_size 4
```

After 30 epoches training, our replication of pix2seq can achieve 12.1 mAP on MSCOCO. We will release better checkpoint later. 

COCO bbox detection val5k evaluation results:
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.239
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.107
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.007
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.091
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.267
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.350
```

## Observation
(1). The sequence is tend to generate End of Sentence(EOS) early. After generating EOS token, langauge modeling will still genrate boudning box. (2). Repeatable sequence which is a common problem in seq2seq modeling. 

## Full inference code with model checkpoint will be released at 28th, Seq
1). Add sequence likelihood evaluationn on validation dataset


# Step 3
Add tricks proposed in Pix2SEQ like droplayer, bounding box augmentation, multiple crop augmentation and so on.



# Acknowledegement 
This codebase heavily borrow from DETR, CART, minGPT and Fairseq and motivated by the method explained in Pix2Seq
