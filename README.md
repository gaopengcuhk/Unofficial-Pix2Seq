# Unofficial-Pix2Seq
Unofficial implementation of Pix2SEQ. Please use this code with causion. Many implemtation details are not following original paper and significantly simplified. 

# Aim
This project aims for a step by step replication of Pix2Seq starting from DETR codebase. 

# Step 1
Starting from DETR, we add bounding box quantization over normalized coordinate, sequence generator from normalized coordinate, auto-regressive decoder and training code for Pix2SEQ.

## How to use?
Install packages following original DETR and command line is same as DETR.

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path ../../data/coco/
```

## Released at 8pm, 26th, Seq
Problem to be solved : 1) better logging 2) correct padding, end of sentence, start of sentence token 3) efficient padding 4) better organization of code 5) fixed order of bounding box 6) shared dictionary between position and category

## Released at 10pm, 26th, Seq
Problem to be solved: 1) better organization of code 2) fixed order of bounding box

# Step 2
Finish inference code of pix2seq and report performance on object detection benchmark.

# Step 3
Add tricks proposed in Pix2SEQ like droplayer, bounding box augmentation and so on.



# Acknowledegement 
This codebase heavily borrow from DETR, CART, minGPT and Fairseq and motivated by the method explained in Pix2Seq
