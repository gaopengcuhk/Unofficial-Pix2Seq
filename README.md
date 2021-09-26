# Unofficial-Pix2Seq
Unofficial implementation of Pix2SEQ. Please use this code with causion. Many implemtation details are not following original paper and significantly simplified. 

# Aim
This project aims for a step by step replication of Pix2Seq starting from DETR codebase. Unoffical-pix2seq will implement the idea step-by-step and perform ablation study over ideas proposed in pix2seq.

# Step 1
Starting from DETR, we add bounding box quantization over normalized coordinate, sequence generator from normalized coordinate, auto-regressive decoder and training code for Pix2SEQ.

# Step 2
Finish inference code of pix2seq and report performance on object detection benchmark.

# Step 3
Add tricks proposed in Pix2SEQ like droplayer, bounding box augmentation and so on.



# Acknowledegement 
This codebase heavily borrow from DETR and CATR and motivated by the method explained in Pix2Seq
