#  Eyes_detector

### 1. Introduction


This repo's goal is to train a model which can detect open and closed state of eyes.

I'm a newcomer to ML & DL, so I hope you can point out the mistakes and shortcomings of the code if this repo is lucky enough to be seen by you.

This is changed from [YOLOv3_Tensorflow](https://github.com/wizyoung/YOLOv3_TensorFlow), I  just use 52*52 featuremap of 
YOLOv3.Thank you, [wizyoung](https://github.com/wizyoung)!
### 2. Environment

Python version:  3

Packages:

- tensorflow >= 1.4.0 (theoretically any version that supports tf.data is ok)
- opencv-python
- tqdm

### 3. Checkpoint File download
I trained a model with 400 pictures, some of this pictures are from the Internet, some are from my own photo. So it has very low accuracy. 

You can download the checkpoint file  via [My Google Drive link](https://drive.google.com/file/d/16m4NCxFGEy2fE6ZAAT9DQ09kvyCnYuVi/view?usp=sharing) and then place it to the same directory.

### 4. Running demos

There are two demo images  under the `./data/demo_data/`. You can run the demo by:

Single image test demo:

```shell
python test_single_image.py ./data/demo_data/open.jpg
```


Some results:

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/open.jpg?raw=true)

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/close.jpg?raw=true)

### 5. Model architecture
I used part of the yolo_v3_architecture, thanks to [wizyoung](https://github.com/wizyoung) again!

You can refer to the following picture. I just deleted the upper right part which is in red box.
![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/docs/yolo_v3_architecture.png?raw=true)

### 6. Training

#### 6.1 Data preparation 

(1) annotation file

Generate `train.txt/val.txt/test.txt` files under `./data/my_data/` directory. One line for one image, in the format like `image_index image_absolute_path img_width img_height eye_1 eye_2`. eye_n format: `label_index point_x point_y`. (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).) `image_index` is the line index which starts from zero. `label_index` is in range [0, 1], 0 represents closed eye, 1 represents open eye.

For example:

```
0 xxx/xxx/a.jpg 1411 800 1 389 657 1 586 637
1 xxx/xxx/b.jpg 1411 800 0 327 634 0 510 632
...
```

You can label eyes on pictures by [HyperLabelImg](https://github.com/yujunhua/HyperLabelImg), this is changed from [zeusees](https://github.com/zeusees/HyperLabelImg), thanks!

Then, you can merge sigle label info to one file by [this file](https://github.com/yujunhua/myprivatetoolshub/blob/master/xml/mergefiles_to_txt.py)

(2)  class_names file:

Generate the `data.names` file under `./data/my_data/` directory. Each line represents a class name. Here we only have two classes, close and open.

For example:

```
close
open
```

#### 6.2 Training

Using `train.py`. The hyper-parameters and the corresponding annotations can be found in `args.py`:

```shell
CUDA_VISIBLE_DEVICES=GPU_ID python train.py
```

Check the `args.py` for more details. You should set the parameters yourself in your own specific task.

### 7. Evaluation

Using `eval.py` to evaluate the validation or test dataset. The parameters are as following:

```shell
$ python eval.py -h
usage: eval.py [-h] [--eval_file EVAL_FILE] 
               [--restore_path RESTORE_PATH]
               [--class_name_path CLASS_NAME_PATH]
               [--batch_size BATCH_SIZE]
               [--img_size [IMG_SIZE [IMG_SIZE ...]]]
               [--num_threads NUM_THREADS]
               [--prefetech_buffer PREFETECH_BUFFER]
               [--score_threshold SCORE_THRESHOLD] 
```

Check the `eval.py` for more details. You should set the parameters yourself. 

You will get the loss, recall, precision, average precision and mAP metrics results.



