# Evaluating Trackers

## Installation
* Clone this repo

creating an conda environment from an ``environment.yml`` file
```
conda env create -f environment.yml
```
The first line of the ``yml`` file sets the new environment's name. For more details see [Conda Documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

Activate the new environment
```
conda activate track_eval
```
Verify the new environment
```
conda env list
```

## List of Papers
### Single Object Tracking
- [SiamCAR](https://arxiv.org/abs/1911.07241v2) 
Generating prediction results [Code](https://github.com/versey-sherry/pysot)
- [SiamMask](https://arxiv.org/abs/1812.05050)
Generating prediction results [Code](https://github.com/versey-sherry/pysot)
- [SiamRPN++](https://arxiv.org/abs/1812.11703)
Generating prediction results [Code](https://github.com/versey-sherry/pysot)
- [Goturn](https://arxiv.org/abs/1604.01802)
Generating prediction results [Code](https://github.com/versey-sherry/pygoturn)

### Multiple Object Tracking
- [SORT](https://arxiv.org/abs/1602.00763)
Generating prediction results [Code](https://github.com/versey-sherry/sort)
- [Deep SORT](https://arxiv.org/abs/1703.07402)
Generating prediction results [Code](https://github.com/versey-sherry/deep_sort)
- [FairMOT](https://arxiv.org/abs/2004.01888)
Generating prediction results [Code](https://github.com/versey-sherry/FairMOT)

### Cell Tracking
- [Usiigaci](https://www.biorxiv.org/content/10.1101/524041v1)
Generating prediction results [Code](https://github.com/versey-sherry/Usiigaci)

## List of Datasets/Benchmarks
### Single Object Tracking
- [TB-100](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html)
- [GOT-10k](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html)
- [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)
- [UAV123](https://uav123.org/)

### Multiple Object Tracking
- [MOT15](https://motchallenge.net/data/2D_MOT_2015/)
- [MOT16](https://motchallenge.net/data/MOT16/)
- [MOT17](https://motchallenge.net/data/MOT17/)
- [MOT20](https://motchallenge.net/data/MOT20/)
### Cell Tracking
- [CTC](http://celltrackingchallenge.net/datasets/)
- [DeepCell](https://www.deepcell.org/data)
- [C2C12](https://osf.io/ysaq2/)

## Evaluation Metrics
### Single Object Tracking
- **Accuracy**
Accuracy is measured with the Intersection Over Union for each frame between ground-truth and predicted bounding box.
The accuracy is the average overlap between the predicted and ground truth bounding boxes during successful tracking periods. 
- **Robustness**
Robustness measures how many times the tracker loses the target (fails) during tracking.
- **Expected average overlap (EAO)**
EAO is the expected overlap to attain on a large collection of short-term sequences with the same visual properties as the given dataset.

More information can be found in [The sixth Visual Object Tracking VOT2018 challenge results](http://prints.vicos.si/publications/365)

```
python evaluation.py --prediction_dir result_example/sot_results/DATASET_NAME/MODEL_NAME \
--video_dir DATASET_DIR/DATASET_NAME \
--gt_name groundtruth_rect.txt \
--evaluation_type single \
--threshold 0.5 \
--writeout Ture
```
### Multiple Object Tracking
- **The multiple object tracking accuracy(MOTA)**
Ratio of object configuration errors made by the tracker, false positives, misses, mismatches, over all frames.
- **The multiple object tracking precision (MOTP)**
The total error in estimated position for matched object-hypothesis pairs over all frames, averaged by the total number of matches made.
- **Precision**
Number of detected objects over sum of detected and false positives.
- **Recall**
Number of detections over number of objects.
- **Number of Matches** 
Total number matches. 
- **Number of Switches**
Total number of track switches.
- **Number of False Positive**
Total number of false positives (false-alarms).
- **Number of Misses**
Total number of misses.
- **Mostly Tracked**
Number of objects tracked for at least 80 percent of lifespan.
- **Mostly Lost**
Number of objects tracked less than 20 percent of lifespan.

Evaluation is implemented with `py-motmetrics`, more information can be found in [py-motmetrics](https://github.com/cheind/py-motmetrics), Bernardin, Keni, and Rainer Stiefelhagen. [Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics](https://doi.org/10.1155/2008/246309) and [MOT20](https://motchallenge.net/results/MOT20/)

```
python evaluation.py --prediction_dir result_example/mot_results/DATASET_NAME/MODEL_NAME \
--video_dir DATASET_DIR/DATASET_NAME \
--gt_name gt/gt.txt \
--evaluation_type multiple \
--threshold 0.5 \
--writeout Ture
```

### Cell Tracking
- **Precision**
Number of detected objects over sum of detected and false positives.
- **Recall**
Number of detections over number of objects.
- **Number of Matches** 
Total number matches. 
- **Number of Switches**
Total number of track switches.
- **Number of False Positive**
Total number of false positives (false-alarms).
- **Number of Misses**
Total number of misses.
- **Mostly Tracked**
Number of objects tracked for at least 80 percent of lifespan.
- **Mostly Lost**
Number of objects tracked less than 20 percent of lifespan.

- **Segmentation measure (SEG)** [To be implemented]
Segmentation (SEG) accuracy measurement is to understand how well the segmented cells match the actual cell regions. The segmentation measure is a computed with Jaccard similarity index.
- **Tracking measure (TRA)** [To be implemented]
The goal of the tracking (TRA) measurement is to evaluate the ability of the tracking algorithms to detect the cells and follow them in time. This metric represents the predicted and the ground truth tracks as graphs and computes the number of steps required to match both graphs.

Evaluation is implemented with `py-motmetrics`, more information can be found in [py-motmetrics](https://github.com/cheind/py-motmetrics). Evaluation specific to cell tracking can be found in Maška, Martin, Vladimír Ulman, David Svoboda, Pavel Matula, Petr Matula, Cristina Ederra, Ainhoa Urbiola, et al. [A Benchmark for Comparison of Cell Tracking Algorithms](https://doi.org/10.1093/bioinformatics/btu080)

```
python evaluation.py --prediction_dir result_example/cell_results/DATASET_NAME/MODEL_NAME \
--video_dir DATASET_DIR/DATASET_NAME \
--gt_name ANNOTATION.XML \
--evaluation_type cell \
--threshold 15 \
--writeout Ture

```
