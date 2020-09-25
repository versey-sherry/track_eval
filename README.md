# Evaluating Trackers

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
-[SORT](https://arxiv.org/abs/1602.00763)
Generating prediction results [Code](https://github.com/versey-sherry/sort)
-[Deep SORT](https://arxiv.org/abs/1703.07402)
Generating prediction results [Code](https://github.com/versey-sherry/deep_sort)
-[FairMOT](https://arxiv.org/abs/2004.01888)
Generating prediction results [Code](https://github.com/versey-sherry/FairMOT)

### Cell Tracking
-[Usiigaci](https://www.biorxiv.org/content/10.1101/524041v1)
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
### Cell Tracking
