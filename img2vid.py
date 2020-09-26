import os
import argparse
from glob import glob
import cv2
import numpy as np

'''
python img2vid.py --img data/single_tracking/OTB/Ironman/img --gt data/single_tracking/OTB/Ironman/groundtruth_rect.txt --fps 30
python img2vid.py --img data/single_tracking/OTB/Jump/img --gt data/single_tracking/OTB/Jump/groundtruth_rect.txt --fps 30
python img2vid.py --img data/single_tracking/OTB/Biker/img --gt data/single_tracking/OTB/Biker/groundtruth_rect.txt --fps 20
python img2vid.py --img data/single_tracking/OTB/Trans/img --gt data/single_tracking/OTB/Trans/groundtruth_rect.txt --fps 30
python img2vid.py --img data/single_tracking/LASOT/chameleon/chameleon-1/img --gt data/single_tracking/LASOT/chameleon/chameleon-1/groundtruth.txt --fps 20
python img2vid.py --img data/single_tracking/LASOT/crab/crab-11/img --gt data/single_tracking/LASOT/crab/crab-11/groundtruth.txt --fps 20
                   
python img2vid.py --img data/multi_tracking/MOT17/train/MOT17-02-FRCNN/img1 --fps 30
python img2vid.py --img data/multi_tracking/MOT17/train/MOT17-04-FRCNN/img1 --fps 30
python img2vid.py --img data/multi_tracking/MOT17/train/MOT17-09-FRCNN/img1 --fps 30
python img2vid.py --img data/multi_tracking/MOT17/train/MOT17-11-FRCNN/img1 --fps 30

python img2vid.py --img frame --fps 30

python img2vid.py --img data/multi_tracking/MOT20/train/MOT20-01/img1 --fps 25
python img2vid.py --img data/multi_tracking/MOT20/train/MOT20-02/img1 --fps 25
python img2vid.py --img data/multi_tracking/MOT20/train/MOT20-03/img1 --fps 25
python img2vid.py --img data/multi_tracking/MOT20/train/MOT20-05/img1 --fps 25

'''

def get_frames(img_dir):
    print('reading img')
    #print(os.path.join(img_dir, '*.jp*'))
    images = glob(os.path.join(img_dir, '*.jp*'))
    print(os.path.join(img_dir, '*.jp*'))
    images = glob(os.path.join(img_dir, '*.jp*'))
    images = sorted(images)
    print('image length',len(images))
    #print(images)
    for img in images:
        frame = cv2.imread(img)
        yield frame

def main():
    parser = argparse.ArgumentParser(description='image to video')
    parser.add_argument('--img', type=str, help='img directory')
    parser.add_argument('--gt', default= '', type=str, help='adding gt box to video')
    #parser.add_argument('--out_dir', default= '', type=str, help= 'output dir')
    parser.add_argument('--fps', type=int, help= 'output frame per second')

    args = parser.parse_args()
    if args.gt:
        gt= []
        with open(args.gt) as file:
            for line in file:
                gt.append(eval(line))
        print(len(gt))

    #initialize cv2 output writer
    #print(os.path.join(args.img, '*.jp*'))
    #print(glob(os.path.join(args.img, '*.jp*')))

    #img = cv2.imread(glob(os.path.join(args.img, '*.jp*'))[0])
    img = cv2.imread(glob(os.path.join(args.img, '*.jp*'))[0])
    #print('max is', np.max(img))
    #print('shape is', img.shape)
    height, width, _ = img.shape
    size = (width, height)
    file_name = '_'.join([args.img.split('/')[-2], 'video.mp4'])
    out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'MP4V'), args.fps, size)
    
    a =0
    for frame in get_frames(args.img):
        if args.gt: 
            box = gt[a]
            #print(gt[a])
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 3)
        out.write(frame)
        print(a)
        a+=1

    out.release()


if __name__ == '__main__':
    main()