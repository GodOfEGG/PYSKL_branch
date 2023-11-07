import sys
import types
import pickle
import configparser
from pathlib import Path
import numpy as np


def parse_args():
    config_file = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(config_file)

    # Path Argument
    args = types.SimpleNamespace()
    args.anno_dir = Path(config['argument']['anno_dir'])
    args.ignore_file = config['argument']['ignore_file']
    args.label_file = config['argument']['label_file']
    args.output_file = config['argument']['output_file']
    args.split_file_train = config['argument']['split_file_train']
    args.split_file_val = config['argument']['split_file_val']
    args.split_file_test = config['argument']['split_file_test']

    # Other Argument
    args.num_frames = int(config['argument']['num_frames'])
    args.skeleton_format = config['argument']['skeleton_format']

    return args


def main(args):
    ## Read label file
    label_list = None
    with open(args.label_file, 'r') as label_f:
        data = label_f.read()
        label_list = data.split('\n')
    print(label_list)


    # Read ignore list to remove bad data
    ignore_file_list = []
    with open(args.ignore_file, 'r') as ignore_f:
        data = ignore_f.read()
        ignore_file_list = data.split('\n')


    ## Write annotation list
    final_dict={}
    anno_list = []
    num_clip_dict={}
    for pkl_filepath in args.anno_dir.glob('*.pkl'):
        if pkl_filepath.stem in ignore_file_list:
            continue

        with open(pkl_filepath, 'rb') as pkl_f:
            data = pickle.loads(pkl_f.read()) 

            keypoints = None
            if args.skeleton_format == 'coco':
                keypoints = data['keypoints3d']
            elif args.skeleton_format == 'smpl':
                keypoints = data['smpl_poses'].reshape(-1, 24, 3).astype(np.float64)
            assert (keypoints is not None), "Invalid skeleton format"

            new_keypoints = []
            for x in keypoints:
                if False in np.isfinite(x):
                    continue
                new_keypoints.append(x)
            keypoints = np.array(new_keypoints)
            assert False not in np.isfinite(keypoints)

            total_len = keypoints.shape[0]
            if args.num_frames > total_len:
                continue

            num_clip=1
            for i in range(0, total_len, args.num_frames):
                start_i = i if i+args.num_frames <= total_len else total_len-args.num_frames
                end_i = start_i+args.num_frames
                anno_dict = {}
                anno_dict['keypoint'] = np.expand_dims(keypoints[start_i:end_i], axis=0)
                anno_dict['frame_dir'] = pkl_filepath.stem + f'_{num_clip:02d}'
                anno_dict['total_frames'] = args.num_frames
                anno_dict['label'] = label_list.index(anno_dict['frame_dir'][1:3])
            
                anno_list.append(anno_dict)
                num_clip+=1
            num_clip_dict[pkl_filepath.stem] = num_clip-1

    final_dict['annotations'] = anno_list

    ## Write split
    split_dict = {}
    with open(args.split_file_train, 'r') as train_f:
        data = train_f.read()
        org_train_list = data.split('\n')
        new_train_list = []
        for id in org_train_list:
            if id not in num_clip_dict.keys():
                continue
            num_clip = num_clip_dict[id]

            for i in range(num_clip):
                new_train_list.append(id + f'_{i+1:02d}')

        split_dict['train'] = new_train_list

    with open(args.split_file_val, 'r') as val_f:
        data = val_f.read()
        org_val_list = data.split('\n')
        new_val_list = []
        for id in org_val_list:
            if id not in num_clip_dict.keys():
                continue
            num_clip = num_clip_dict[id]

            for i in range(num_clip):
                new_val_list.append(id + f'_{i+1:02d}')

        split_dict['val'] = new_val_list

    with open(args.split_file_test, 'r') as test_f:
        data = test_f.read()
        org_test_list = data.split('\n')
        new_test_list = []
        for id in org_test_list:
            if id not in num_clip_dict.keys():
                continue
            num_clip = num_clip_dict[id]

            for i in range(num_clip):
                new_test_list.append(id + f'_{i+1:02d}')

        split_dict['test'] = new_test_list

    final_dict['split'] = split_dict


    # Dump into pickle file
    with open(args.output_file, 'wb') as out_f:
        pickle.dump(final_dict, out_f)

if __name__ == '__main__':
    args = parse_args()
    main(args)
