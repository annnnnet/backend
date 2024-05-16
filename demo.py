import argparse
import base64
import tempfile
from PIL import Image
from flask import Flask, jsonify, request
import torch
import os, sys
import numpy as np
import progressbar
import matplotlib.pyplot as plt

from nets.FullVggCompositionNet import FullVggCompositionNet as CompositionNet
from nets.SiameseNet import SiameseNet
from datasets import data_transforms
from py_utils import dir_utils, load_utils, bboxes
# from datasets.get_test_image_list import get_test_list, get_pdefined_anchors
import cuda_model
import datasets.val_pdefined_anchors as dataset_loader

parser = argparse.ArgumentParser(description="Full VGG trained on combined dataset")
parser.add_argument('--l1', default=1024, type=int)
parser.add_argument('--l2', default=512, type=int)
parser.add_argument("--gpu_id", default='1', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
parser.add_argument('--resume', '-r', default='snapshots/FullVGG-1024x512/params/best-5000-0.55-0.77.pth.tar', type=str, help='resume from checkpoint')

app = Flask(__name__)

def process_image_with_model(image, single_pass_net, pdefined_anchors, t_transform, useCuda):
    image_crops, image_bboxes = dataset_loader.Get895Crops(image, pdefined_anchors)
    print("[INFO] Processing image")
    pbar = progressbar.ProgressBar(max_value=len(image_crops))
    s_image_scores = []
    s_image_bboxes = []
    for crop_idx, (s_image_crop, s_image_bbox) in enumerate(zip(image_crops, image_bboxes)):
        pbar.update(crop_idx)
        t_image_crop = t_transform(s_image_crop)

        if useCuda:
            t_image_crop = t_image_crop.cuda()

        t_input = torch.autograd.Variable(t_image_crop)
        t_output = single_pass_net(t_input.unsqueeze(0))
        s_image_scores.append(t_output.data.cpu().numpy()[0][0])
        s_image_bboxes.append(s_image_bbox)

    idx_sorted = np.argsort(-np.array(s_image_scores))
    s_image_scores_sorted = [s_image_scores[i] for i in idx_sorted]
    s_image_bboxes_sorted = [s_image_bboxes[i] for i in idx_sorted]
    s_scores_nms, s_bboxes_nms, _ = bboxes.bboxes_nms(s_image_scores_sorted, s_image_bboxes_sorted, nms_threshold=0.6)

    return s_scores_nms[:5], s_bboxes_nms[:5]



@app.route('/process_image', methods=['POST'])
def process_image():
    image_data = request.form.get('image')
    if not image_data:
        response = jsonify({'error': 'No image data found'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name
        print(temp_file_path)

    scores, bboxes = process_image_with_model(temp_file_path, single_pass_net, pdefined_anchors, t_transform, useCuda)


    response = jsonify({'success': True, 'scores': scores, 'bboxes': bboxes})
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    os.remove(temp_file_path)
    
    return response

if __name__ == '__main__':
    args = parser.parse_args()
    identifier = 'FullVGG'

    running_name = '{:s}-{:d}x{:d}'.format(identifier, args.l1, args.l2)

    save_dir = dir_utils.get_dir('./snapshots/{:s}'.format(running_name))
    save_file = os.path.join(save_dir, '{:s}.txt'.format(running_name))
    param_save_dir = dir_utils.get_dir(os.path.join(save_dir, 'params'))

    ckpt_file = 'model_params/net.pth.tar'
    if ckpt_file is not None:
        if not os.path.isfile(ckpt_file):
            print("CKPT {:s} NOT EXIST").format(ckpt_file)
            sys.exit(-1)
        print("load from {:s}".format(ckpt_file))

        single_pass_net = CompositionNet(pretrained=False, LinearSize1=args.l1, LinearSize2=args.l2)
        siamese_net = SiameseNet(single_pass_net)
        ckpt = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        model_state_dict = ckpt['state_dict']
        siamese_net.load_state_dict(model_state_dict)
    else:
        single_pass_net = CompositionNet(pretrained=True, LinearSize1=args.l1, LinearSize2=args.l2)
        siamese_net = SiameseNet(single_pass_net)

    print("Number of Params in {:s}\t{:d}".format(identifier, sum([p.data.nelement() for p in single_pass_net.parameters()])))

    useCuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)
    single_pass_net = cuda_model.convertModel2Cuda(single_pass_net, args.gpu_id, args.multiGpu)
    single_pass_net.eval()

    # pdefined_anchors = get_pdefined_anchors()
    t_transform = data_transforms.get_val_transform(224)

    # image_list = get_test_list()
    # image_list = image_list[0:1]
    # print(image_list)

    # n_images = len(image_list)
    # print("Number of Images:\t{:d}".format(len(image_list)))

    image_annotation = {}

    # for image_path in image_list:
    #     scores, bboxes = process_image_with_model(image_path, single_pass_net, pdefined_anchors, t_transform, useCuda)
    #     image_name = os.path.basename(image_path)
    #     image_annotation[image_name] = {'scores': scores, 'bboxes': bboxes}

    print("Done Computing, saving to {:s}".format(save_file))
    load_utils.save_json(image_annotation, save_file)

    app.run(host='0.0.0.0', port=5000, debug=True)