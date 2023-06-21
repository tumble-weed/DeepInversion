import argparse
import torch
import torchvision
import skimage.io
import glob
from PIL import Image
vgg_mean = (0.485, 0.456, 0.406)
vgg_std = (0.229, 0.224, 0.225)
def get_vgg_transform(size=224):
    vgg_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size),
            torchvision.transforms.Normalize(mean=vgg_mean,std=vgg_std),
            ]
        )
    return vgg_transform

def evaluate_network(x,target_ids):
    model(x)
    prob_correct  = <TODO>
    acc_correct = <TODO>
    avg_prob_correct = prob_correct
    avg_acc_correct = acc_correct
    return avg_prob_correct,avg_acc_correct

def calculate_improvement():
    metrics = argparse.Namespace()
    metrics.n_improved
    metrics.relative_increase
    metrics.relative_decrease
    metrics.n_worse = 
        
    return vars(metrics)

def one_step_evaluate(modelname,
                      pretrained=True,
                      validation_images=None):
    validation_images = <TODO>
    model = torchvision.models.__dict__[modelname](pretrained=pretrained)
    model.eval()
    optimizer = <TODO>
    initial_stats = evaluate_network(<TODO evalute_network before training>)
    pred_score = model(x)
    loss =<TODO>
    trends['loss'].append(loss.item())
    loss.backward()
    optimizer.step()
    stats = evaluate_network(<TODO evaluate_network after training>)
    metrics = calculate_improvement(initial_stats,stats,<TODO argument list>)
    return metrics

def load_deepinversion_images(root_dir):
    impaths = glob.glob(os.path.join(root_dir,'*.png'))
    images = []
    for impath in impaths:
        im = skimage.io.imread(impath)
        images.append(im)
    return images

def transform_images(images,transform):
    refs = []
    for im in images:
        ref = transform(Image.fromarray(im))
        refs.append(ref)
    refs = torch.stack(refs,dim=0)
    return refs
def get_deepinversion_tensors(root_dir):
    images = load_deepinversion_images(root_dir)
    transform = get_vgg_transform()
    tensors = transform_images(images,transform)
    return tensors
def test_images():
    root_dir = <TODO>
    tensors = get_deepinversion_tensors(root_dir)
    import ipdb;ipdb.set_trace()
def load_validation_images(target_ids):
    try:
        from find_imagenet_images import find_images_for_class_id
    except:
        import ipdb;ipdb.set_trace()
    images_for_class = {}
    for target_id in target_ids:
        image_paths = find_images_for_class_id(153)
        images_for_class[target_id] = image_paths
    return images_for_class
def load_validation_tensors(target_ids):
    images_for_class = load_validation_images(target_ids)
    tensors_for_class = {}
    transform = get_vgg_transform()
    for target_id, impaths in images_for_class.items():
        images = []
        for impath in impaths:
            im = skimage.io.imread(impath)
            images.append(im)        
        tensors = transform_images(images,transform)
        tensors_for_class[target_id] = tensors
    return tensors_for_class
def run():
    
    pass