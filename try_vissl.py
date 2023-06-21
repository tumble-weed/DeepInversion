
import vissl
import tensorboard
import apex
import torch
# !wget -q -O /content/resnet_simclr.torch https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch
#======================================================================+
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict

# Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
# All other options override the simclr_8node_resnet.yaml config.

cfg = [
#   'config=pretrain/simclr/simclr_8node_resnet.yaml',
  'config=pretrain/simclr/simclr_8node_resnet.yaml',
  'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=vissl/resnet_simclr.torch', # Specify path for the model weights.
  'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
  'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
  'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
  'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
  'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]' # Extract only the res5avg features.
]

# Compose the hydra configuration.
cfg = compose_hydra_configuration(cfg)
# Convert to AttrDict. This method will also infer certain config options
# and validate the config is valid.
_, cfg = convert_to_attrdict(cfg)
     
     

#======================================================================+
from vissl.models import build_model

model = build_model(cfg.MODEL, cfg.OPTIMIZER)
     
#======================================================================+

from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights

# Load the checkpoint weights.
weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
import ipdb; ipdb.set_trace()

# Initializei the model with the simclr model weights.
init_model_from_consolidated_weights(
    config=cfg,
    model=model,
    state_dict=weights,
    state_dict_key_name="classy_state_dict",
    skip_layers=[],  # Use this if you do not want to load all layers
)

print("Weights have loaded")
#======================================================================+

from PIL import Image
import torchvision.transforms as transforms

def extract_features(path):
  image = Image.open(path)

  # Convert images to RGB. This is important
  # as the model was trained on RGB images.
  image = image.convert("RGB")

  # Image transformation pipeline.
  pipeline = transforms.Compose([
      transforms.CenterCrop(224),
      transforms.ToTensor(),
  ])
  x = pipeline(image)

  features = model(x.unsqueeze(0))

  features_shape = features[0].shape

  print(f"Features extracted have the shape: { features_shape }")

extract_features("vissl/test_image.png")
import ipdb; ipdb.set_trace()