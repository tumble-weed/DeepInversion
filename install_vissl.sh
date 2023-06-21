pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/{version_str}/download.html
git clone --recursive https://github.com/facebookresearch/vissl.git
cd vissl/
git checkout v0.1.6
git checkout -b v0.1.6
#pip install --progress-bar off -r requirements.txt
#pip install opencv-python
pip uninstall -y classy_vision
pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d
pip uninstall -y fairscale
pip install fairscale@https://github.com/facebookresearch/fairscale/tarball/df7db85cef7f9c30a5b821007754b96eb1f977b6
pip install -e .[dev]
# https://stackoverflow.com/questions/66610378/unencryptedcookiesessionfactoryconfig-error-when-importing-apex
pip uninstall apex
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
pip uninstall numpy
pip install numpy
wget -q -O vissl/test_image.png https://raw.githubusercontent.com/facebookresearch/vissl/master/.github/logo/Logo_Color_Light_BG.png
# fix kinetics400 error in
# /opt/conda/envs/pytorch3d/lib/python3.9/site-packages/classy_vision/dataset/classy_kinetics400.py
# from torchvision.datasets.kinetics import Kinetics as Kinetics400
