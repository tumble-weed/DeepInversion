import sys
class DeepSphereModel:
    def __init__(self,):
        super().__init__()
        sys.path.append(/'root/evaluate-saliency-4/fong-invert/DeepInversion/deepsphere-pytorch/')
        
        from deepsphere.models.spherical_unet.unet_model import SphericalUNet
        pooling_class = 'icosahedron'
        n_pixels = 10242 # because 768*1152//66 -
        depth = 6
        laplacian_type = 'combinatorial'
        
        self.unet = SphericalUNet(
            pooling_class, 
            n_pixels, 
            depth, 
            laplacian_type, 
            kernel_size)
    def forward(self,sphere_image):
        as_image = self.unet(sphere_image)
        return 
