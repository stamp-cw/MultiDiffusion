"""
Metrics utilities including FID score calculation
"""

import torch
import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
from tqdm import tqdm

class InceptionStatistics:
    """Calculate inception statistics for FID score"""
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        
    def get_activations(self, images, batch_size=50):
        """Get inception activations for images"""
        self.model.eval()
        
        # Get the number of images
        n_images = len(images)
        n_batches = int(np.ceil(n_images / batch_size))
        
        # Get the inception features
        pred_arr = np.empty((n_images, 2048))
        
        for i in tqdm(range(n_batches), desc="Calculating inception features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_images)
            
            batch = images[start_idx:end_idx].to(self.device)
            
            # Resize images if needed
            if batch.shape[2] != 299 or batch.shape[3] != 299:
                batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Get predictions
            with torch.no_grad():
                pred = self.model(batch)[0]
            
            # If model output is not scalar, apply global spatial average pooling
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:end_idx] = pred
            
        return pred_arr

def calculate_statistics(acts):
    """Calculate mean and covariance statistics"""
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma

def calculate_fid(generated_images, real_dataset, batch_size=50, device=None):
    """Calculate FID score between generated images and real dataset"""
    inception = InceptionStatistics(device)
    
    # Get activations for generated images
    gen_acts = inception.get_activations(generated_images, batch_size)
    
    # Get activations for real images
    real_acts = []
    n_samples = len(generated_images)
    
    for i, (images, _) in enumerate(real_dataset):
        if len(real_acts) * batch_size >= n_samples:
            break
        with torch.no_grad():
            acts = inception.get_activations(images, batch_size)
            real_acts.append(acts)
    
    real_acts = np.concatenate(real_acts, axis=0)[:n_samples]
    
    # Calculate statistics
    m1, s1 = calculate_statistics(gen_acts)
    m2, s2 = calculate_statistics(real_acts)
    
    # Calculate FID score
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two multivariate Gaussians"""
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + 
            np.trace(sigma1) + 
            np.trace(sigma2) - 
            2 * tr_covmean) 