# Standard library imports
from typing import Optional, Any, Union, Tuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn
from torch import Tensor
import math
from sklearn.mixture import BayesianGaussianMixture
import torch.distributions as D
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA   

# Local imports
from .conv import DoubleConv, SingleConv
from .linear import DoubleLinear, SingleLinear
from .utils import (
    get_downsampling,
    get_dropout,
    get_upsampling,
    get_projection,
    is_real_mode,
)
from .learn_poly_sampling.layers import PolyphaseInvariantUp2D, PolyphaseInvariantDown2D
from cvnn.utils import setup_logging

logger = setup_logging(__name__)


class Down(nn.Module):
    """
    Downscaling block for U-Net architecture.

    Applies downsampling (pooling or strided convolution) followed by
    multiple convolution blocks with optional dropout.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        normalization: Type of normalization ('batch', 'instance', etc.)
        downsampling: Downsampling method ('maxpool', 'avgpool', etc.)
        downsampling_factor: Factor for spatial dimension reduction
        residual: Whether to use residual connections
        dropout: Dropout probability (0.0 to disable)
        num_blocks: Number of successive convolution blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        layer_mode: str,
        num_blocks: int,
        projection: Optional[str] = None,
        projection_config: Optional[dict] = None,
        normalization: str = None,
        downsampling: str = None,
        downsampling_factor: int = 2,
        kernel_size: int = 3,
        stride: Union[int, str] = 1,
        residual: bool = False,
        dropout: float = 0.0,
        gumbel_softmax: str = None,
    ) -> None:
        """Initialize downscaling block."""
        super().__init__()

        # --- LPD / Downsampling Logic ---
        if downsampling in ["LPD", "LPD_F"]:
            lpd_conv = DoubleConv(
                in_ch=out_channels,
                out_ch=out_channels,
                conv_mode=layer_mode,
                activation=activation,
                normalization=normalization,
                residual=residual,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        else:
            lpd_conv = None

        self.down = get_downsampling(
            downsampling=downsampling or None,
            projection=projection or None,
            projection_config=projection_config,
            factor=downsampling_factor,
            layer_mode=layer_mode,
            conv=lpd_conv,
            in_channels=out_channels,
            out_channels=out_channels,
            gumbel_softmax_type=gumbel_softmax,
        )

        # --- Dynamic Convolution Blocks ---
        # We perform `num_blocks` convolutions.
        # The first (num_blocks - 1) maintain in_channels.
        # The last one projects to out_channels.
        layers = []
        for i in range(num_blocks):
            is_last = (i == num_blocks - 1)
            # --- Stride/Padding Logic ---
            if is_last and downsampling is None:
                stride = downsampling_factor
                
            layers.append(
                DoubleConv(
                    in_ch=in_channels,
                    out_ch=out_channels if is_last else in_channels,
                    conv_mode=layer_mode,
                    activation=activation,
                    normalization=normalization,
                    residual=residual,
                    stride=stride,
                    padding=kernel_size // 2,
                    kernel_size=kernel_size,
                )
            )
        
        self.convs = nn.Sequential(*layers)
        self.dropout = get_dropout(dropout, layer_mode, spatial=True)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        # Apply sequential blocks
        x = self.convs(x)
        if isinstance(self.down, PolyphaseInvariantDown2D):
            x, prob = self.down(x, ret_prob=True)
        else:
            prob = None
            x = self.down(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x, prob

class Up(nn.Module):
    """
    Upscaling block for U-Net architecture.

    Applies upsampling (transpose convolution or interpolation) followed by
    multiple convolution blocks. Supports skip connections.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        upsampling: Upsampling method ('transpose', 'interpolate', etc.)
        skip_connections: Whether to concatenate skip connections
        normalization: Type of normalization ('batch', 'instance', etc.)
        upsampling_factor: Factor for spatial dimension increase
        residual: Whether to use residual connections
        num_blocks: Number of successive convolution blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        layer_mode: str,
        num_blocks: int,
        upsampling: Optional[str],
        skip_connection: bool = False,
        normalization: str = None,
        upsampling_factor: int = 2,
        kernel_size: int = 3,
        residual: bool = False,
        gumbel_softmax: Optional[str] = None,
    ) -> None:
        """Initialize upscaling block."""
        super().__init__()

        # Handle channel count for conv layer input
        if upsampling != "transpose":
            self.conv_adjust = DoubleConv(
                in_ch=in_channels,
                out_ch=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_mode=layer_mode,
                activation=activation,
                normalization=normalization,
                residual=residual,
            )
            self.up = get_upsampling(
                upsampling=upsampling,
                factor=upsampling_factor,
                layer_mode=layer_mode,
                in_channels=out_channels,
                out_channels=out_channels,
                gumbel_softmax_type=gumbel_softmax,
            )
        else:
            self.up = get_upsampling(
                upsampling=upsampling,
                factor=upsampling_factor,
                layer_mode=layer_mode,
                in_channels=in_channels,
                out_channels=out_channels,
                gumbel_softmax_type=gumbel_softmax,
            )
            self.conv_adjust = None
        
        in_channels = out_channels # Regardless of upsampling, after up we have out_channels as transpose conv or conv_adjust adjust the channels

        if skip_connection:
            in_channels += out_channels

        # --- Dynamic Convolution Blocks ---
        # The first (num_blocks - 1) maintain the effective in_channels (including skip).
        # The last one projects to out_channels.
        layers = []
        for i in range(num_blocks):
            is_last = (i == num_blocks - 1)

            layers.append(
                DoubleConv(
                    in_ch=in_channels,
                    out_ch=out_channels if is_last else in_channels,
                    conv_mode=layer_mode,
                    activation=activation,
                    normalization=normalization,
                    residual=residual,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
        
        self.convs = nn.Sequential(*layers)

    def forward(self, x1: Tensor, x2: Optional[Tensor] = None, prob: Optional[Tensor] = None) -> Tensor:
        """Apply upsampling and convolution."""
            
        if isinstance(self.up, c_nn.ConvTranspose2d) or isinstance(self.up, nn.ConvTranspose2d):
            x1 = self.up(x1)
        else:
            x1 = self.conv_adjust(x1)            
            if isinstance(self.up, PolyphaseInvariantUp2D):
                x1 = self.up(x1, prob=prob)
            else:
                x1 = self.up(x1)            
                
        x = concat(x1, x2)
        x = self.convs(x)
        return x
    
class FullyConnected(nn.Module):
    """
    Fully connected layer for classifier architectures.

    Compresses spatial feature maps to a lower-dimensional latent space
    representation. Supports both real and complex-valued data.

    Args:
        in_channels: Number of input channels
        latent_dim: Dimensionality of latent space
        input_size: Spatial size of input (assumes square)
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        normalization: Type of normalization ('batch', 'instance', etc.)

    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        activation: nn.Module,
        layer_mode: str = "complex",
        normalization: Optional[str] = None,
        projection: Optional[str] = None,
        projection_config: Optional[dict] = None,
        dropout: Optional[float] = None,
    ) -> None:

        """Initialize latent bottleneck layer."""
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.normalization = normalization
        self.dropout = get_dropout(dropout, layer_mode, spatial=False)

        # store mode/activation for potential lazy re-init of fc_1
        self._activation = activation
        self._normalization = normalization
        self._layer_mode = layer_mode

        # Initial fc_1 expects flattened pooled features of size `in_channels` -> latent_dim
        self.fc_1 = DoubleLinear(
            in_ch=in_channels,
            mid_ch=in_channels,
            out_ch=num_classes,
            linear_mode=layer_mode,
            activation=activation,
            normalization=normalization,
        )

        # Use real AvgPool2d for real mode and complex AvgPool2d for complex mode
        if is_real_mode(layer_mode):
            # For real tensors use PyTorch AvgPool2d
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            # For complex tensors use torchcvnn AvgPool2d
            self.avg_pool = c_nn.AdaptiveAvgPool2d(1)
        self.projection = get_projection(
            projection=projection,
            layer_mode=layer_mode,
            projection_config=projection_config,
        )

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Any]]:
        """
        Forward pass through the bottleneck.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Prediction (B, num_classes)
        """
        if len(x.shape) == 4:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc_1(x)
        x_projected = self.projection(x)
        return x, x_projected


class VariationalBottleneck(nn.Module):
    """
    Variational bottleneck that predicts latent distribution parameters.
    
    Supports:
      - Linear Mode: Flatten -> Linear -> Unflatten (Standard VAE)
      - Conv 1x1 Mode: Preserves spatial dimensions (Fully Convolutional VAE)
    """
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: int,
                 activation: nn.Module,
                 layer_mode: str = None,
                 projection: Optional[str] = None,
                 cov_mode: str = "diag",
                 normalization: Optional[str] = None,
                 force_circular: bool = False,
                 use_conv_1x1: bool = False,
                 standard_reparam: bool = True,
                 ):
        super().__init__()
        assert cov_mode in {"diag", "full"}
        
        self.C = in_channels
        self.D = latent_dim
        self.cov_mode = cov_mode
        self.layer_mode = layer_mode
        self.is_complex = layer_mode in ["complex", "split"]
        self.force_circular = force_circular
        self.use_conv_1x1 = use_conv_1x1          
        self.input_size = input_size
        self.standard_reparam = standard_reparam        # Helper for projection
        self.projection = get_projection(projection, layer_mode)
        self.input_conv = DoubleConv(
                    in_ch=in_channels,
                    out_ch=latent_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_mode=layer_mode,
                    activation=activation,
                    normalization=normalization,
                    residual=True,
                )

        # --- HEADS CONSTRUCTION HELPER ---
        def make_head(out_channels_head):
            if self.use_conv_1x1:
                return DoubleConv(
                    in_ch=latent_dim,
                    out_ch=out_channels_head,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_mode=layer_mode,
                    activation=activation,
                    normalization=normalization,
                    residual=True,
                )
            else:
                return SingleLinear(
                    in_ch=latent_dim* input_size * input_size,
                    out_ch=latent_dim* input_size * input_size,
                    linear_mode=layer_mode,
                    activation=None,
                    normalization=None,
                )
        
        # --- MEAN HEAD (mu) ---
        self.fc_mu = make_head(latent_dim)
        
        # Init mu to 0
        layer_to_init = self.fc_mu
        nn.init.zeros_(layer_to_init.block[0].conv_block[0].conv.weight if self.use_conv_1x1 else layer_to_init.linear.linear_block[0].linear.weight)
        nn.init.zeros_(layer_to_init.block[0].conv_block[0].conv.bias if self.use_conv_1x1 else layer_to_init.linear.linear_block[0].linear.bias)

        # --- COVARIANCE HEADS ---
        self.fc_p1 = None # W (WL) or Var (Standard)
        self.fc_p2 = None # V (WL) or Delta (Standard)
        self.n_tril = None

        if self.is_complex:
            out_dim = latent_dim if cov_mode == "diag" else latent_dim * latent_dim
            
            self.fc_p1 = make_head(out_dim) # W or Var
            self.fc_p2 = make_head(out_dim) # V or Delta
            
            # Init Weights to 0
            layer_p1 = self.fc_p1
            layer_p2 = self.fc_p2
            w_p1 = layer_p1.block[0].conv_block[0].conv.weight if self.use_conv_1x1 else layer_p1.linear.linear_block[0].linear.weight
            w_p2 = layer_p2.block[0].conv_block[0].conv.weight if self.use_conv_1x1 else layer_p2.linear.linear_block[0].linear.weight
            
            nn.init.zeros_(w_p1)
            nn.init.zeros_(w_p2)

            # Init Bias
            with torch.no_grad():
                bias_p1 = layer_p1.block[0].conv_block[0].conv.bias if self.use_conv_1x1 else layer_p1.linear.linear_block[0].linear.bias
                if bias_p1 is not None:
                    if not self.standard_reparam: 
                        # WL Mode: Init W as Identity
                        if cov_mode == "diag":
                            bias_p1.data.fill_(1.0)
                        else:
                            identity_flat = torch.eye(latent_dim).view(-1)
                            bias_p1.data.copy_(identity_flat)
                    else:
                        # Standard Mode: Init Var > 0 (softplus init)
                        bias_p1.data.fill_(0.5413) # Softplus(0.54) ~ 1.0

        else:
            # === REAL CASE ===
            if cov_mode == "diag":
                self.fc_p1 = make_head(latent_dim) # Var. We multiply by 2 to have more capacity for the variance head in real case, as we don't have the luxury of complex parameters to model covariance structure. This is a design choice to give the model more flexibility in modeling variances across different dimensions.
            else:
                self.n_tril = latent_dim * (latent_dim + 1) // 2
                self.fc_tril = make_head(self.n_tril)
                self.register_buffer("tril_idx", torch.tril_indices(row=latent_dim, col=latent_dim, offset=0), persistent=False)

        # --- DECODER ---
        if self.use_conv_1x1:
            self.fc_2 = DoubleConv(
                in_ch=latent_dim,
                out_ch=latent_dim,
                kernel_size=3, stride=1, padding=1,
                conv_mode=layer_mode,
                activation=activation,
                normalization=normalization,
                residual=True,
            )
            self.unflatten = nn.Identity()
        else:
            self.fc_2 = SingleLinear(
                in_ch=latent_dim* input_size * input_size,
                out_ch=latent_dim* input_size * input_size,
                linear_mode=layer_mode,
                activation=None,
                normalization=None,
                bias=False,
            )
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(latent_dim, input_size, input_size))
        self.out_conv = DoubleConv(
                in_ch=latent_dim,
                out_ch=in_channels,
                kernel_size=3, stride=1, padding=1,
                conv_mode=layer_mode,
                activation=activation,
                normalization=normalization,
                residual=True,
            )

    def _build_real_tril(self, raw_tril: torch.Tensor) -> torch.Tensor:
        """Constructs Real Lower Triangular matrix L. Handles (B, D, H, W) for Conv."""
        if raw_tril.ndim == 4:
            B, C, H, W = raw_tril.shape
            # Complex logic for spatial full covariance, falling back to diag recomended for Conv
            raise NotImplementedError("Full Covariance not yet implemented for Conv 1x1 mode")
        
        B = raw_tril.size(0)
        L = raw_tril.new_zeros(B, self.D, self.D)
        i, j = self.tril_idx[0], self.tril_idx[1]
        L[:, i, j] = raw_tril
        idx = torch.arange(self.D, device=L.device)
        diag_vals = L[:, idx, idx]
        L[:, idx, idx] = F.softplus(diag_vals) + 1e-6
        return L

    def to_latent(self, x: torch.Tensor, cap: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.input_conv(x)
        if not self.use_conv_1x1:                
            x = torch.flatten(x, 1)        
        
        mu = self.fc_mu(x)

        if self.is_complex:
            out_p1 = self.fc_p1(x)
            out_p2 = self.fc_p2(x)

            if self.standard_reparam:
                # === MODE STANDARD ===
                # p1 -> Variance (Real Positive)
                # p2 -> Delta (Complex)
                var = F.softplus(torch.abs(out_p1)) + 1e-6
                var = var.clamp(1e-6, 1e4)
                
                delta = var * cap * out_p2   # |delta| < cap * var to ensure positive-definiteness and stability. The cap is a hyperparameter that can be tuned.
                
                return mu, var, delta

            else:
                # === MODE WIDELY LINEAR ===
                # p1 -> W, p2 -> V
                W_raw = out_p1
                if self.force_circular:
                    V_raw = torch.zeros_like(W_raw)
                else:
                    V_raw = out_p2

                if self.cov_mode == "diag":
                    return mu, W_raw, V_raw
                else:
                    W = W_raw.view(-1, self.D, self.D)
                    V = V_raw.view(-1, self.D, self.D)
                    return mu, W, V
        else:
            # Real Params
            if self.cov_mode == "diag":
                var = F.softplus(self.fc_p1(x)) + 1e-6
                return mu, var, None
            else:
                L = self._build_real_tril(self.fc_tril(x))
                return mu, L, None

    def reparameterize(self, mu, p1, p2):
        """
        Unified Reparameterization.
        Args:
           p1: W (WL) OR Var (Standard)
           p2: V (WL) OR Delta (Standard)
        """
        if self.is_complex:
            if self.standard_reparam:
                # === STANDARD REPARAM ===
                # p1 = var, p2 = delta
                var, delta = p1, p2
                
                # Bruits
                epsx = torch.randn_like(var.real)
                epsy = torch.randn_like(var.real) # var is real/complex tensor with 0j

                re = delta.real
                
                a = (var.real + re).clamp_min(1e-8) # var should be real
                denom = torch.sqrt(2.0 * a)
                
                s1 = var + delta
                s2_sq = (var**2 - (delta.abs()**2)).clamp_min(1e-12)
                s2 = torch.sqrt(s2_sq)

                # Sampling z
                term1 = epsx * (s1 / denom)
                term2 = 1j * epsy * (s2 / denom)
                
                return mu + term1 + term2

            else:
                # === WIDELY LINEAR REPARAM ===
                # p1 = W, p2 = V
                eps_real = torch.randn_like(mu.real)
                eps_imag = torch.randn_like(mu.imag)
                
                if self.layer_mode == "split":
                    eps = torch.complex(eps_real, eps_imag) / math.sqrt(2.0)
                else:
                    eps = (eps_real + 1j * eps_imag) / math.sqrt(2.0)

                if self.cov_mode == "diag":
                    return mu + (p1 * eps) + (p2 * eps.conj())
                else:
                    return mu + torch.einsum('bij,bj->bi', p1, eps) \
                              + torch.einsum('bij,bj->bi', p2, eps.conj())
        else:
            # Real Case
            eps = torch.randn_like(mu)
            if self.cov_mode == "diag":
                std = torch.sqrt(p1)
                return mu + std * eps
            else:
                return mu + torch.einsum('bij,bj->bi', p1, eps)
                
    def to_input(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_2(z)
        if not self.use_conv_1x1:
            x = self.unflatten(x)
        x = self.out_conv(x)
        return x  
      
    def forward(self, x: torch.Tensor):
        mu, p1, p2 = self.to_latent(x)
        z = self.reparameterize(mu, p1, p2)
        x_hat = self.to_input(z)
        return x_hat, z, mu, p1, p2

    def sample(self, num_samples: int, sample_gmm: bool = False) -> torch.Tensor:
        """
        Sample from the latent prior.
        Args:
            num_samples: Batch size
            sample_gmm: Whether to sample from GMM prior if available.
        """
        if sample_gmm and hasattr(self, 'gmm_fitted') and self.gmm_fitted:
            return self.sample_gmm(num_samples)
        else:
            return self.sample_standard_normal(num_samples)

    def sample_standard_normal(self, num_samples: int) -> torch.Tensor:
        """
        Samples from standard normal prior.
        """
        B = num_samples
        D = self.D
        target_size = (self.input_size, self.input_size)
        
        # Handle shape (Conv vs Linear)
        if self.use_conv_1x1:
            shape = (B, D, *target_size)
        else:
            # In linear mode, the latent vector contains all flattened dimensions
            flat_dim = D * target_size[0] * target_size[1]
            shape = (B, flat_dim)
            
        device = next(self.parameters()).device
        
        # 1. White noise generation (Independent)
        if self.is_complex:
            eps_r = torch.randn(*shape, device=device) / math.sqrt(2.0)
            eps_i = torch.randn(*shape, device=device) / math.sqrt(2.0)
            if self.layer_mode == "split":
                z = torch.complex(eps_r, eps_i)
            else:
                z = eps_r + 1j * eps_i
        else:
            z = torch.randn(*shape, device=device)
                         
        return z

    def sample_gmm(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the dynamically sized Bayesian GMM.
        """
        if not hasattr(self, 'gmm_fitted') or not self.gmm_fitted:
            raise RuntimeError("GMM prior is not fitted! Call fit_gmm_prior(latents) first.")

        device = next(self.parameters()).device
        target_size = (self.input_size, self.input_size)
        
        total_samples = num_samples
        if self.use_conv_1x1:
            total_samples = num_samples * target_size[0] * target_size[1]

        # 1. Sample components indices based on Dirichlet weights
        probs = self.gmm_weights / self.gmm_weights.sum()
        cat_dist = D.Categorical(probs=probs)
        comp_indices = cat_dist.sample((total_samples,))

        means = self.gmm_means[comp_indices]
        covs = self.gmm_covs[comp_indices]
        
        # 2. Multivariate Normal sampling
        epsilon = 1e-5
        if covs.ndim == 2: # Diagonal covariance
            covs_matrix = torch.diag_embed(covs + epsilon)
            gmm_dist = D.MultivariateNormal(loc=means, covariance_matrix=covs_matrix)
        else: # Full covariance (if ever used)
            I = torch.eye(covs.shape[-1], device=covs.device)
            gmm_dist = D.MultivariateNormal(loc=means, covariance_matrix=covs + epsilon * I)
        
        z_raw = gmm_dist.sample().to(device) # [total_samples, 2*D or flat_dim]
        
        if hasattr(self, 'has_pca') and self.has_pca:
            # On ramène le vecteur de l'espace PCA vers l'espace VAE d'origine
            # Formule : z_original = z_pca @ matrice_pca + moyenne_pca
            z_raw = torch.matmul(z_raw, self.pca_components) + self.pca_mean

        # 3. Complex Conversion
        if self.is_complex:
            D_dim = z_raw.shape[1] // 2
            z_r = z_raw[:, :D_dim]
            z_i = z_raw[:, D_dim:]
            
            if self.layer_mode == "split":
                z_final = torch.complex(z_r, z_i)
            else:
                z_final = z_r + 1j * z_i
        else:
            z_final = z_raw

        # 4. Spatial Reshape (Only for Conv 1x1 mode, Linear mode keeps it flat)
        if self.use_conv_1x1:
            # Reconstruct the image shape from flattened pixels
            # Currently z_final is (B*H*W, D)
            z_final = z_final.view(num_samples, target_size[0], target_size[1], self.D)
            z_final = z_final.permute(0, 3, 1, 2) # -> (B, D, H, W)
            
        return z_final
        
    def fit_gmm_prior(self, latents: torch.Tensor, k_max: int,
                      weight_threshold: float, max_samples_for_gmm: int = 50000):
        """
        Fits a Bayesian GMM on the aggregate posterior.
        Automatically prunes empty clusters using a Dirichlet prior.
        """     
        logger.info(f"Fitting Bayesian GMM with upper bound k_max={k_max}...")
                
        if latents.shape[0] > max_samples_for_gmm:
            logger.info(f"Dataset too large ({latents.shape[0]} points). Subsampling to {max_samples_for_gmm}...")
            indices = np.random.choice(latents.shape[0], size=max_samples_for_gmm, replace=False)
            z_flat_fit = latents[indices]
        else:
            z_flat_fit = latents

        bgmm = BayesianGaussianMixture(
            n_components=k_max, 
            covariance_type='full', 
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1.0/k_max,
            random_state=42,
            max_iter=1000,
            n_init=3,
            init_params='k-means++',
            tol=0.001,
            verbose=1
        )

        pca = PCA(n_components=0.9, random_state=42) # Keep 90% variance, can be tuned as a hyperparameter
        with threadpool_limits(limits=1, user_api='blas'):
            z_reduced = pca.fit_transform(z_flat_fit)

        logger.info(f"Initial dimension : {z_flat_fit.shape[1]}")
        logger.info(f"Final dimension after PCA : {z_reduced.shape[1]}")

        # On sauvegarde les paramètres PCA dans le modèle pour la rétro-projection
        self.register_buffer('pca_components', torch.from_numpy(pca.components_).float())
        self.register_buffer('pca_mean', torch.from_numpy(pca.mean_).float())
        self.has_pca = True

        # Puis on donne cet espace purifié au BGM !
        with threadpool_limits(limits=1, user_api='blas'):
            bgmm.fit(z_reduced)       
        
         # 2. Prune "dead" components
        active_components = bgmm.weights_ > weight_threshold
        final_k = np.sum(active_components)
        
        logger.info(f"Bayesian GMM converged. Kept {final_k}/{k_max} active components.")
        
        if final_k == 0:
            logger.warning("Warning: All components pruned! Reverting to 1 component.")
            active_components[np.argmax(bgmm.weights_)] = True
            final_k = 1

        # 3. Extract and re-normalize active components
        active_weights = bgmm.weights_[active_components]
        active_weights = active_weights / np.sum(active_weights) # Ensure sum is exactly 1.0
        
        active_means = bgmm.means_[active_components]
        active_covs = bgmm.covariances_[active_components]

        # 4. Register buffers
        self.register_buffer('gmm_weights', torch.from_numpy(active_weights).float())
        self.register_buffer('gmm_means', torch.from_numpy(active_means).float())
        self.register_buffer('gmm_covs', torch.from_numpy(active_covs).float())
        self.register_buffer('gmm_fitted', torch.tensor(True))
                            
class LatentBottleneck(nn.Module):
    """
    Latent bottleneck layer for autoencoder architectures.

    Compresses spatial feature maps to a lower-dimensional latent space
    representation and reconstructs back to original dimensions. Supports
    both real and complex-valued data.

    Args:
        in_channels: Number of input channels
        latent_dim: Dimensionality of latent space (or channels if use_conv_1x1 is True)
        input_size: Spatial size of input (assumes square)
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        normalization: Type of normalization ('batch', 'instance', etc.)
        use_conv_1x1: If True, uses 1x1 convolutions instead of Linear layers (Preserves spatial dims, drastically fewer params).
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        input_size: int,
        activation: nn.Module,
        layer_mode: str = "complex",
        normalization: Optional[str] = None,
    ) -> None:
        """Initialize latent bottleneck layer."""
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.normalization = normalization
        self.activation = activation
        self.layer_mode = layer_mode
        
        self.encoder = SingleLinear(
            in_ch=in_channels * input_size * input_size,
            out_ch=latent_dim,
            linear_mode=layer_mode,
            activation=None,
            normalization=None,
        )

        self.unflatten = nn.Unflatten(
            dim=1, unflattened_size=(in_channels, input_size, input_size)
        )
        
        self.decoder = SingleLinear(
            in_ch=latent_dim,
            out_ch=in_channels * input_size * input_size,
            linear_mode=layer_mode,
            activation=None,
            normalization=None,
        )

    def to_latent(self, x: Tensor) -> Tensor:
        """
        Encode input to latent space.        """
        x = torch.flatten(x, 1)            
        x = self.encoder(x)
        return x

    def to_input(self, x: Tensor) -> Tensor:
        """
        Decode latent space back to input shape.
        """
        x = self.decoder(x)
        x = self.unflatten(x)      
        return x

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Any]]:
        x = self.to_latent(x)
        x = self.to_input(x)
        return x

def concat(x1, x2):
    """
    Concatenate two tensors with automatic padding for size matching.

    Pads x1 to match x2's spatial dimensions, then concatenates along
    the channel dimension. Used in U-Net style architectures.

    Args:
        x1: First tensor (CHW format)
        x2: Second tensor (CHW format) or None

    Returns:
        Concatenated tensor along channel dimension, or x1 if x2 is None
    """
    if x2 is None:
        return x1
    else:
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x
