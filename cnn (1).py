import numpy as np
import matplotlib.pyplot as plt

# ====================== Common Kernels ======================
kernels = {
    "Identity": np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]),
    
    "Edge_Laplacian": np.array([[ 0, -1,  0],
                                [-1,  4, -1],
                                [ 0, -1,  0]]),
    
    "Sobel_Horizontal": np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]),
    
    "Sobel_Vertical": np.array([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]]),
    
    "Sharpen": np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]]),
    
    "Gaussian_Blur": np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]]) / 16.0,
    
    "Emboss": np.array([[-2, -1, 0],
                        [-1,  1, 1],
                        [ 0,  1, 2]])
}

# ====================== IMPLEMENTATIONS ======================

def convolve2d(image, kernel, stride=1, padding=0):
    """
    Perform 2D convolution (forward pass).

    Parameters:
        image  : 2D numpy array (H, W) — grayscale
        kernel : 2D numpy array (kH, kW)  — single kernel
                 OR 3D numpy array (kH, kW, out_channels) — multi-channel output
        stride : int
        padding: int  — zero-padding added to each side

    Returns:
        output : 2D array if kernel is 2D, 3D array (H_out, W_out, out_channels) if 3D kernel
    """
    # --- handle 3D kernel (multi-output-channel) ---
    if kernel.ndim == 3:
        kH, kW, out_channels = kernel.shape
        outputs = []
        for c in range(out_channels):
            outputs.append(convolve2d(image, kernel[:, :, c], stride=stride, padding=padding))
        return np.stack(outputs, axis=-1)

    # --- single 2D kernel from here ---
    H, W = image.shape
    kH, kW = kernel.shape

    # Zero-pad the image
    if padding > 0:
        image_padded = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    else:
        image_padded = image

    # Output spatial dimensions
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    output = np.zeros((H_out, W_out), dtype=np.float64)

    # Convolution loop (cross-correlation — standard in deep learning)
    for i in range(H_out):
        for j in range(W_out):
            row_start = i * stride
            col_start = j * stride
            patch = image_padded[row_start:row_start + kH, col_start:col_start + kW]
            output[i, j] = np.sum(patch * kernel)

    return output


def max_pool2d(image, pool_size=2, stride=2):
    """
    2D Max Pooling.

    Parameters:
        image     : 2D numpy array (H, W)
        pool_size : int — size of the pooling window (square)
        stride    : int

    Returns:
        output : 2D numpy array
    """
    H, W = image.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = np.zeros((H_out, W_out), dtype=image.dtype)

    for i in range(H_out):
        for j in range(W_out):
            row_start = i * stride
            col_start = j * stride
            patch = image[row_start:row_start + pool_size, col_start:col_start + pool_size]
            output[i, j] = np.max(patch)

    return output


def avg_pool2d(image, pool_size=2, stride=2):
    """
    2D Average Pooling.

    Parameters:
        image     : 2D numpy array (H, W)
        pool_size : int — size of the pooling window (square)
        stride    : int

    Returns:
        output : 2D numpy array
    """
    H, W = image.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = np.zeros((H_out, W_out), dtype=image.dtype)

    for i in range(H_out):
        for j in range(W_out):
            row_start = i * stride
            col_start = j * stride
            patch = image[row_start:row_start + pool_size, col_start:col_start + pool_size]
            output[i, j] = np.mean(patch)

    return output
