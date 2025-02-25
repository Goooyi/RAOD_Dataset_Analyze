# %%
import os

import cv2
import numpy as np
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
)
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

# Constants
BIT8 = 256
BIT16 = 65536
BIT24 = 16777216


def read_raw_24b(file_path, img_shape=(1, 1, 1856, 2880), read_type=np.uint8):
    """
    Read 24-bit raw image data
    """
    raw_data = np.fromfile(file_path, dtype=read_type)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32)

    return raw_data[0, 0]


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    """
    # Convert to LAB color space (luminance + color channels)
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)

    # Normalize L channel to [0, 255] for CLAHE
    l_channel = lab[:, :, 0]
    l_channel_norm = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel_norm)

    # Update L channel and convert back to RGB
    lab[:, :, 0] = cv2.normalize(cl, None, 0, 100, cv2.NORM_MINMAX)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced


def process_raw_image(
    file_path,
    is_night=False,
    night_gamma=2.2,
    brightness_target=0.5,
    pre_amplification=1.0,
    contrast_stretch=0.5,
    saturation_boost=1.2,
    use_local_tone_mapping=True,
    clahe_strength=2.5,
):
    """
    Process raw image with night-specific enhancements if needed

    Parameters:
    file_path: Path to the raw image file
    is_night: Boolean flag for night-specific processing
    night_gamma: Gamma correction value for night scenes (default: 2.2)
    brightness_target: Target mean brightness after processing (0-1)
    pre_amplification: Manual pre-amplification factor (1.0 = no change)
    contrast_stretch: Strength of contrast stretching (0-1, 0=none, 1=full range)
    saturation_boost: Saturation multiplier (1.0 = no change)
    use_local_tone_mapping: Whether to use local tone mapping
    clahe_strength: Strength of CLAHE contrast enhancement (1.0-5.0)
    """
    # Read and debayer the raw data
    raw_data = read_raw_24b(file_path)
    debayered_data = demosaicing_CFA_Bayer_bilinear(raw_data)
    debayered_data = cv2.resize(
        debayered_data, (640, 640), interpolation=cv2.INTER_LINEAR
    )

    # Initial brightness assessment
    initial_brightness = np.mean(debayered_data)
    max_value = np.max(debayered_data)

    # Apply manual pre-amplification (user controlled)
    if is_night and pre_amplification > 1.0:
        debayered_data = debayered_data * pre_amplification

    # Apply white balance
    if is_night:
        # For night scenes, use upper percentile
        percentile = 90
        r_ref = np.percentile(debayered_data[:, :, 0], percentile)
        g_ref = np.percentile(debayered_data[:, :, 1], percentile)
        b_ref = np.percentile(debayered_data[:, :, 2], percentile)

        # Ensure we don't divide by zero
        if r_ref > 0 and b_ref > 0:
            debayered_data[:, :, 0] *= g_ref / r_ref
            debayered_data[:, :, 2] *= g_ref / b_ref
    else:
        # Original white balance for day scenes
        mean_r = debayered_data[:, :, 0].mean()
        mean_g = debayered_data[:, :, 1].mean()
        mean_b = debayered_data[:, :, 2].mean()

        if mean_r > 0 and mean_b > 0:
            debayered_data[:, :, 0] *= mean_g / mean_r
            debayered_data[:, :, 2] *= mean_g / mean_b

    # Normalize to [0, 1]
    im = np.clip(debayered_data, 0, BIT24 - 1) / (BIT24 - 1) * (BIT8 - 1)

    # Night-specific processing
    if is_night:
        # Apply noise reduction
        noise_reduction_strength = 5
        if noise_reduction_strength % 2 == 0:  # Ensure kernel size is odd
            noise_reduction_strength += 1

        # Convert to 8-bit for bilateral filter
        im_8bit = (im * 255).astype(np.uint8)

        # Apply bilateral filter with modest strength
        im_8bit = cv2.bilateralFilter(im_8bit, noise_reduction_strength, 25, 25)

        # Convert back to float [0,1] range
        im = im_8bit.astype(np.float32) / 255.0

        # Apply gamma correction with the specified value
        im_gamma = cv2.pow(im, 1 / night_gamma)

        if use_local_tone_mapping:
            # Apply CLAHE for local contrast enhancement with user-specified strength
            im_8bit = (im_gamma * 255).astype(np.uint8)

            # Apply CLAHE to each channel independently
            clahe = cv2.createCLAHE(clipLimit=clahe_strength, tileGridSize=(8, 8))
            channels = cv2.split(im_8bit)
            enhanced_channels = []

            for ch in channels:
                enhanced_channels.append(clahe.apply(ch))

            im_enhanced = cv2.merge(enhanced_channels)
            im_gamma = im_enhanced.astype(np.float32) / 255.0

        # Apply contrast stretching if requested
        if contrast_stretch > 0:
            # Calculate percentile values for contrast stretching (more conservative with lower stretch values)
            lower_percentile = contrast_stretch * 2  # ranges from 0 to 2
            upper_percentile = 100 - (contrast_stretch * 2)  # ranges from 100 to 96

            p_low, p_high = np.percentile(
                im_gamma, [lower_percentile, upper_percentile]
            )

            # Avoid division by zero
            if p_high > p_low:
                # Apply contrast stretching with intensity based on contrast_stretch parameter
                stretched = (im_gamma - p_low) / (p_high - p_low)
                im_gamma = np.clip(stretched, 0, 1)

        # Apply manual brightness adjustment (user controlled via brightness_target)
        current_brightness = np.mean(
            0.299 * im_gamma[:, :, 0]
            + 0.587 * im_gamma[:, :, 1]
            + 0.114 * im_gamma[:, :, 2]
        )

        # Only adjust if current brightness is significantly different from target
        if abs(current_brightness - brightness_target) > 0.05:
            brightness_factor = brightness_target / max(current_brightness, 0.01)
            # Limit the adjustment factor to reasonable values
            brightness_factor = max(0.5, min(2.0, brightness_factor))
            im = np.clip(im_gamma * brightness_factor, 0, 1)
        else:
            im = im_gamma

        # Apply saturation adjustment if requested
        if saturation_boost != 1.0:
            hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 1)
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        # Regular day processing
        im = cv2.pow(im, 1 / 3.2)

    # Final normalization to ensure proper display range
    im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)

    return im


# Function to display a more comprehensive range of processing parameters
def display_comprehensive_night_comparison(file_path):
    """
    Display a wide range of processing parameters to find the best settings
    """
    # Create a figure to show multiple versions
    plt.figure(figsize=(15, 10))

    # Basic parameters
    gamma = 2.0
    brightness = 0.4
    pre_amp = 1.0

    # Original image (minimally processed)
    plt.subplot(3, 3, 1)
    img1 = process_raw_image(
        file_path,
        is_night=True,
        night_gamma=gamma,
        brightness_target=brightness,
        pre_amplification=pre_amp,
        contrast_stretch=0.0,
        saturation_boost=1.0,
        use_local_tone_mapping=False,
    )
    plt.imshow(img1 / 255.0)
    plt.title(f"Minimal Processing\nGamma: {gamma}, Brightness: {brightness}")
    plt.axis("off")

    # Low enhancement
    plt.subplot(3, 3, 2)
    img2 = process_raw_image(
        file_path,
        is_night=True,
        night_gamma=1.5,
        brightness_target=0.3,
        pre_amplification=1.2,
        contrast_stretch=0.2,
        saturation_boost=1.0,
        clahe_strength=1.5,
    )
    plt.imshow(img2 / 255.0)
    plt.title("Low Enhancement\nGamma: 1.5, Pre-Amp: 1.2")
    plt.axis("off")

    # Medium-low enhancement
    plt.subplot(3, 3, 3)
    img3 = process_raw_image(
        file_path,
        is_night=True,
        night_gamma=1.8,
        brightness_target=0.4,
        pre_amplification=1.5,
        contrast_stretch=0.3,
        saturation_boost=1.1,
        clahe_strength=2.0,
    )
    plt.imshow(img3 / 255.0)
    plt.title("Medium-Low Enhancement\nGamma: 1.8, Pre-Amp: 1.5")
    plt.axis("off")

    # Medium enhancement
    plt.subplot(3, 3, 4)
    img4 = process_raw_image(
        file_path,
        is_night=True,
        night_gamma=2.2,
        brightness_target=0.5,
        pre_amplification=2.0,
        contrast_stretch=0.4,
        saturation_boost=1.2,
        clahe_strength=2.5,
    )
    plt.imshow(img4 / 255.0)
    plt.title("Medium Enhancement\nGamma: 2.2, Pre-Amp: 2.0")
    plt.axis("off")

    # Medium-high enhancement
    plt.subplot(3, 3, 5)
    img5 = process_raw_image(
        file_path,
        is_night=True,
        night_gamma=2.6,
        brightness_target=0.6,
        pre_amplification=2.5,
        contrast_stretch=0.5,
        saturation_boost=1.3,
        clahe_strength=3.0,
    )
    plt.imshow(img5 / 255.0)
    plt.title("Medium-High Enhancement\nGamma: 2.6, Pre-Amp: 2.5")
    plt.axis("off")

    # High enhancement
    plt.subplot(3, 3, 6)
    img6 = process_raw_image(
        file_path,
        is_night=True,
        night_gamma=3.0,
        brightness_target=0.7,
        pre_amplification=3.0,
        contrast_stretch=0.6,
        saturation_boost=1.4,
        clahe_strength=3.5,
    )
    plt.imshow(img6 / 255.0)
    plt.title("High Enhancement\nGamma: 3.0, Pre-Amp: 3.0")
    plt.axis("off")

    # Very high enhancement
    plt.subplot(3, 3, 7)
    img7 = process_raw_image(
        file_path,
        is_night=True,
        night_gamma=3.5,
        brightness_target=0.8,
        pre_amplification=3.5,
        contrast_stretch=0.7,
        saturation_boost=1.5,
        clahe_strength=4.0,
    )
    plt.imshow(img7 / 255.0)
    plt.title("Very High Enhancement\nGamma: 3.5, Pre-Amp: 3.5")
    plt.axis("off")

    # Extreme enhancement
    plt.subplot(3, 3, 8)
    img8 = process_raw_image(
        file_path,
        is_night=True,
        night_gamma=4.0,
        brightness_target=0.9,
        pre_amplification=4.0,
        contrast_stretch=0.8,
        saturation_boost=1.6,
        clahe_strength=4.5,
    )
    plt.imshow(img8 / 255.0)
    plt.title("Extreme Enhancement\nGamma: 4.0, Pre-Amp: 4.0")
    plt.axis("off")

    # Custom preset (adjust as needed)
    plt.subplot(3, 3, 9)
    img9 = process_raw_image(
        file_path,
        is_night=True,
        night_gamma=2.0,
        brightness_target=0.45,
        pre_amplification=1.75,
        contrast_stretch=0.35,
        saturation_boost=1.15,
        clahe_strength=2.25,
    )
    plt.imshow(img9 / 255.0)
    plt.title("Balanced Preset\nGamma: 2.0, Pre-Amp: 1.75")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "minimal": img1,
        "low": img2,
        "medium_low": img3,
        "medium": img4,
        "medium_high": img5,
        "high": img6,
        "very_high": img7,
        "extreme": img8,
        "balanced": img9,
    }


# Custom parameter exploration
def explore_single_parameter(file_path, parameter="gamma", values=None):
    """
    Explore the effect of changing a single parameter while keeping others constant

    Parameters:
    file_path: Path to the image file
    parameter: The parameter to vary ('gamma', 'brightness', 'pre_amp', 'contrast', 'saturation')
    values: List of values to try for the parameter
    """
    # Default parameter values
    params = {
        "night_gamma": 2.2,
        "brightness_target": 0.5,
        "pre_amplification": 1.5,
        "contrast_stretch": 0.4,
        "saturation_boost": 1.2,
        "clahe_strength": 2.5,
    }

    # Set default values if not provided
    if values is None:
        if parameter == "gamma":
            values = [1.2, 1.6, 2.0, 2.4, 2.8]
        elif parameter == "brightness":
            values = [0.3, 0.4, 0.5, 0.6, 0.7]
        elif parameter == "pre_amp":
            values = [1.0, 1.5, 2.0, 2.5, 3.0]
        elif parameter == "contrast":
            values = [0.0, 0.25, 0.5, 0.75, 1.0]
        elif parameter == "saturation":
            values = [0.8, 1.0, 1.2, 1.4, 1.6]
        elif parameter == "clahe":
            values = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Create the figure
    cols = min(5, len(values))
    rows = (len(values) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = (
        np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])
    )

    # Process and display each image
    for i, value in enumerate(values):
        # Update the parameter
        if parameter == "gamma":
            params["night_gamma"] = value
            param_name = "Gamma"
        elif parameter == "brightness":
            params["brightness_target"] = value
            param_name = "Brightness"
        elif parameter == "pre_amp":
            params["pre_amplification"] = value
            param_name = "Pre-Amp"
        elif parameter == "contrast":
            params["contrast_stretch"] = value
            param_name = "Contrast"
        elif parameter == "saturation":
            params["saturation_boost"] = value
            param_name = "Saturation"
        elif parameter == "clahe":
            params["clahe_strength"] = value
            param_name = "CLAHE"

        # Process the image
        img = process_raw_image(file_path, is_night=True, **params)

        # Display the image
        axes[i].imshow(img / 255.0)
        axes[i].set_title(f"{param_name}: {value}")
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(values), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# %%

file_path = "../Dataset/01Valid/night-06010.raw"
result_images = display_comprehensive_night_comparison(file_path)

# %%
# Pre-amplification exploration
# explore_single_parameter(file_path, 'gamma')
# explore_single_parameter(file_path, 'pre_amp')
# explore_single_parameter(file_path, 'contrast')
# explore_single_parameter(file_path, 'brightness')
# explore_single_parameter(file_path, 'clahe')
# explore_single_parameter(file_path, 'saturation')

img = process_raw_image(
    file_path,
    is_night=True,
    night_gamma=1.8,
    pre_amplification=1.6,
    contrast_stretch=0.0,
    saturation_boost=1.1,
)
plt.figure(figsize=(10, 8))
plt.imshow(img / 255.0)
# plt.title('Enhanced Night Image (Gamma: 3.0, Brightness: 0.8)')
plt.axis("off")
plt.show()

# %%
# processing the whole dataset
read_dir = "../Dataset/01Valid"
save_dir = "../Dataset/01Valid-night-debayer-640"
for filename in os.listdir(read_dir):
    if filename.startswith("night") and filename.endswith(".raw"):
        file_path = os.path.join(read_dir, filename)
        img = process_raw_image(
            file_path,
            is_night=True,
            night_gamma=1.8,
            pre_amplification=1.6,
            contrast_stretch=0.0,
            saturation_boost=1.1,
        )
        cv2.imwrite(os.path.join(save_dir, filename.replace(".raw", ".png")), img)

# %%

# class RaodDebayer:
#     def __init__(self):
#         # Define the five 3x3 kernels
#         self.kernels = np.array(
#             [
#                 # Kernel 0: Direct copy
#                 [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
#                 # Kernel 1: + pattern
#                 [[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]],
#                 # Kernel 2: x pattern
#                 [[0.25, 0, 0.25], [0, 0, 0], [0.25, 0, 0.25]],
#                 # Kernel 3: horizontal
#                 [[0, 0, 0], [0.5, 0, 0.5], [0, 0, 0]],
#                 # Kernel 4: vertical
#                 [[0, 0.5, 0], [0, 0, 0], [0, 0.5, 0]],
#             ]
#         )

#         # Define kernel indices for each channel and pattern
#         self.kernel_indices = {
#             "r": {"RG": [0, 3], "GB": [4, 2]},
#             "g": {"RG": [1, 0], "GB": [0, 1]},
#             "b": {"RG": [2, 4], "GB": [3, 0]},
#         }

#     def apply_kernel(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
#         """Apply a single kernel using convolution"""
#         return convolve(img, kernel, mode="reflect")

#     def debayer(self, raw: np.ndarray) -> np.ndarray:
#         """
#         Convert Bayer pattern raw image to RGB

#         Args:
#             raw: Input raw image (H, W)
#         Returns:
#             RGB image (3, H, W)
#         """
#         H, W = raw.shape
#         output = np.zeros((3, H, W), dtype=raw.dtype)

#         # Create masks for different Bayer positions
#         rg_mask = np.zeros((H, W), dtype=bool)
#         gb_mask = np.zeros((H, W), dtype=bool)

#         # Set masks based on Bayer pattern (assuming RGGB)
#         rg_mask[0::2, 0::2] = True  # R positions
#         rg_mask[1::2, 1::2] = True  # Second G positions
#         gb_mask[1::2, 0::2] = True  # First G positions
#         gb_mask[0::2, 1::2] = True  # B positions

#         # Process each channel
#         for ch_idx, channel in enumerate(["r", "g", "b"]):
#             # Process RG pixels
#             rg_result = np.zeros_like(raw)
#             for kernel_idx in self.kernel_indices[channel]["RG"]:
#                 rg_result += self.apply_kernel(raw * rg_mask, self.kernels[kernel_idx])

#             # Process GB pixels
#             gb_result = np.zeros_like(raw)
#             for kernel_idx in self.kernel_indices[channel]["GB"]:
#                 gb_result += self.apply_kernel(raw * gb_mask, self.kernels[kernel_idx])

#             output[ch_idx] = rg_result + gb_result

#         return output


# class Debayer:
#     def __init__(self):
#         # Define the two standard bilinear kernels
#         self.H_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4

#         self.H_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4

#         # To maintain API compatibility, we'll map the kernels to the original indices structure
#         # Even though we only use two kernels, we'll map them to the existing kernel_indices format
#         self.kernel_indices = {
#             "r": {"RG": [0], "GB": [0]},  # Use H_RB
#             "g": {"RG": [1], "GB": [1]},  # Use H_G
#             "b": {"RG": [0], "GB": [0]},  # Use H_RB
#         }

#     def apply_kernel(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
#         """Apply a single kernel using convolution"""
#         return convolve(img, kernel, mode="reflect")

#     def debayer(self, raw: np.ndarray) -> np.ndarray:
#         """
#         Convert Bayer pattern raw image to RGB
#         Args:
#             raw: Input raw image (H, W)
#         Returns:
#             RGB image (3, H, W)
#         """
#         H, W = raw.shape
#         output = np.zeros((3, H, W), dtype=raw.dtype)

#         # Create masks for different Bayer positions (assuming RGGB)
#         R_mask = np.zeros((H, W), dtype=bool)
#         G_mask = np.zeros((H, W), dtype=bool)
#         B_mask = np.zeros((H, W), dtype=bool)

#         # Set masks based on Bayer pattern
#         R_mask[0::2, 0::2] = True  # R positions
#         G_mask[0::2, 1::2] = True  # G positions in R rows
#         G_mask[1::2, 0::2] = True  # G positions in B rows
#         B_mask[1::2, 1::2] = True  # B positions

#         # Process each channel using appropriate kernel
#         # Red channel
#         output[0] = self.apply_kernel(raw * R_mask, self.H_RB)

#         # Green channel
#         output[1] = self.apply_kernel(raw * G_mask, self.H_G)

#         # Blue channel
#         output[2] = self.apply_kernel(raw * B_mask, self.H_RB)

#         return output
