## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I calibrated the camera using the chessboard photos in camera_cal/. In camera_calibration.py, I created a 9 × 6 grid of inner corners and used cv2.findChessboardCorners() to find them in each image. Then I refined the corner positions with cv2.cornerSubPix(). Using 15 suitable images, I ran cv2.calibrateCamera() to estimate the camera matrix and five lens-distortion coefficients. Three photos didn’t show the full grid, and two had a different resolution, so I left them out. The RMS reprojection error was 0.8529 pixels. Finally, I used cv2.undistort() on each accepted calibration photo and saved 15 before-and-after comparisons to show the correction.

![Camera calibration: original and corrected](./outputs/calibration/comparisons/calibration2_comparison.jpg)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I corrected the lens distortion in the road images using the camera matrix and distortion coefficients from calibration. In undistort_images.py, load_calibration() loads those values, and undistort_image() uses cv2.undistort() to fix each image. Since the road images are 960 × 540 and the calibration images are 1280 × 720, I scaled the focal lengths and principal point by 0.75 and left the distortion coefficients unchanged. This assumes both sets of images were taken with the same camera and have the same field of view. I corrected all 17 road images and saved before-and-after comparisons at their original resolutions.

![Road image: original and distortion corrected](./outputs/undistortion/comparisons/test1_comparison.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I implemented and tested several thresholding methods in `thresholding.py` to identify lane markings in the road images. The experiment included color-based and gradient-based thresholding, as well as two combinations of their results using logical OR and AND operations.

Before applying the thresholds, I corrected the lens distortion and applied a 3 × 3 Gaussian blur to reduce image noise. For color thresholding, I used the HLS color space to detect bright, low-saturation pixels corresponding to white lane marking, and HSV to detect yellows markings. For gradient thresholding, I used the absolute horizontal Sobel derivative of the grayscale image, normalized to the range 0–255, to detect changes in intensity along lane-marking edges.

I tested all four methods on the 17 road images using the same threshold values and without applying a region of interest (ROI). I saved the individual binary masks and comparison images to evaluate how well each method detected lane markings and how much unwanted detail it introduced.

Based on the visual results, I chose the OR combination for the following pipeline stages. Unlike AND, which keeps only pixels detected by both methods, OR preserves pixels detected by either color or gradient thresholding. This helps retain lane markings that might otherwise be missed by one of the methods.

The following code excerpts show the main parts of the implementation in `thresholding.py`.

**Threshold settings**

All threshold values are defined in the `ThresholdSettings` class:

```python
@dataclass(frozen=True)
class ThresholdSettings:
    blur_kernel: int = 3
    white_lightness_min: int = 200
    white_saturation_max: int = 100
    yellow_hue_min: int = 15
    yellow_hue_max: int = 40
    yellow_saturation_min: int = 80
    yellow_value_min: int = 80
    sobel_kernel: int = 3
    gradient_min: int = 30
    gradient_max: int = 255
```

**Color thresholding**

The `color_threshold()` function converts the image to HLS and HSV and creates a binary mask containing pixels that satisfy the white or yellow color thresholds.

```python
def color_threshold(image: np.ndarray, settings: ThresholdSettings) -> np.ndarray:
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(
        hls, (0, settings.white_lightness_min, 0),
        (179, 255, settings.white_saturation_max),
    )
    yellow = cv2.inRange(
        hsv,
        (settings.yellow_hue_min, settings.yellow_saturation_min, settings.yellow_value_min),
        (settings.yellow_hue_max, 255, 255),
    )
    return ((white != 0) | (yellow != 0)).astype(np.uint8)
```

**Gradient thresholding**

The `gradient_threshold()` function applies the Sobel operator in the horizontal direction and selects pixels whose normalized gradient magnitude falls within the specified range.

```python
def gradient_threshold(image: np.ndarray, settings: ThresholdSettings) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=settings.sobel_kernel)
    absolute = np.abs(sobel_x)
    maximum = float(absolute.max())
    if maximum == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    scaled = np.rint(255 * absolute / maximum).astype(np.uint8)
    return ((scaled >= settings.gradient_min) & (scaled <= settings.gradient_max)).astype(np.uint8)
```

**Combining the thresholds**

In `create_masks()`, I apply Gaussian blur before running both thresholding methods. Their results are then combined using a pixelwise OR operation:

```python
filtered = cv2.GaussianBlur(image, (settings.blur_kernel, settings.blur_kernel), 0)
color = color_threshold(filtered, settings)
gradient = gradient_threshold(filtered, settings)
union = color | gradient
```

The resulting `union` mask is a binary image containing values of 0 and 1, where 1 represents a pixel selected by at least one thresholding method. The output is saved as a black-and-white PNG in `outputs/thresholding/masks/or/`.

The following images show the selected binary mask and a comparison of all four thresholding methods.

![Selected OR binary mask](outputs/thresholding/masks/or/straight_lines1.png)

![Color, gradient, OR, and AND thresholding comparison](outputs/thresholding/comparisons/straight_lines1_comparison.png)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I implemented the perspective transformation in `perspective_transform.py` to obtain a bird's-eye view of the road. This makes the lane markings appear approximately parallel, making them easier to detect and fit with polynomials in the following stages.

I selected four source points near the lane boundaries using the undistorted straight-road images. These points were mapped to a rectangle with 25% margins on both sides of the output image. The reference coordinates were `(580, 460)`, `(705, 460)`, `(1020, 660)`, and `(295, 660)` for an image resolution of 1280 × 720. For images with different resolutions, I scaled the coordinates accordingly.

The `create_perspective_transform()` function uses `cv2.getPerspectiveTransform()` to calculate the forward and inverse transformation matrices. I then used `warp_binary()` to transform the selected OR binary mask with `cv2.warpPerspective()`. Nearest-neighbor interpolation was used to preserve the binary pixel values.

To visualize the transformation, I also implemented `get_source_roi()` and `draw_region_of_interest()`. These functions determine the visible region of the original image that contributes to the bird's-eye view and outline it in green. This allows a direct comparison between the selected road region and the transformed binary image.

The following code excerpts show the main parts of the implementation.

**Perspective settings**

The reference image size, source points, and destination margins are defined in `PerspectiveSettings`:

```python
@dataclass(frozen=True)
class PerspectiveSettings:
    reference_size: tuple[int, int] = (1280, 720)
    source_points: tuple[tuple[float, float], ...] = (
        (580, 460), (705, 460), (1020, 660), (295, 660),
    )
    destination_margin: float = 0.25
```

**Calculating the transformation matrices**

In `create_perspective_transform()`, I scale the source points to match the image resolution, define the destination rectangle, and calculate both transformation matrices:

```python
source = source * np.float32([
    width / settings.reference_size[0],
    height / settings.reference_size[1],
])
margin = settings.destination_margin
destination = np.float32([
    [margin, 0], [1 - margin, 0], [1 - margin, 1], [margin, 1],
]) * np.float32([width - 1, height - 1])
matrix = cv2.getPerspectiveTransform(source, destination)
inverse_matrix = cv2.getPerspectiveTransform(destination, source)
```

**Applying the perspective transformation**

Finally, `warp_binary()` applies the transformation to the binary image:

```python
def warp_binary(binary: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    return cv2.warpPerspective(
        binary, matrix, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
```

The following comparison shows the selected region in the undistorted image and the resulting bird's-eye binary image.

![Source ROI and bird's-eye binary mask](./outputs/perspective/comparisons/straight_lines1_comparison.png)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

