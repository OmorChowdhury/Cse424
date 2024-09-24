 # EuroSAT Dataset: Preprocessing Overview

## Data Before Preprocessing

The EuroSAT dataset contains Sentinel-2 satellite images covering 13 spectral bands and 10 classes. It provides a rich dataset for land use and land cover classification. You can explore the original dataset and access further details from the following resources:

- [Explore the EuroSAT Dataset on Papers With Code](https://paperswithcode.com/dataset/eurosat)
- [EuroSAT GitHub Repository](https://github.com/phelber/eurosat)

---

## Data After Preprocessing

[Pre-processing](https://github.com/Abrarkhan88/CSE424/blob/main/Task%204/Task%204c2/cse424_EUROSAT%2BEDA.ipynb).

### Exploratory Data Analysis (EDA) Steps

**Pre-Exploratory Analysis**:
- **Data Distribution in Classes**: Analyzed the number of images across different classes to ensure balance.
- **Visualizing Sample Images**: Random images from each class were visualized to confirm correct loading.
- **Pixel Intensity Analysis**: Checked pixel intensity statistics such as sum, mean, and median values.
- **Correlation Between Channels**: Visualized the correlation between RGB channels for initial analysis.

**Post-Exploratory Analysis**:
- **Data Distribution in Classes**: Ensured data distribution remained consistent after preprocessing.
- **Normalizing and Transforming Images**: Pixel values were normalized from 0-255 to 0-1, and random transformations like resize and flip were applied.
- **RGB Distribution**: Visualized the RGB distribution on the transformed images.
- **Image Ratio Distribution**: Ensured that image height and width remained consistent after transformations.
- **Correlation Between Channels**: Re-checked the correlation between RGB channels post-normalization.
- **Post-Process Pixel Intensity Distribution**: Analyzed pixel intensity distribution after normalization.
- **Data Augmentation**: Applied data augmentation techniques using PyTorch’s `permute()` for tensor manipulation.
