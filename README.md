# SAR-Methods

### Description

This repository includes some methods for analyzing and post-processing on Synthetic Aperature Radar (SAR) imagery. SAR is a sensor that sends and receives microwave signals. Some of this data is publically available, but pre-processing software is needed to create digital images that a user can interpret. So far, the methods in this repository have been tested with Sentinel-1 data processed with ISCE and ESA's Sentinel-1 Toolbox.

The methods in this repository include:
- ARIA method from JPL<sup>1</sup>
- Thresholding based off of a bimodal distribution (generally used for water detection)<sup>2</sup>
- Unsupervised clustering using ISODATA<sup>3</sup>

Please visit the `notebooks` folder for some examples of processing!

### Software Dependencies

- Python 3.8
- Run `pip install -r requirements.txt`

### References

<sup>1</sup>Yun, S. H., Fielding, E., Simons, M., Agram, P., Rosen, P., Owen, S., & Webb, F. (2012). Rapid and reliable damage proxy map from InSAR coherence.

<sup>2</sup>Cao, H., Zhang, H., Wang, C., & Zhang, B. (2019). Operational flood detection using Sentinel-1 SAR data over large areas. Water, 11(4), 786.

<sup>3</sup>Uddin, K., Matin, M. A., & Meyer, F. J. (2019). Operational flood mapping using multi-temporal sentinel-1 SAR images: a case study from Bangladesh. Remote Sensing, 11(13), 1581.
