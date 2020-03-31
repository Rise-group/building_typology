|            | Table   | Figure                   | Algorithm   |
| ---------- | ------- | ------------------------ | ----------- |
| 1_dataset  | Table 1 | Figure 5, Figure 6 (b,c) |             |
| 2_training |         | Figure 7, Figure 8       | Algorithm 1 |

## Folder: 1_dataset

Files:

    - Raw_Data_Information: [CSV] containes 5 columns divided into photo, longitude, latitude, typology, stories and economic stratum.
    - spatial_and_stratum_distribution: [PNG] Figure A shows Medellín's building characteristics obtained from non-publicy available cadastral data provided by the municipality. Figure B and C show predominant number of stories and socio-economic stratum from the surveyed buildings, this images is obtained from the file Raw_Data_Information.
    - surveyed_buildings: [PNG] presents the geographical distribution for the buildings used during training, this image is obtained from the file Raw_Data_Information.

Dictionary of Variables:

| Variable Name    | Description                                                                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| photo            | id for each image                                                                                                                                      |
| longitude        | angular measurement from the prime meridian                                                                                                            |
| latitude         | angular measurement from the equator line                                                                                                              |
| typology         | typology is a clasification of the stuctures based on the main characteristics that defined it's strutural behaviour when is subjected to seismic load |
| stories          | number of stories on each building                                                                                                                     |
| economic_stratum | economic stratification of the building, ranging from lower-low(1), low(2), upper-low(3) medium(4), medium-high(5), high(6)                            |

## Folder: 2_training

Files:

    - methodology: [PNG] shows a general outline of the processes applied to this problem.
    - train_modalities: [PNG] shows a comparison of architectures trained using only images and multimodal architectures
    - algorithm: [PNG] presents the pseudocode for the selection of the best performing architecture considering all three data distributions and chosen metrics.
    - data: [FOLDER] includes .csv files for three data distributions used during training, each distribution has train/val/test .csv files

Dictionary of Variables:

| Variable Name | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| photo         | id for each image                                              |
| class         | label of each image                                            |
| p\_#N         | One-hot vector encoding for the number of stories in the image |
