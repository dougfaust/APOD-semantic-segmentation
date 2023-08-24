# A no-mask image semantic segmentation model
This functions as a demonstration of end-to-end data science/machine learning skills from web scraping to deploy

1) scrapes NASA astronomy data to build a dataset
2) builds a basic classifier (a CNN based on the TinyVGG)
3) implements a novel segmentation algorithm which does not require training mask data
4) deploys model

## 1) Dataset creation

The dataset used here is created from the search function of the [Astronomy Picture of The Day](https://apod.nasa.gov/apod/astropix.html)

The APOD search page is used to select images from each class is located in
`notebooks/apod data scraper.ipynb`

Tools:

* `beautifulsoup`
* Jupyter notebooks

## 2) CNN classifier
## 3) Build image mask for inference
## 4) Deploy
