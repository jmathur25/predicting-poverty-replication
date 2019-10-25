# predicting-poverty-replication
The purpose of this repository is to replicate the Jean et. al. (2016) paper using only Python3 and PyTorch. These more up-to-date tools and instructions should help anyone trying to build off this work.

# Setup
I recommend creating a virtual environment for this project. I prefer using Anaconda.

First run:
```
git clone https://github.com/jmather625/predicting-poverty-replication
conda create -n <ENV_NAME> python=3.7 pip gdal
conda activate <ENV_NAME>
pip install -r requirements.txt
```

If you want to run Jupyter Notebooks in an environment, run the following inside the environment:
```
pip install --user ipykernel
python -m ipykernel install --user --name=<ENV_NAME>
```

# Downloading Data
There are several steps here, make sure to follow all of them.

1. Download the nightlights data by running the Jupyter file in `process_data/scripts_python/download_nightlights_data.ipynb`
2. Download the 2016-2017 Malawi survey data from https://microdata.worldbank.org/index.php/catalog. The World Bank wants to know how people use their data, so you will have to sign in and explain why you want their data. Query 'LSMS' and filter the years to help find the search. The title of the data when I downloaded it was `Fourth Integrated Household Survey 2016-2017`. If you look at the data description tab, you should see a huge list of files starting with `HH_Metadata`. Navigate to `Get Microdata` to download the data. Make sure to download the Stata version.
3. Unzip the downloaded Malawi data into `process_data/data/input/LSMS`, and rename the folder to `malawi-2016`.
4. Run the scripts in `process_data/scripts` in the following order: <br>
    a. `download_nightlights_data.ipynb` <br>
        This downloads a file from the NOAA that determines the nightlight value at any given latitude and longitude. <br>
    b. `process_survey_data.ipynb` <br>
        This processes the survey data that you download in #2-3<br>
    c. `get_image_download_locations.ipynb` <br>
        This takes the files created by #4b and generated the locations to download <br>
5. Get a Google static maps API key at https://developers.google.com/maps/documentation/maps-static/intro. Save it to `process_data/data/api_key.txt`. <br>
    a. Downloading aerial imagery with a timestamp isn't free. The static maps API lets you query lat/long but not time. So, the images are likely from 2019 (current year) but we can't be sure. By using 2016 data, we are effectively trying to use 2019 images to predict their 2016 values. As a result, we can't fully match the original paper's results, but at least we can do this at no cost. <br>
    b. I did try using this same approach but with 2013 Malawi data, and the outcome was far worse. I think this indicates that the images are the problem, because using 2019 images to predict 2016 values should be easier than trying to predict 2013 values.
6. Navigate to `process_data/data/api_key.txt` and run `downlod_mw_2016.ipynb`. It downloads satellite images of size 400x400 to `process_data/ims-malawi-2016`. This script takes several hours, as over 30k images are being downloaded. I hope to make all the images available on Dropbox or some hosting service to cut down the runtime. <br>
    a. `evaluate_download_progress.ipynb` can be used to see how many images have been downloaded. Note that if you are working on a VM like Google's Deep Learning VM, connections can close after extended periods of time. This doesn't stop the script itself from running, but there's no way to see the printed output anymore if you reopen the Jupyter file. This new file I made will simply read the number of images that are download. If the number keeps increasing, you know the script is still running. <br>
    b. `im_download_demo.ipynb` demonstrates the API call in a standalone file.



