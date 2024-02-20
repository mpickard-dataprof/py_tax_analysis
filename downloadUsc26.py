from urllib.request import urlretrieve
import zipfile
import os.path
import logging
 
def downloadFile(url, filename="usc26.zip", unzip=False):
    filename = "./downloads/" + os.path.basename(url)
    if not os.path.exists(filename):
        print("Downloading " + filename)
        f = urlretrieve(url, filename)
    if (unzip and zipfile.is_zipfile(filename)):
        with zipfile.ZipFile(filename, "r") as archive:
            archive.extractall("./downloads/")
            return archive.filename


usc26_url = "https://uscode.house.gov/download/releasepoints/us/pl/118/39/xml_usc26@118-39.zip"
print("Successfully downloaded and extracted:  ",
      downloadFile(usc26_url, "usc26.zip", unzip=True))