"""
download_real_dataset.py
-------------------------
Downloads and organizes the REAL public dataset used by classifier.py:
  - Traffic images (dense_traffic / sparse_traffic) + fire images
    from OlafenwaMoses/Traffic-Net (https://github.com/OlafenwaMoses/Traffic-Net)
  - Extra real fire images from cair/Fire-Detection-Image-Dataset
    (https://github.com/cair/Fire-Detection-Image-Dataset)

Produces:
  real_dataset/traffic/heavy/*.jpg   (from Traffic-Net dense_traffic)
  real_dataset/traffic/light/*.jpg   (from Traffic-Net sparse_traffic)
  real_dataset/fire/fire/*.jpg       (from Traffic-Net fire + cair fire images)
  real_dataset/fire/no_fire/*.jpg    (from Traffic-Net accident + extra sparse_traffic)

Run this once, then run classifier.py (it auto-detects real_dataset/ and
uses it instead of the synthetic generate_dataset.py fallback).

Requires internet access to github.com / release-assets.githubusercontent.com.
"""
import os
import zipfile
import shutil
import subprocess
import urllib.request

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "real_dataset")

TRAFFICNET_URL = "https://github.com/OlafenwaMoses/Traffic-Net/releases/download/1.0/trafficnet_dataset_v1.zip"
CAIR_FIRE_URL = "https://codeload.github.com/cair/Fire-Detection-Image-Dataset/zip/refs/heads/master"


def download(url, dest):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")


def extract_n(zf, names, prefix, outdir, n, skip=0):
    os.makedirs(outdir, exist_ok=True)
    files = [x for x in names if x.startswith(prefix) and x.lower().endswith((".jpg", ".jpeg", ".png"))][skip:skip + n]
    for f in files:
        data = zf.read(f)
        with open(os.path.join(outdir, os.path.basename(f)), "wb") as out:
            out.write(data)
    return len(files)


def build():
    os.makedirs(OUT, exist_ok=True)
    tn_zip = os.path.join(BASE, "_trafficnet.zip")
    download(TRAFFICNET_URL, tn_zip)

    with zipfile.ZipFile(tn_zip) as z:
        names = z.namelist()
        n1 = extract_n(z, names, "trafficnet_dataset_v1/train/dense_traffic/",
                        os.path.join(OUT, "traffic", "heavy"), 60)
        n2 = extract_n(z, names, "trafficnet_dataset_v1/train/sparse_traffic/",
                        os.path.join(OUT, "traffic", "light"), 60)
        n3 = extract_n(z, names, "trafficnet_dataset_v1/train/fire/",
                        os.path.join(OUT, "fire", "fire"), 60)
        n4 = extract_n(z, names, "trafficnet_dataset_v1/train/accident/",
                        os.path.join(OUT, "fire", "no_fire"), 40)
        n5 = extract_n(z, names, "trafficnet_dataset_v1/train/sparse_traffic/",
                        os.path.join(OUT, "fire", "no_fire"), 20, skip=60)
    os.remove(tn_zip)
    print(f"Traffic-Net: {n1} heavy, {n2} light, {n3} fire, {n4}+{n5} no_fire")

    # extra real fire images for more diversity (requires unrar-free: apt-get install -y unrar-free)
    cair_zip = os.path.join(BASE, "_cairfire.zip")
    try:
        download(CAIR_FIRE_URL, cair_zip)
        with zipfile.ZipFile(cair_zip) as z:
            z.extract("Fire-Detection-Image-Dataset-master/Fire images.rar", os.path.join(BASE, "_cairtmp"))
        rar_path = os.path.join(BASE, "_cairtmp", "Fire-Detection-Image-Dataset-master", "Fire images.rar")
        subprocess.run(["unrar-free", rar_path], cwd=os.path.join(BASE, "_cairtmp"), check=False)
        extracted_dir = os.path.join(BASE, "_cairtmp", "Fire images")
        if os.path.isdir(extracted_dir):
            dst = os.path.join(OUT, "fire", "fire")
            files = [f for f in os.listdir(extracted_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))][:40]
            for f in files:
                shutil.copy(os.path.join(extracted_dir, f), os.path.join(dst, "cair_" + f))
            print(f"cair dataset: added {len(files)} extra fire images")
    except Exception as e:
        print(f"(optional) cair fire dataset skipped: {e}")
    finally:
        for p in (cair_zip, os.path.join(BASE, "_cairtmp")):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                os.remove(p)

    print(f"\nReal dataset ready at: {OUT}")


if __name__ == "__main__":
    build()
