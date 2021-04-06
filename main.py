from pandas import DataFrame
import numpy as np
from PIL import Image
from glob import glob
from os.path import sep,join,splitext
from skimage.feature import greycomatrix, greycoprops
from itertools import product

feature_names = ('dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity', 'ASM')

distances = (1,3,5)

angles = (0, np.pi/4, np.pi/2, 3*np.pi/4)

def get_full_names():
    dist_str = ('1','3','5')
    angles_str = '0deg,45deg,90deg,135deg'.split(',')
    return ['_'.join(f) for f in product(feature_names,dist_str,angles_str)]


def get_glcm_feature_array(patch):
    patch_64 = (patch/np.max(patch)*63).astype('uint8')
    glcm = greycomatrix(patch_64,distances,angles,64,True,True)
    feature_vector = []
    for feature in feature_names:
        feature_vector.extend(list(greycoprops(glcm,feature).flatten()))
    return feature_vector

texture_folder = "Texture"
samples_folder = "TextureSamples"
paths =glob(texture_folder + '\\*\\*.jpg')

fil2 = [p.split(sep) for p in paths]
_, categories, files = zip(*fil2)
size = 128,128

features = []
for category, infile in zip(categories,files):
    img=Image.open(join(texture_folder,category,infile))
    xr=np.random.randint(0, img.width-size[0],10)
    yr=np.random.randint(0, img.height-size[1],10)
    base_name, _ = splitext(infile)
    for i, (x, y) in enumerate(zip(xr,yr)):
        img_sample = img.crop((x,y,x+size[0],y+size[1]))
        img_sample.save(join(samples_folder,category,f'{base_name:s}_{i:02d}.jpg'))
        img_grey = img.convert('L')
        feature_vector = get_glcm_feature_array(np.array(img_grey))
        feature_vector.append(feature_vector)

full_feature_names = get_full_names()
full_feature_names.append('Category')

df = DataFrame(data=features, columns=full_feature_names)
df.to_csv('textures_data.csv', sep=',', index=False)
