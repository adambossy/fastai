#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fastai
from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')


# In[14]:


urls = search_images('brownstone brooklyn photos', max_images=1)
urls[0]


# In[15]:


from fastdownload import download_url
dest = 'brooklyn.jpg'
download_url(urls[0], dest)


# In[16]:


from fastai.vision.all import *
i = Image.open(dest)
i.to_thumb(256, 256)


# In[17]:


download_url(search_images('manhattan from street level photos', max_images=1)[0], 'manhattan.jpg')


# In[18]:


Image.open('manhattan.jpg').to_thumb(256)


# In[19]:


searches = 'brownstone brooklyn', 'manhattan from street level'
path = Path('brooklyn_or_manhattan')
from time import sleep

for s in searches:
    dest = (path/s)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{s} photo'))
    sleep(1)
    resize_images(path/s, max_size=400, dest=path/s)


# In[20]:


failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)


# In[21]:


dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')],
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)


# In[22]:


learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)


# In[25]:


from fastai.vision.all import PILImage

img = PILImage.create('manhattan.jpg')
img.to_thumb(256)

is_manhattan, _, probs = learn.predict(img)
print(f"This is a: {is_manhattan}.")
print(f"Probability it's Manhattan: {probs[0]:.4f}")


# In[ ]:




