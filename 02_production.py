#!/usr/bin/env python
# coding: utf-8

# In[2]:


from fastbook import *


# In[3]:


get_ipython().run_line_magic('env', 'AZURE_SEARCH_KEY=27e299fbd44b4eddab9a44c539458005')
key = os.environ.get('AZURE_SEARCH_KEY')
results = search_images_bing(key, 'greenwich village')
images = results.attrgot('contentUrl')
len(images)
images[0]
dest = 'greenwich_village.jpg'
download_url(images[0], dest)
image = Image.open(dest)
image.to_thumb(256)


# In[4]:


nyc_hoods = 'greenwich village', 'west village', 'tribeca'
path = Path('neighbhorhoods')
if not path.exists():
    path.mkdir()
    for h in nyc_hoods:
        dest = (path/h)
        dest.mkdir(exist_ok=True)
        results = search_images_bing('27e299fbd44b4eddab9a44c539458005', f'{h} neighborhood')
        download_images(dest, urls=results.attrgot('contentUrl'))


# In[5]:


image_files = get_image_files(path)


# In[6]:


failed = verify_images(image_files)
failed.map(Path.unlink)
len(failed)


# In[7]:


hoods = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(),
    get_y=parent_label,
    item_tfms=Resize(128)
)


# In[8]:


data_loaders = hoods.dataloaders(path)
data_loaders.valid.show_batch(max_n=4, nrows=1)


# In[9]:


# hoods = hoods.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
# data_loaders = hoods.dataloaders(path)
# data_loaders.train.show_batch(max_n=8, nrows=2, unique=True)


# In[ ]:


hoods = hoods.new(item_tfms=Resize(128), batch_tfms=aug_transforms())
data_loaders = hoods.dataloaders(path)
# data_loaders.train.show_batch(max_n=8, nrows=2, unique=True)

learn = vision_learner(data_loaders, resnet18, metrics=error_rate)
learn.fine_tune(4)

