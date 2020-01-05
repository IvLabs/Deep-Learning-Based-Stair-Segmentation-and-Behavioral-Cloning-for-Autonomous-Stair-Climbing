from fastai.vision import *
import warnings
import random, string

warnings.filterwarnings("ignore")

new_path = Path("../data/bh_data")

tfms = get_transforms(do_flip=False)
data = (ImageList.from_folder(new_path)
       .split_by_rand_pct(seed=38)
       .label_from_folder()
       .transform(tfms)
       .databunch(bs=5)
       .normalize(imagenet_stats))

learn = cnn_learner(data, models.resnet34, metrics=accuracy, wd=1e-2)
learn.fit_one_cycle(10,1e-2)
learn.save('bh_learner')