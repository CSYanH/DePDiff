# Multiscale-Aided Patch Diffusion Model (MAP-Diff) 

This is the code repository.


Currently the repository is ***still*** being prepared, more details to be included soon...

## Environment Preparing
```
python 3.9.12
torch 1.12.0
```
You should prepare at least 3080ti gpu or change the batch size. 

```pip install -r requirement.txt``` </br>


### Training process

Run the following command:

```python3 train_diffusion.py --confign {THE PATH OF TASK CONFIG}```

### Testing process

Run the following command:

```python3 eval_diffusion.py --confign {THE PATH OF TASK CONFIG} --resume {THE PATH OF PRETRAINED MODEL}```


### Dataset preparing

Training data [[Google Drive]](https://drive.google.com/drive/folders/1fwqz8G2Ej3jQFsYECh0?usp=sharing) (LOL Dataset, LOL-v1 Dataset, LOL-v2 Dataset)

Testing data [[Google Drive]](https://drive.google.com/open?id=1PrvL8dDxBY1oJR72iDf) (including LIME, MEF, NPE, VV, DICP)


Parts of this code repository is based on the following works:

* https://github.com/ermongroup/ddim
* https://github.com/bahjat-kawar/ddrm
* https://github.com/JingyunLiang/SwinIR
* https://github.com/IGITUGraz/WeatherDiffusion
