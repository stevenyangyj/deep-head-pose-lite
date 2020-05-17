# Hopenet-lite
A lite-version hopenet for head pose estimation with PyTorch  

## Note  
Hopenet-lite uses unofficial-implement ShuffleNetV2 as backbone network, and now the lastest PyTorch contains official ShuffleNetV2 with various width. If you are seeking for stable performance, please use official implementation and re-train hopenet-lite!  
'''  
import torchvision.models as models  
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)  
...  
https://pytorch.org/docs/stable/torchvision/models.html#classification

## Doc.  
The project is based on **natanielruiz's excellent work named Hopenet**.

The link: https://github.com/natanielruiz/deep-head-pose

You can run the network on CPU (i7-8700 six cores) with **35 FPS** or GPU (RTX 2070) with **130 FPS**

**If you used natanielruiz's code in your project, then do not need to change anything except the nueral network you used. At the same time, please refer to the [training code](https://github.com/natanielruiz/deep-head-pose/blob/master/code/train_hopenet.py) of natanielruiz's project for training your own model (if you need)**

'''  
import hopenetlite_v2  
net = hopenetlite_v2.HopeNetLite()  
saved_state_dict = torch.load('hopenet_lite_6MB.pkl', map_location="cpu")  
net.load_state_dict(saved_state_dict, strict=False)  
net.eval()  
'''

The Pre-trained model in "model" folder, but the model is not very robust to image quality, we will release more 
robust model in the future.

Thanks for natanielruiz's excellent work again.  

## Update  
Hi, guys, I finally have time to update this project...  
I uploaded the lastest hopenet-lite model with official ShuffleNetV2 from Pytorch torchvision, you can use it like this:  
'''  
import stable_hopenetlite  
pos_net = stable_hopenetlite.shufflenet_v2_x1_0()  
saved_state_dict = torch.load('model/shuff_epoch_120.pkl', map_location="cpu")  
pos_net.load_state_dict(saved_state_dict, strict=False)  
pos_net.eval()  
'''  
The Pre-trained model named "shuff_epoch_120.pkl" in "model" folder. If you think my training is not perfect, you could re-train the model. Just enjoy yourself !  

Here are some examples:  
![](https://github.com/OverEuro/deep-head-pose-lite/blob/master/figs/th1.png)  
![](https://github.com/OverEuro/deep-head-pose-lite/blob/master/figs/th2.png)  
![](https://github.com/OverEuro/deep-head-pose-lite/blob/master/figs/th3.png)  
![](https://github.com/OverEuro/deep-head-pose-lite/blob/master/figs/th4.png)  
![](https://github.com/OverEuro/deep-head-pose-lite/blob/master/figs/th5.png)
