# Hopenet-lite
A lite-version hopenet for head pose estimation with PyTorch  

## Note  
Hopenet-lite uses unofficial-implement ShuffleNetV2 as backbone network, and now the lastest PyTorch contains official ShuffleNetV2 with various width. If you are seeking stable performance, please use official implementation and re-train hopenet-lite!  
'''  
import torchvision.models as models  
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)  
...  
https://pytorch.org/docs/stable/torchvision/models.html#classification

## Doc.  
The project is based on **natanielruiz's excellent work named Hopenet**.

The link: https://github.com/natanielruiz/deep-head-pose

You can run the network on CPU (i7-8700 six cores) with **35 FPS** or GPU (RTX 2070) with **130 FPS**

If you used natanielruiz's code in your project, then do not need to change anything except the nueral network you used.

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
