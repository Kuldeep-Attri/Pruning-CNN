import torchvision
import re
import torch

in_height = 224
in_width = 224

def get_num_gen(gen):
    return sum(1 for x in gen)

def flops_layer(layer):
    """
    Calculate the number of flops for given a string information of layer.
    We extract only resonable numbers and use them.
    
    Args:
        layer (str) : example
            Linear (512 -> 1000)
            Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    """
    #print(layer)
    idx_type_end = layer.find('(')
    type_name = layer[:idx_type_end]
    
    params = re.findall('[^a-z](\d+)', layer)    
    flops = 1
    
    if layer.find('Linear') >= 0:
        C1 = int(params[0])
        C2 = int(params[1])
        flops = C1*C2
        
    elif layer.find('Conv2d') >= 0:
        C1 = int(params[0])
        C2 = int(params[1])
        K1 = int(params[2])
        K2 = int(params[3])
        # print("Check : ", params[7])
        # image size
        # H = 224
        # W = 224
        global in_height
        global in_width
        flops = C1*C2*K1*K2*in_height*in_width
        in_height = (in_height-K1 + 2*int(params[6]))/int(params[4]) +1
        in_width = (in_width - K2 + 2*int(params[6]))/int(params[4]) +1

    elif layer.find('Pool') >=0:
        # print("Hello there is poooling")
        C1 = int(params[0])
        C2 = int(params[2])
        global in_height
        global in_width
        in_height = (in_height - C1)/C2 + 1
        in_width = (in_width - C1)/C2 + 1
        # print C1, "   ", C2



    return flops

def calculate_flops(gen):
    """
    Calculate the flops given a generator of pytorch model.
    It only compute the flops of forward pass.
    
    Example:
        >>> net = torchvision.models.resnet18()
        >>> calculate_flops(net.children())
    """
    flops = 0;
    
    for child in gen:
        num_children = get_num_gen(child.children())
        
        # leaf node
        if num_children == 0:
            flops += flops_layer(str(child))
        
        else:
            flops += calculate_flops(child.children())
    
    return flops

#net = torchvision.models.resnet18()
def main_(model=None, input_height=224, input_width=224):


    in_height=input_height
    in_width=input_width    
    print("in_height: ",in_height, "  in_width",in_width )
    net = model
    flops = calculate_flops(net.children())
    print(flops / 10.0**9, 'Billion')
# 11.435429919 G
