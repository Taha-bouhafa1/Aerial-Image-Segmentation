import torch
import torch.nn as nn
import torch.nn.functional as F



class conv_block(nn.Module):
    def __init__(self, in_c, out_c,act=True):
        super().__init__()
        layers=[nn.Conv2d(in_c,out_c,kernel_size=3,padding=1)]
        if act==True:
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))

        self.conv=nn.Sequential(*layers)

    def forward(self,x):
        return self.conv(x)


class encoder_block(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.c1=nn.Sequential(
            conv_block(in_c,out_c,act=act),
            conv_block(out_c,out_c,act=act),
        )
        self.p1=nn.MaxPool2d(2)
    def forward(self,x):
        x=self.c1(x)
        p=self.p1(x)
        return x,p


class UNet3Plus(nn.Module):
    def __init__(self,num_classes=6):
        super().__init__()


        """ ENCODER """
        self.e1=encoder_block(3,64)
        self.e2=encoder_block(64,128)
        self.e3=encoder_block(128,256)
        self.e4=encoder_block(256,512)

        """ Bottleneck """
        self.e5=nn.Sequential(
            conv_block(512,1024),
            conv_block(1024,1024),
        )

        """ DECODER 4 """
        self.e1_d4=conv_block(64,64)
        self.e2_d4=conv_block(128,64)
        self.e3_d4=conv_block(256,64)
        self.e4_d4=conv_block(512,64)
        self.e5_d4=conv_block(1024,64)

        self.d4=conv_block(64*5,64)

        """ DECODER 3 """
        self


    def forward(self,inputs):
        """ Encoder """
        e1,p1=self.e1(inputs)
        e2,p2=self.e2(p1)
        e3,p3=self.e3(p2)
        e4,p4=self.e4(p3)

        """ Bottleneck """
        e5=self.e5(p4)

        """ DECODER 4 """

        e1_d4=F.max_pool2d(e1,kernel_size=8,stride=8)
        e1_d4=self.e1_d4(e1_d4)

        e2_d4=F.max_pool2d(e2,kernel_size=4,stride=4)
        e2_d4=self.e2_d4(e2_d4)

        e3_d4=F.max_pool2d(e3,kernel_size=2,stride=2)
        e3_d4=self.e3_d4(e3_d4)

        e4_d4=self.e4_d4(e4)

        e5_d4=F.interpolate(e5,scale_factor=2,mode="bilinear",align_corners=True)
        e5_d4=self.e5_d4(e5_d4)

        d4=torch.cat([e1_d4,e2_d4,e3_d4,e4_d4,e5_d4],dim=1)
        d4=self.d4(d4)





