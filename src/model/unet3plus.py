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
    def __init__(self,in_c,out_c,act=True):
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
    def __init__(self,num_classes=6,deep_sup=True):
        super().__init__()
        self.deep_sup=deep_sup


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
        """ Classification """
        self.cls = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(1024, 2, kernel_size=1, padding=0),
            nn.AdaptiveMaxPool2d((1))
        )
        """ DECODER 4 """
        self.e1_d4=conv_block(64,64)
        self.e2_d4=conv_block(128,64)
        self.e3_d4=conv_block(256,64)
        self.e4_d4=conv_block(512,64)
        self.e5_d4=conv_block(1024,64)

        self.d4=conv_block(64*5,64)

        """ DECODER 3 """
        self.e1_d3 = conv_block(64, 64)
        self.e2_d3 = conv_block(128, 64)
        self.e3_d3 = conv_block(256, 64)
        self.e4_d3 = conv_block(64, 64)
        self.e5_d3 = conv_block(1024, 64)

        self.d3=conv_block(64*5,64)

        """ DECODER 2 """
        self.e1_d2 = conv_block(64, 64)
        self.e2_d2 = conv_block(128, 64)
        self.e3_d2 = conv_block(64, 64)
        self.e4_d2 = conv_block(64, 64)
        self.e5_d2 = conv_block(1024, 64)
        self.d2 = conv_block(64 * 5, 64)

        """ DECODER 1 """

        self.e1_d1 = conv_block(64, 64)
        self.e2_d1 = conv_block(64, 64)
        self.e3_d1 = conv_block(64, 64)
        self.e4_d1 = conv_block(64, 64)
        self.e5_d1 = conv_block(1024, 64)
        self.d1 = conv_block(64 * 5, 64)

        """ Deep Supervision """
        if deep_sup == True:
            self.y1 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            self.y2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            self.y3 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            self.y4 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            self.y5 = nn.Conv2d(1024, num_classes, kernel_size=3, padding=1)
        else:
            self.y1 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self,inputs):
        """ Encoder """
        e1,p1=self.e1(inputs)
        e2,p2=self.e2(p1)
        e3,p3=self.e3(p2)
        e4,p4=self.e4(p3)

        """ Bottleneck """
        e5=self.e5(p4)

        """ Classification """
        cls_logits = self.cls(e5)
        cls_probs = F.softmax(cls_logits, dim=1)
        cls_mask = cls_probs[:, 1].view(-1, 1, 1, 1)

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

        """ DECODER 3 """

        e1_d3 = F.max_pool2d(e1, 4, 4)
        e1_d3 = self.e1_d3(e1_d3)

        e2_d3 = F.max_pool2d(e2, 2, 2)
        e2_d3 = self.e2_d3(e2_d3)

        e3_d3 = self.e3_d3(e3)

        e4_d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        e4_d3 = self.e4_d3(e4_d3)

        e5_d3 = F.interpolate(e5, scale_factor=4, mode="bilinear", align_corners=True)
        e5_d3 = self.e5_d3(e5_d3)

        d3 = torch.cat([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3], dim=1)
        d3 = self.d3(d3)


        """ DECODER 2 """

        # Decoder 2
        e1_d2 = F.max_pool2d(e1, 2, 2)
        e1_d2 = self.e1_d2(e1_d2)

        e2_d2 = self.e2_d2(e2)

        e3_d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        e3_d2 = self.e3_d2(e3_d2)

        e4_d2 = F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=True)
        e4_d2 = self.e4_d2(e4_d2)

        e5_d2 = F.interpolate(e5, scale_factor=8, mode="bilinear", align_corners=True)
        e5_d2 = self.e5_d2(e5_d2)

        d2 = torch.cat([e1_d2, e2_d2, e3_d2, e4_d2, e5_d2], dim=1)
        d2 = self.d2(d2)

        """ DECODER 1 """
        # Decoder 1
        e1_d1 = self.e1_d1(e1)

        e2_d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        e2_d1 = self.e2_d1(e2_d1)

        e3_d1 = F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=True)
        e3_d1 = self.e3_d1(e3_d1)

        e4_d1 = F.interpolate(d4, scale_factor=8, mode="bilinear", align_corners=True)
        e4_d1 = self.e4_d1(e4_d1)

        e5_d1 = F.interpolate(e5, scale_factor=16, mode="bilinear", align_corners=True)
        e5_d1 = self.e5_d1(e5_d1)

        d1 = torch.cat([e1_d1, e2_d1, e3_d1, e4_d1, e5_d1], dim=1)
        d1 = self.d1(d1)

        if self.deep_sup == True:
            y1 = self.y1(d1) * cls_mask
            y2 = F.interpolate(self.y2(d2), scale_factor=2, mode="bilinear", align_corners=True) * cls_mask
            y3 = F.interpolate(self.y3(d3), scale_factor=4, mode="bilinear", align_corners=True) * cls_mask
            y4 = F.interpolate(self.y4(d4), scale_factor=8, mode="bilinear", align_corners=True) * cls_mask
            y5 = F.interpolate(self.y5(e5), scale_factor=16, mode="bilinear", align_corners=True) * cls_mask

            return y1, y2, y3, y4, y5


if __name__ == "__main__":
    inputs=torch.rand((1,3,512,512))
    model=UNet3Plus(num_classes=6,deep_sup=True)

    y=model(inputs)

    print(len(y))




