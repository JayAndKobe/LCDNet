
import torch

from thop import profile, clever_format
from torch import nn as nn
from lib.SMT import smt_t
from lib.module import DepthNet, Reduction, CRB, LearnableCluster, GBAM, MEA_Decoder


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rgb_encoder = smt_t(pretrained=True)
        # self.depth_encoder = Depth_Backbone()
        self.depth_encoder = DepthNet()

        self.cluster = LearnableCluster()
        self.refine_r = CRB(3)
        self.refine_d = CRB(3)

        channel_list = [64, 128, 256, 512]


        self.cr1 = Reduction(channel_list[1], 64)
        self.cr2 = Reduction(channel_list[2], 64)
        self.cr3 = Reduction(channel_list[3], 64)


        self.fusion1 = GBAM(64,7)
        self.fusion2 = GBAM(64,5)
        self.fusion3 = GBAM(64,3)
        self.fusion4 = GBAM(64,1)



        self.Decoder = MEA_Decoder()



    def forward(self, r, d, od):
        bins = self.cluster(od)

        rgb = self.rgb_encoder(r)
        rgb = self.refine_r(rgb, bins)
        d = self.depth_encoder(d)
        d = self.refine_d(d, bins)

        rgb1 = rgb[0]
        rgb2 = self.cr1(rgb[1])
        rgb3 = self.cr2(rgb[2])
        rgb4 = self.cr3(rgb[3])

        d1 = d[0]
        d2 = self.cr1(d[1])
        d3 = self.cr2(d[2])
        d4 = self.cr3(d[3])

        f1 = self.fusion1(rgb1, d1)
        f2 = self.fusion2(rgb2, d2)
        f3 = self.fusion3(rgb3, d3)
        f4 = self.fusion4(rgb4, d4)

        pred = self.Decoder([f1, f2, f3, f4])

        return pred,bins


if __name__ == '__main__':
    model = Net()
    r = torch.randn([1, 3, 384, 384])
    d = torch.randn([1, 3, 384, 384])
    # print(model)
    params, flops = profile(model, inputs=(r, d))
    flops, params = clever_format([flops, params], "%.2f")
    print(flops, params)
