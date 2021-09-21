import torch
import time
from    torch import nn


class ConvPool(nn.Module):
    def __init__(self, in_ch, out_ch, k_c=9, k_p=2, p_c=4, p_p=1):
        super(ConvPool, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k_c, padding=p_c)
        self.activate = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=k_p, stride=2, padding=p_p)
    def forward(self, x):
        out = self.conv(x)
        out = self.activate(out)
        out = self.pool(out)
        return out


class CPM(nn.Module):
    def __init__(self, landmark):
        super(CPM, self).__init__()
        self.lm = landmark
        self.center_pool = nn.AvgPool2d(kernel_size=9, stride=8, padding=1, ceil_mode=True)

        conv1_stage1 = ConvPool(3, 128)
        conv2_stage1 = ConvPool(128, 128)
        conv3_stage1 = ConvPool(128, 128)
        conv4_ch, conv4_k, conv4_p = [128, 32, 512, 512, self.lm+1], [5, 9, 1, 1], [2, 4, 0, 0]
        conv4s_stage1 = [nn.Sequential(
            nn.Conv2d(conv4_ch[i], conv4_ch[i + 1], kernel_size=conv4_k[i], padding=conv4_p[i]),
            nn.ReLU() ) for i in range(4)]
        self.stage1 = [conv1_stage1, conv2_stage1, conv3_stage1, conv4s_stage1]


        conv1_stagex, conv2_stagex, conv3_stagex, conv4_stagex, Mconv_stagex = \
            [], [], [], [], []

        for _ in range(5):
            conv1_stage2 = ConvPool(3, 128)
            conv2_stage2 = ConvPool(128, 128)
            conv3_stage2 = ConvPool(128, 128)
            conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
            mcov_ch, mcov_k, mcov_p = [32 + self.lm + 2, 128, 128, 128, 128, self.lm + 1], \
                                      [11, 11, 11, 1, 1], [5, 5, 5, 0, 0]
            Mconv_stage2 = [nn.Sequential(
                nn.Conv2d(mcov_ch[i], mcov_ch[i + 1], kernel_size=mcov_k[i], padding=mcov_p[i]),
                nn.ReLU() ) for i in range(5)]
            conv1_stagex.append(conv1_stage2)
            conv2_stagex.append(conv2_stage2)
            conv3_stagex.append(conv3_stage2)
            conv4_stagex.append(conv4_stage2)
            Mconv_stagex.append(Mconv_stage2)
        self.stagex = [conv1_stagex, conv2_stagex, conv3_stagex, conv4_stagex, Mconv_stagex]

    def forward(self, img, center_map):
        pool_center_map = self.center_pool(center_map).unsqueeze(0)

        stage1_map = img
        for num, layer in enumerate(self.stage1):
            if num != 3:
                stage1_map = layer(stage1_map)
            else:
                for sub_layer in layer:
                    stage1_map = sub_layer(stage1_map)
        stagex_map = [stage1_map for _ in range(7)]
        for stage_num in range(5):
            out = img
            for layer_num, layer in enumerate(self.stagex):
                if layer_num == 4:
                    stagex_map[stage_num + 1] = torch.cat([out, stagex_map[stage_num + 1], pool_center_map], dim=1)
                    for num_sub, sub_layer in enumerate(layer[stage_num]):
                        stagex_map[stage_num + 1] = sub_layer(stagex_map[stage_num+1])
                    stagex_map[stage_num + 2] = stagex_map[stage_num + 1]
                else:
                    out = layer[stage_num](out)
        return stagex_map

def test():
    img = torch.rand((1, 3, 368, 368))
    centermap = torch.zeros((1, 368, 368))
    model = CPM(15)
    st = time.time()
    mapping = model(img, centermap)
    end = time.time()
    print(end - st)


if __name__ == '__main__':
    test()

