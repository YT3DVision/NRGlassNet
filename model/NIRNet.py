import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.backbone.swin_transformer.swin_transformer_v2 import SwinTransformerV2_demo


def add_conv_stage(
    dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False
):
    if useBN:
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            # nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(),
            nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ReLU(),
        )


def upsample(ch_coarse, ch_fine, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        # nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ConvTranspose2d(
            ch_coarse, ch_fine, kernel_size, stride, padding, bias=False
        ),
        nn.ReLU(),
    )


class Conv2DLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        k_size,
        stride,
        padding=None,
        dilation=1,
        norm=1,
        act=1,
        bias=False,
    ):
        super(Conv2DLayer, self).__init__()
        # use default padding value or (kernel size // 2) * dilation value
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (k_size - 1) // 2

        self.add_module(
            "conv2d",
            nn.Conv2d(
                in_channels,
                out_channels,
                k_size,
                stride,
                padding,
                dilation=dilation,
                bias=bias,
            ),
        )
        if norm is not None:
            self.add_module("norm", norm(out_channels))
        if act is not None:
            self.add_module("act", act)


class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)

        return x * y


class MCC(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
    def __init__(
        self,
        channel,
        norm=nn.BatchNorm2d,
        dilation=1,
        bias=False,
        res_scale=0.9,
        act=nn.ReLU(True),
    ):
        super(MCC, self).__init__()
        self.rgb_conv = Conv2DLayer(
            channel,
            channel,
            k_size=3,
            stride=1,
            dilation=dilation,
            norm=norm,
            act=act,
            bias=bias,
        )
        self.nir_conv = Conv2DLayer(
            channel,
            channel,
            k_size=3,
            stride=1,
            dilation=dilation,
            norm=norm,
            act=act,
            bias=bias,
        )
        self.fused_conv = Conv2DLayer(
            2 * channel,
            channel,
            k_size=3,
            stride=1,
            dilation=dilation,
            norm=norm,
            act=act,
            bias=bias,
        )
        self.rgb_att = SElayer(channel=channel)
        self.nir_att = SElayer(channel=channel)
        self.fuse_att = SElayer(channel=channel)
        self.res_scale = res_scale

    def forward(self, rgb_fea, nir_fea):
        rgb = self.rgb_conv(rgb_fea)
        nir = self.nir_conv(nir_fea)
        fused = self.fused_conv(torch.cat((rgb_fea, nir_fea), 1))

        rgb = self.rgb_att(rgb) * self.res_scale + rgb
        nir = self.nir_att(nir) * self.res_scale + nir
        fused = self.fuse_att(fused) * self.res_scale + fused

        glass_region = torch.sigmoid(rgb - nir)
        fused = fused * glass_region

        return fused


class NIRNet(nn.Module):
    def __init__(
        self,
        backbone_path=None,
    ):
        super(NIRNet, self).__init__()
        swin_transformer = SwinTransformerV2_demo(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=24,
            drop_path_rate=0.2,
        )

        swin_transformer2 = SwinTransformerV2_demo(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=24,
            drop_path_rate=0.2,
        )

        if backbone_path is not None:
            state_dict = torch.load(backbone_path)
            pretrained_dict = state_dict["model"]
            print("---start load pretrained modle of swin encoder---")
            swin_transformer.load_state_dict(pretrained_dict, strict=False)
            swin_transformer2.load_state_dict(pretrained_dict, strict=False)

        self.pebed = swin_transformer.patch_embed
        self.pos_drop = swin_transformer.pos_drop
        self.rgb_layer0 = swin_transformer.layers[0]
        self.rgb_layer1 = swin_transformer.layers[1]
        self.rgb_layer2 = swin_transformer.layers[2]
        self.rgb_layer3 = swin_transformer.layers[3]

        self.pebed = swin_transformer2.patch_embed
        self.pos_drop = swin_transformer2.pos_drop
        self.nir_layer0 = swin_transformer2.layers[0]
        self.nir_layer1 = swin_transformer2.layers[1]
        self.nir_layer2 = swin_transformer2.layers[2]
        self.nir_layer3 = swin_transformer2.layers[3]

        self.mcc1 = MCC(channel=1024)
        self.mcc2 = MCC(channel=512)
        self.mcc3 = MCC(channel=256)
        self.mcc4 = MCC(channel=128)

        self.up_32 = upsample(1024, 512)
        self.up_21 = upsample(512, 256)
        self.up_10 = upsample(256, 128)
        self.up_final = upsample(128, 64, kernel_size=8, stride=4, padding=2)

        self.conv3m = add_conv_stage(1024, 1024, useBN=True)
        self.conv2m = add_conv_stage(1024, 512, useBN=True)
        self.conv1m = add_conv_stage(512, 256, useBN=True)
        self.conv0m = add_conv_stage(256, 128, useBN=True)

        self.final_pred = nn.Conv2d(64, 1, 3, 1, 1)
        self.pred0 = nn.Conv2d(128, 1, 3, 1, 1)
        self.pred1 = nn.Conv2d(256, 1, 3, 1, 1)
        self.pred2 = nn.Conv2d(512, 1, 3, 1, 1)
        self.pred3 = nn.Conv2d(1024, 1, 3, 1, 1)
        self.fuse_pred = nn.Conv2d(3 + 5, 1, 3, 1, 1)

    def forward(self, x, nir):
        input = x
        b, c, h, w = x.shape
        x = self.pebed(x)
        x = self.pos_drop(x)
        rgb_layer0, rgb_layer0_d = self.rgb_layer0(x)  # 3
        rgb_layer1, rgb_layer1_d = self.rgb_layer1(rgb_layer0_d)  # 1.5
        rgb_layer2, rgb_layer2_d = self.rgb_layer2(rgb_layer1_d)  # 0.75
        rgb_layer3 = self.rgb_layer3(rgb_layer2_d)  # 0.75
        # torch.Size([1, 96, 96, 64]) torch.Size([1, 48, 48, 128]) torch.Size([1, 24, 24, 256]) torch.Size([1, 12, 12, 1024])

        rgb_layer0 = rgb_layer0.view(
            b, h // 4, w // 4, -1
        )  # torch.Size([1, 96, 96, 64])
        rgb_layer1 = rgb_layer1.view(
            b, h // 8, w // 8, -1
        )  # torch.Size([1, 48, 48, 128])
        rgb_layer2 = rgb_layer2.view(
            b, h // 16, w // 16, -1
        )  # torch.Size([1, 24, 24, 256])
        rgb_layer3 = rgb_layer3.view(
            b, h // 32, w // 32, -1
        )  # torch.Size([1, 12, 12, 1024])

        rgb_layer0 = rgb_layer0.permute(0, 3, 1, 2)
        rgb_layer1 = rgb_layer1.permute(0, 3, 1, 2)
        rgb_layer2 = rgb_layer2.permute(0, 3, 1, 2)
        rgb_layer3 = rgb_layer3.permute(0, 3, 1, 2)

        ####################################
        nir = self.pebed(nir)
        nir = self.pos_drop(nir)
        nir_layer0, nir_layer0_d = self.nir_layer0(nir)  # 3
        nir_layer1, nir_layer1_d = self.nir_layer1(nir_layer0_d)  # 1.5
        nir_layer2, nir_layer2_d = self.nir_layer2(nir_layer1_d)  # 0.75
        nir_layer3 = self.nir_layer3(nir_layer2_d)  # 0.75
        # torch.Size([1, 96, 96, 64]) torch.Size([1, 48, 48, 128]) torch.Size([1, 24, 24, 256]) torch.Size([1, 12, 12, 1024])

        nir_layer0 = nir_layer0.view(
            b, h // 4, w // 4, -1
        )  # torch.Size([1, 96, 96, 64])
        nir_layer1 = nir_layer1.view(
            b, h // 8, w // 8, -1
        )  # torch.Size([1, 48, 48, 128])
        nir_layer2 = nir_layer2.view(
            b, h // 16, w // 16, -1
        )  # torch.Size([1, 24, 24, 256])
        nir_layer3 = nir_layer3.view(
            b, h // 32, w // 32, -1
        )  # torch.Size([1, 12, 12, 1024])

        nir_layer0 = nir_layer0.permute(0, 3, 1, 2)
        nir_layer1 = nir_layer1.permute(0, 3, 1, 2)
        nir_layer2 = nir_layer2.permute(0, 3, 1, 2)
        nir_layer3 = nir_layer3.permute(0, 3, 1, 2)

        layer3 = self.mcc1(rgb_layer3, nir_layer3)
        layer2 = self.mcc2(rgb_layer2, nir_layer2)
        layer1 = self.mcc3(rgb_layer1, nir_layer1)
        layer0 = self.mcc4(rgb_layer0, nir_layer0)

        conv3m_out = self.conv3m(layer3)  # [1, 1024, 12, 12]

        conv3m_out_ = torch.cat((self.up_32(conv3m_out), layer2), dim=1)
        conv2m_out = self.conv2m(conv3m_out_)  # [1, 512, 24, 24]

        conv2m_out_ = torch.cat((self.up_21(conv2m_out), layer1), dim=1)
        conv1m_out = self.conv1m(conv2m_out_)  # [1, 256, 48, 48]

        conv1m_out_ = torch.cat((self.up_10(conv1m_out), layer0), dim=1)
        conv0m_out = self.conv0m(conv1m_out_)  # [1, 128, 96, 96]

        convfm_out = self.up_final(conv0m_out)  # [1, 64, 384, 384]

        final_pred = self.final_pred(convfm_out)

        layer3_pred = self.pred3(conv3m_out)
        layer2_pred = self.pred2(conv2m_out)
        layer1_pred = self.pred1(conv1m_out)
        layer0_pred = self.pred0(conv0m_out)
        layer3_pred = F.upsample(
            layer3_pred, size=input.size()[2:], mode="bilinear", align_corners=True
        )
        layer2_pred = F.upsample(
            layer2_pred, size=input.size()[2:], mode="bilinear", align_corners=True
        )
        layer1_pred = F.upsample(
            layer1_pred, size=input.size()[2:], mode="bilinear", align_corners=True
        )
        layer0_pred = F.upsample(
            layer0_pred, size=input.size()[2:], mode="bilinear", align_corners=True
        )
        fuse_feature = torch.cat(
            (input, layer0_pred, layer1_pred, layer2_pred, layer3_pred, final_pred),
            dim=1,
        )
        fuse_pred = self.fuse_pred(fuse_feature)

        return (
            F.sigmoid(layer3_pred),
            F.sigmoid(layer2_pred),
            F.sigmoid(layer1_pred),
            F.sigmoid(layer0_pred),
            F.sigmoid(final_pred),
            F.sigmoid(fuse_pred),
        )


if __name__ == "__main__":
    x = torch.randn(1, 3, 384, 384).cuda()
    x2 = torch.randn(1, 3, 384, 384).cuda()
    net = NIRNet(
        use_NIR=True,
    ).cuda()

    out = net(x, x2)
    print(out[0].shape)
