import torch
import torch.nn as nn


class Fuser(nn.Module):
    """
    Fuse the two input images.
    """

    def __init__(self, loop_num=3, feather_num=64, fine_tune=False):
        super().__init__()
        self.loop_num = loop_num
        self.fine_tune = fine_tune

        # attention layer
        self.att_a_conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.att_b_conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

        # dilation conv layer
        self.dil_conv_1 = nn.Sequential(nn.Conv2d(3, feather_num, 3, 1, 1, 1), nn.BatchNorm2d(feather_num), nn.ReLU())
        self.dil_conv_2 = nn.Sequential(nn.Conv2d(3, feather_num, 3, 1, 2, 2), nn.BatchNorm2d(feather_num), nn.ReLU())
        self.dil_conv_3 = nn.Sequential(nn.Conv2d(3, feather_num, 3, 1, 3, 3), nn.BatchNorm2d(feather_num), nn.ReLU())

        # fuse conv layer
        self.fus_conv = nn.Sequential(nn.Conv2d(3 * feather_num, 1, 3, padding=1), nn.BatchNorm2d(1), nn.Tanh())

    def forward(self, im_p):
        """
        :param im_p: image pair
        """

        # unpack im_p
        im_a, im_b = im_p

        # recurrent sub network
        # generate f_0 with manual function
        im_f = [torch.max(im_a, im_b)]  # init im_f_0
        att_a = []
        att_b = []

        # loop in sub network
        for e in range(self.loop_num):
            im_f_x, att_a_x, att_b_x = self._sub_forward(im_a, im_b, im_f[-1])
            im_f.append(im_f_x)
            att_a.append(att_a_x)
            att_b.append(att_b_x)

        # return im_f, att list
        return im_f, att_a, att_b

    def _sub_forward(self, im_a, im_b, im_f):
        # attention
        att_a = self._attention(self.att_a_conv, im_a, im_f)
        att_b = self._attention(self.att_b_conv, im_b, im_f)
        att_a = att_a.detach() if self.fine_tune else att_a
        att_b = att_b.detach() if self.fine_tune else att_b

        # focus on attention
        im_a_att = im_a * att_a
        im_b_att = im_b * att_b

        # image concat
        im_cat = torch.cat([im_a_att, im_f, im_b_att], dim=1)
        im_cat = im_cat.detach() if self.fine_tune else im_cat

        # dilation
        dil_1 = self.dil_conv_1(im_cat)
        dil_2 = self.dil_conv_2(im_cat)
        dil_3 = self.dil_conv_3(im_cat)

        # feather concat
        f_cat = torch.cat([dil_1, dil_2, dil_3], dim=1)

        # fuse
        im_f_n = self.fus_conv(f_cat)

        return im_f_n, att_a, att_b

    @staticmethod
    def _attention(att_conv, im_x, im_f):
        x = torch.cat([im_x, im_f], dim=1)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([x_max, x_avg], dim=1)
        x = att_conv(x)
        return torch.sigmoid(x)
