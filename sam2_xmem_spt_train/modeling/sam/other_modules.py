"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask.
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode = 'bilinear'):
        """
        ALPModule
        Args:
            proto_grid:     Grid size when doing multi-prototyping. For a 32-by-32 feature map, a size of 16-by-16 leads to a pooling window of 2-by-2
            feature_hw:     Spatial size of input feature map

        """
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [ ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)  ]
        self.avg_pool_op = nn.AvgPool2d( kernel_size  )

    def forward(self, qry, sup_x, sup_y, mode, thresh, isval = False, val_wsize = None, vis_sim = False, **kwargs):
        """
        Now supports
        Args:
            mode: 'mask'/ 'grid'. if mask, works as original prototyping
            qry: [way(1), nc, h, w]
            sup_x: [nb, nc, h, w]
            sup_y: [nb, 1, h, w]
            vis_sim: visualize raw similarities or not
        New
            mode:       'mask'/ 'grid'. if mask, works as original prototyping
            qry:        [way(1), nb(1), nc, h, w]:: query feature
            sup_x:      [way(1), shot, nb(1), nc, h, w]: support feature
            sup_y:      [way(1), shot, nb(1), h, w]: support mask (bg/fg)
            vis_sim:    visualize raw similarities or not
        """

        def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        if mode == 'mask': # class-level prototype only
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C

            proto = proto.mean(dim = 0, keepdim = True) # 1 X C, take the mean of everything
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps = 1e-4) * 20.0 # [1, h, w]

            vis_dict = {'proto_assign': None} # things to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict  # just a placeholder. pred_mask returned as [1, way(1), h, w]

        # no need to merge with gridconv+
        elif mode == 'gridconv': # using local prototypes only

            input_size = qry.shape # [way(1), nc, h, w]
            nch = input_size[1]

            sup_nshot = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  ) # [nshot, nc, h, w]

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0) # 1,nshot, hw, nc
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0) # 1,1,(nshot*hw),nc

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y) #[nshot, 1, h, w]
            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0) # 1,1,(nshot*hw)

            protos = n_sup_x[sup_y_g > thresh, :] # npro, nc
            pro_n = safe_norm(protos)
            qry_n = safe_norm(qry) # N x C x H' x W'

            # 用卷积计算proto和query feature之间的similarity
            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20 # N x npro x H' x W'

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True) # N x 1 x H' x W'
            debug_assign = dists.argmax(dim = 1).float().detach()

            vis_dict = {'proto_assign': debug_assign} # things to visualize

            if vis_sim: # return the similarity for visualization
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict


        elif mode == 'gridconv+': # local and global prototypes

            input_size = qry.shape
            nch = input_size[1]
            nb_q = input_size[0]

            sup_size = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  )

            sup_nshot = sup_x.shape[0]

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)

            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]

            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) #  1, nc

            pro_n = safe_norm( torch.cat( [protos, glb_proto], dim = 0 ) ) # npro+1, nc

            qry_n = safe_norm(qry)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20 # N，npro+1,H,W

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            raw_local_sims = dists.detach()


            debug_assign = dists.argmax(dim = 1).float()

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict

        else:
            raise NotImplementedError


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g_conv = nn.Conv2d(g_dims, mid_dim, kernel_size=1)

        self.transform = nn.Conv2d(mid_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g_conv(g)

        g = torch.cat([g, h], 1)

        # defined slightly differently than standard GRU,
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, self.hidden_dim:self.hidden_dim * 2])
        new_value = torch.tanh(values[:, self.hidden_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = nn.Conv2d(g_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 1)

        # defined slightly differently than standard GRU,
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, self.hidden_dim:self.hidden_dim * 2])
        new_value = torch.tanh(values[:, self.hidden_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value

        return new_h

#
# class ValueEncoder(nn.Module):
#     def __init__(self, value_dim, hidden_dim, single_object=False):
#         super().__init__()
#
#         self.single_object = single_object
#         network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
#         self.conv1 = network.conv1
#         self.bn1 = network.bn1
#         self.relu = network.relu  # 1/2, 64
#         self.maxpool = network.maxpool
#
#         self.layer1 = network.layer1  # 1/4, 64
#         self.layer2 = network.layer2  # 1/8, 128
#         self.layer3 = network.layer3  # 1/16, 256
#
#         self.distributor = MainToGroupDistributor()
#         self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
#         if hidden_dim > 0:
#             self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
#         else:
#             self.hidden_reinforce = None
#
#     def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
#         """
#
#         :param image: b,c,h,w
#         :param image_feat_f16: b,dim,h,w
#         :param h: b,num_objects,dim,h,w
#         :param masks: b,num_objects,h,w
#         :param others: b,num_objects,h,w
#         :param is_deep_update:
#         :return:
#         """
#         # image_feat_f16 is the feature from the key encoder
#         if not self.single_object:
#             g = torch.stack([masks, others], 2)  # b,num_objects,2,h,w
#         else:
#             g = masks.unsqueeze(2)
#         g = self.distributor(image, g)
#
#         batch_size, num_objects = g.shape[:2]
#         g = g.flatten(start_dim=0, end_dim=1)  # b*num_objects,2,h,w
#
#         g = self.conv1(g)
#         g = self.bn1(g)  # 1/2, 64
#         g = self.maxpool(g)  # 1/4, 64
#         g = self.relu(g)
#
#         g = self.layer1(g)  # 1/4
#         g = self.layer2(g)  # 1/8
#         g = self.layer3(g)  # 1/16
#
#         g = g.view(batch_size, num_objects, *g.shape[1:])  # b,num_objects,c1,h,w
#         g = self.fuser(image_feat_f16, g)  # feature reuse # b,num_objects,c1+c2,h,w
#
#         if is_deep_update and self.hidden_reinforce is not None:
#             h = self.hidden_reinforce(g, h)
#
#         return g, h
#
#
# class KeyEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         network = resnet.resnet50(pretrained=True)
#         self.conv1 = network.conv1
#         self.bn1 = network.bn1
#         self.relu = network.relu  # 1/2, 64
#         self.maxpool = network.maxpool
#
#         self.res2 = network.layer1  # 1/4, 256
#         self.layer2 = network.layer2  # 1/8, 512
#         self.layer3 = network.layer3  # 1/16, 1024
#
#     def forward(self, f):
#         x = self.conv1(f)
#         x = self.bn1(x)
#         x = self.relu(x)  # 1/2, 64
#         x = self.maxpool(x)  # 1/4, 64
#         f4 = self.res2(x)  # 1/4, 256
#         f8 = self.layer2(f4)  # 1/8, 512
#         f16 = self.layer3(f8)  # 1/16, 1024
#
#         return f16, f8, f4
#
#
# class UpsampleBlock(nn.Module):
#     def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
#         super().__init__()
#         self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
#         self.distributor = MainToGroupDistributor(method='add')
#         self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
#         self.scale_factor = scale_factor
#
#     def forward(self, skip_f, up_g):
#         skip_f = self.skip_conv(skip_f)
#         g = upsample_groups(up_g, ratio=self.scale_factor)
#         g = self.distributor(skip_f, g)
#         g = self.out_conv(g)
#         return g
#
#
# class KeyProjection(nn.Module):
#     def __init__(self, in_dim, keydim):
#         super().__init__()
#
#         self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
#         # shrinkage
#         self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
#         # selection
#         self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
#
#         nn.init.orthogonal_(self.key_proj.weight.data)
#         nn.init.zeros_(self.key_proj.bias.data)
#
#     def forward(self, x, need_s, need_e):
#         shrinkage = self.d_proj(x) ** 2 + 1 if (need_s) else None  # b*t,1,h,w
#         selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None
#
#         return self.key_proj(x), shrinkage, selection
#
#
# class Decoder(nn.Module):
#     def __init__(self, val_dim, hidden_dim):
#         super().__init__()
#
#         self.fuser = FeatureFusionBlock(1024, val_dim + hidden_dim, 512, 512)
#         if hidden_dim > 0:
#             self.hidden_update = HiddenUpdater([512, 256, 256 + 1], 256, hidden_dim)
#         else:
#             self.hidden_update = None
#
#         self.up_16_8 = UpsampleBlock(512, 512, 256)  # 1/16 -> 1/8
#         self.up_8_4 = UpsampleBlock(256, 256, 256)  # 1/8 -> 1/4
#
#         self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)
#
#     def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
#         batch_size, num_objects = memory_readout.shape[:2]
#
#         if self.hidden_update is not None:
#             g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
#         else:
#             g16 = self.fuser(f16, memory_readout)
#
#         g8 = self.up_16_8(f8, g16)
#         g4 = self.up_8_4(f4, g8)
#         logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))
#
#         if h_out and self.hidden_update is not None:
#             g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
#             hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
#         else:
#             hidden_state = None
#
#         logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
#         logits = logits.view(batch_size, num_objects, *logits.shape[-2:])
#
#         return hidden_state, logits
