import torch
from unet import UNet
from kornia.geometry.transform import Resize
import numpy as np
import math
from sklearn.cluster import KMeans

class ThresholdSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=0.5):
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input


class ThresholdSTEGEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=0.5):
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input


def mask_clustering(mask_ps, num_iters=5, n_patch=3):
    mask = mask_ps[0, 0].detach().cpu().numpy()
    idx_x, idx_y = mask.nonzero()
    idx_tuple = [[x, y] for x, y in zip(idx_x, idx_y)]
    if len(idx_tuple) > 2:
        kmeans = KMeans(n_clusters=n_patch, max_iter=num_iters).fit(idx_tuple)
        label = kmeans.labels_
        mask_clustered = []
        for i in range(n_patch):
            idx_x_i, idx_y_i = idx_x[label==i], idx_y[label==i]
            mask_clustered_i = np.zeros_like(mask)
            mask_clustered_i[idx_x_i, idx_y_i] = 1
            mask_clustered_i = torch.from_numpy(mask_clustered_i).unsqueeze(0).unsqueeze(0).cuda()
            mask_clustered.append(mask_clustered_i)
    else:
        mask_clustered = [torch.zeros_like(mask_ps).cuda() for _ in range(n_patch)]
    return mask_clustered



def ShapeCompletionL1KnownSquare(mask, size, l1_thresh, soft=False):
    mask = mask[0, 0].double()
    masksum = mask.sum()
    corner_sums = mask.cumsum(dim=0).cumsum(dim=1)
    included_sums = torch.zeros_like(mask)
    included_sums[:-size, :-size] = corner_sums[:-size, :-size] + corner_sums[size:, size:] - corner_sums[:-size,
                                                                                              size:] - corner_sums[
                                                                                                       size:, :-size]
    external_sum = masksum - included_sums
    l1_dists = size * size - included_sums + external_sum
    if (soft):
        is_to_be_masked = ThresholdSTEFunction.apply(1.5 - l1_dists.float() / (size * size) - l1_thresh)
        t = 10 * math.log(100)
        is_to_be_masked_corners = torch.logcumsumexp(torch.logcumsumexp(is_to_be_masked * t, dim=1), dim=0) / t
        out_mask = torch.zeros_like(mask)
        out_mask_2 = torch.zeros_like(mask)
        q1 = torch.max(torch.stack(
            [t * is_to_be_masked_corners[:-size, :-size], t * is_to_be_masked_corners[size:, size:],
             t * is_to_be_masked_corners[:-size, size:], t * is_to_be_masked_corners[size:, :-size]]), dim=0)[0]
        out_mask[size:, size:] = torch.exp(t * is_to_be_masked_corners[:-size, :-size] - q1) + torch.exp(
            t * is_to_be_masked_corners[size:, size:] - q1) - torch.exp(
            t * is_to_be_masked_corners[:-size, size:] - q1) - torch.exp(
            t * is_to_be_masked_corners[size:, :-size] - q1)
        out_mask_2[size:, size:] = (torch.log(out_mask[size:, size:]) + q1) / t
        q2 = \
        torch.max(torch.stack([t * is_to_be_masked_corners[size:, :size], t * is_to_be_masked_corners[:-size, :size]]),
                  dim=0)[0]
        out_mask[size:, :size] = torch.exp(t * is_to_be_masked_corners[size:, :size] - q2) - torch.exp(
            t * is_to_be_masked_corners[:-size, :size] - q2)
        out_mask_2[size:, :size] = (torch.log(out_mask[size:, :size]) + q2) / t
        q3 = \
        torch.max(torch.stack([t * is_to_be_masked_corners[:size, size:], t * is_to_be_masked_corners[:size, :-size]]),
                  dim=0)[0]
        out_mask[:size, size:] = torch.exp(t * is_to_be_masked_corners[:size, size:]) - torch.exp(
            t * is_to_be_masked_corners[:size, :-size])
        out_mask_2[:size, size:] = (torch.log(out_mask[:size, size:]) + q3) / t
        out_mask_2[:size, :size] = is_to_be_masked_corners[:size, :size]
        return ThresholdSTEGEFunction.apply(out_mask / 2).reshape((1, 1, mask.shape[0], mask.shape[1]))
    else:
        is_to_be_masked = ThresholdSTEFunction.apply(1.5 - l1_dists.float() / (size * size) - l1_thresh)
        is_to_be_masked_corners = is_to_be_masked.cumsum(dim=0).cumsum(dim=1)
        out_mask = torch.zeros_like(mask)
        out_mask[size:, size:] = is_to_be_masked_corners[:-size, :-size] + is_to_be_masked_corners[size:,
                                                                           size:] - is_to_be_masked_corners[:-size,
                                                                                    size:] - is_to_be_masked_corners[
                                                                                             size:, :-size]
        out_mask[size:, :size] = is_to_be_masked_corners[size:, :size] - is_to_be_masked_corners[:-size, :size]
        out_mask[:size, size:] = is_to_be_masked_corners[:size, size:] - is_to_be_masked_corners[:size, :-size]
        out_mask[:size, :size] = is_to_be_masked_corners[:size, :size]
        return ThresholdSTEGEFunction.apply(out_mask / 2).reshape((1, 1, mask.shape[0], mask.shape[1]))


def ShapeCompletionL1(mask, init_l1_thresh=0.9, thresh_decay=0.7, break_its=15, square_sizes=[100, 75, 50, 25],
                      soft=False):
    # print("square_sizes - ", square_sizes)
    # print("break_its - ", break_its)
    if (break_its == 0):
        print("No Patch")
        return torch.zeros_like(mask)
    else:
        if (soft):
            combined_masks = ThresholdSTEGEFunction.apply(torch.stack(
                list([ShapeCompletionL1KnownSquare(mask, x, init_l1_thresh, soft=True) for x in square_sizes])).sum(
                dim=0) / 2.)
            if (combined_masks.sum() > 0):
                return combined_masks
            else:
                return ShapeCompletionL1(mask, init_l1_thresh=init_l1_thresh * thresh_decay, thresh_decay=thresh_decay,
                                         break_its=break_its - 1, square_sizes=square_sizes, soft=True)
        else:
            combined_masks = ThresholdSTEGEFunction.apply(
                torch.stack(list([ShapeCompletionL1KnownSquare(mask, x, init_l1_thresh) for x in square_sizes])).sum(
                    dim=0) / 2.)
            if (combined_masks.sum() > 0):
                return combined_masks
            else:
                return ShapeCompletionL1(mask, init_l1_thresh=init_l1_thresh * thresh_decay, thresh_decay=thresh_decay,
                                         break_its=break_its - 1, square_sizes=square_sizes)


def ShapeCompletionMultiPatch(mask, init_l1_thresh=0.9, thresh_decay=0.7, break_its=15, square_sizes=[100, 75, 50, 25],
                      soft=False, n_patch=3):
    mask_clustered = mask_clustering(mask, n_patch=n_patch)
    mask_completed = torch.zeros_like(mask).cuda()
    for i in range(n_patch):
        patch_i = mask * mask_clustered[i]
        patch_i_completed = ShapeCompletionL1(patch_i, init_l1_thresh=init_l1_thresh, thresh_decay=thresh_decay,
                                              break_its=break_its, square_sizes=square_sizes, soft=soft)
        mask_completed += patch_i_completed
    return mask_completed


def ShapeCompletion(mask):
    _, _, idx1, idx2 = mask.nonzero(as_tuple=True)  # assume batch size is 1
    if idx1.size()[0] > 500:
        xmin = idx1.min().cpu().numpy()
        ymin = idx2.min().cpu().numpy()
        xmax = idx1.max().cpu().numpy()
        ymax = idx2.max().cpu().numpy()
        xmin = int(np.floor(xmin))
        ymin = int(np.floor(ymin))
        xmax = int(np.ceil(xmax))
        ymax = int(np.ceil(ymax))
        mask = torch.zeros_like(mask)
        mask[:, :, xmin:xmax, ymin:ymax] = 1
    return mask


class PatchDetector(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, base_filter=64, image_size=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), unet=None,
                 square_sizes=[100, 75, 50, 25], n_patch=1, mask_type = 0, iou_test = False, offsets = None, black_vals = [0.0, 0.0, 0.0]):
        super(PatchDetector, self).__init__()
        if unet:
            self.unet = unet
        else:
            self.unet = UNet(n_channels, n_classes, bilinear=bilinear, base_filter=base_filter)
        self.image_size = image_size
        self.device = device
        self.square_sizes = square_sizes
        self.n_patch = n_patch
        self.mask_type = mask_type
        self.iou_test = iou_test
        self.offsets = offsets

    def forward(self, image_list, bpda=False, shape_completion=False, simple_shape_completion=False,
                soft_shape_completion=False, union=False):
        output_list = []
        mask_list = []  # mask: 1 patch; 0 background
        raw_mask_list = []
        for x in image_list:
            # print("x - ", x.shape)
            h, w = x.shape[1:]
            # print("h - ", h)
            # print("w - ", w)
            x = x.unsqueeze(0).to(self.device)
            # print("x after unsqueeze - ", x.shape)
            if self.image_size:
                x_tensor = Resize((self.image_size[0], self.image_size[1]))(x)
                # print("x_tensor after resize - ", x_tensor.shape)
            else:
                x_tensor = x
            # print("dgsk")
            mask_out = self.unet(x_tensor)
            
            # print("mask_out values - ", mask_out)
            
            # print("mask_out - ", mask_out.shape)
            mask = Resize((h, w))(mask_out)
            # print("mask after resize - ", mask.shape)
            mask = torch.sigmoid(mask)
            
            raw_mask = (mask > 0.5).float()
            # print("raw_mask - ", raw_mask.shape)
            raw_mask_list.append(raw_mask)
            mask = raw_mask
            if shape_completion and not simple_shape_completion:
                if self.n_patch > 1:
                    mask = ShapeCompletionMultiPatch(mask, n_patch=self.n_patch, soft=soft_shape_completion)
                else:
                    mask = ShapeCompletionL1(mask, soft=soft_shape_completion, square_sizes = self.square_sizes, break_its=15)
                # print("shape_completion")
            elif simple_shape_completion:
                mask = ShapeCompletion(mask)
                # print("simple_shape_completion")
            if union:
                mask = mask + raw_mask - mask * raw_mask
                # print("union")
            # print("mask after shape completion - ", mask.shape)
            mask_list.append(mask)
            mask = torch.cat((mask, mask, mask), 1)
            # print("mask after cat - ", mask.shape)
            mask = 1.0 - mask
            # Mask types
            # Default 0 : Pixel value 0 - Black mask
            # 1 : Average of non-noise pixels
            # ones = 0
            # zeros = 0
            # for C in range(mask.shape[1]):
            #     for H in range(mask.shape[2]):
            #         for W in range(mask.shape[3]):
            #             # print(mask[0,C,H,W])
            #             if(mask[0,C,H,W].detach().numpy() == 0.0):
            #                 zeros += 1
            #             if(mask[0,C,H,W].detach().numpy() == 1.0):
            #                 ones += 1
            # print("zeros - ", zeros)
            # print("ones - ", ones)
            # print("count - ", torch.bincount(mask))
            # print("mask type - ", type(mask))
            
            add_mask = torch.zeros(mask[0].shape)
            mask_dup = mask
            if(self.mask_type == 1):
                avgs = []
                for C in range(x.shape[1]):
                    sum_ = 0.0
                    count = 0
                    for H in range(x.shape[2]):
                        for W in range(x.shape[3]):
                            if(mask_dup[0,C,H,W].detach().numpy() != 0.0):
                                sum_ += x[0,C,H,W].detach().numpy()
                                count += 1
                    # print("mask --- ")
                    # print(mask[0])
                    print("sum - ", sum_)
                    print("count - ", count)
                    if(count!=0):
                        avg = sum_/count
                    else:
                        avg = 0.0
                    avgs.append(avg)
                # mask_T = 1.0 - mask
                mask_np = mask_dup.detach().numpy()
                for i in range(mask_np.shape[1]):
                    mask_np[0,i,:,:] = np.where(mask_np[0,i,:,:] == 0.0, avgs[i], black_vals[i])
                add_mask = torch.tensor(mask_np, dtype=torch.float32)
                
            iou_val = None
            
            if(self.iou_test == True):
                mask_ = (1.0 - mask).detach().numpy()
                mask_ = np.where(mask_ == 0.0, 2, mask_)
                mask_ = mask_[0]
                # mask_ = 2 * np.ones(mask[0].shape)
                # for i in range(mask.shape[0]):
                #     mask_ = np.where(mask[i] == 0, -1, mask_[i])
                A1, A2, B1, B2 = self.offsets[0], self.offsets[1], self.offsets[2], self.offsets[3]
                input_ = 1 * np.ones(x[0].shape)
                input_[:,A1:B1,A2:B2] = 2
                prod = input_[0] * mask_[0]
                # print("input_ shape - ", input_.shape)
                # print("mask_.shape - ", mask_.shape)
                # print("prod shape - ", prod.shape)
                
                uni = 0
                inter = 0
                # for C in range(mask_.shape[0]):
                for H in range(mask_.shape[1]):
                    for W in range(mask_.shape[2]):
                        if(prod[H,W] == 4):
                            inter += 1
                        if(prod[H,W] >= 2):
                            uni += 1
                if(uni == 0):
                    iou_val = 0.0
                else:
                    iou_val = (inter/uni)*10
                
            # print("mask types - ", type(mask))
            # mask_temp = mask.detach().numpy()
            # for C in range(mask.shape[1]):
            #     if(mask[])
            output_list.append(x[0] * mask[0] + add_mask)
            
        return output_list, mask_list, raw_mask_list, iou_val
