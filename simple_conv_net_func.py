from __future__ import print_function
import torch
import numpy as np


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    # get shape of the input tensor
    batch_size, num_channels_in, height_in, width_in = x_in.size()
    # get shape of the weights
    num_channels_out, _, filter_height, filter_width = conv_weight.size()

    # compute shape of the output tensor
    height_out = height_in - filter_height + 1
    width_out = width_in - filter_width + 1

    # initialize output tensor
    x_out = torch.zeros((batch_size, num_channels_out, height_out, width_out)).to(device)

    # get output tensor
    for batch_idx in range(batch_size):
        for channel_out_idx in range(num_channels_out):
            for channel_in_idx in range(num_channels_in):
                for h in range(height_out):
                    for w in range(width_out):
                        for i in range(filter_height):
                            for j in range(filter_width):
                                x_out[batch_idx, channel_out_idx, h, w] += \
                                x_in[batch_idx, channel_in_idx, h + i - 1, w + j - 1]*conv_weight[channel_out_idx, channel_in_idx, i, j]
                                x_out[batch_idx, channel_out_idx, h, w] += conv_bias[channel_out_idx]

    return x_out


def conv2d_vector(x_in, conv_weight, conv_bias, device):
    # get shape of the input tensor
    batch_size, num_channels_in, height_in, width_in = x_in.size()
    # get shape of the weights
    num_channels_out, _, filter_height, filter_width = conv_weight.size()

    # compute shape of the output tensor
    height_out = height_in - filter_height + 1
    width_out = width_in - filter_width + 1
    
    # get im2col for input tensor
    x_im2col = im2col(x_in, filter_height, height_out, False, device) 
    
    # get im2col for weights
    conv_weight_rows = conv_weight2rows(conv_weight)
    
    # convolve
    x_out = torch.matmul(conv_weight_rows, x_im2col) + conv_bias.view(1, num_channels_out, 1)
    
    return x_out.view(batch_size, num_channels_out, height_out, width_out)


def im2col(X, kernel_size, output_size, pool, device):
    # get shape
    batch_size, num_channels_in, height, width = X.size()
    
    dim_out = output_size
    if pool:
        i0 = torch.arange(kernel_size).repeat(1, kernel_size).view(-1, kernel_size).transpose(0, 1).contiguous()
        i1 = torch.arange(0, height, 2).repeat(1, dim_out).view(-1, dim_out).transpose(0, 1).contiguous()
        i = i0.view(-1, 1) + i1.view(1, -1)
        j0 = torch.arange(kernel_size).repeat(kernel_size)
        j1 = torch.arange(0, height, 2).repeat(dim_out)
        j = j0.view(-1, 1) + j1.view(1, -1)
        return X[:, :, i, j]
    else:
        i0 = torch.arange(kernel_size).repeat(1, kernel_size).view(-1, kernel_size).transpose(0,1).repeat(1, num_channels_in)
        i1 = torch.arange(dim_out).repeat(1, dim_out).view(-1, dim_out).transpose(0, 1).contiguous()
        i = i0.view(-1, 1) + i1.view(1, -1)
        j0 = torch.arange(kernel_size).repeat(kernel_size * num_channels_in)
        j1 = torch.arange(dim_out).repeat(dim_out)
        j = j0.view(-1, 1) + j1.view(1, -1)
        k = torch.arange(num_channels_in).repeat(1, kernel_size*kernel_size).view(-1, kernel_size*kernel_size).transpose(0, 1).contiguous().view(-1, 1)
        return X[:, k, i, j]
    

def conv_weight2rows(conv_weight):
    num_channels_out, num_channels_in, filter_height, filter_width = conv_weight.size()
    conv_weight_rows = conv_weight.view(num_channels_out, num_channels_in * filter_height * filter_width)
    return conv_weight_rows


def pool2d_scalar(a, device):
    # get shape of input tensor
    batch_size, num_channels, height_in, width_in = a.size()

    # compute shape of output tensor
    height_out = int((height_in - 2)/2) + 1
    width_out = int((height_in - 2)/2) + 1

    # init output tensor
    a_out = torch.zeros((batch_size, num_channels, height_out, width_out)).to(device)

    # get output tensor
    for batch_idx in range(batch_size):
        for channel_idx in range(num_channels):
            for h in range(height_out):
                for w in range(width_out):
                    a_out[batch_idx, channel_idx, h, w] += np.max([a[batch_idx, channel_idx, h, w],
                                                                     a[batch_idx, channel_idx, h + 1, w],
                                                                     a[batch_idx, channel_idx, h, w + 1],
                                                                     a[batch_idx, channel_idx, h + 1, w + 1]])
                   
    return a_out


def pool2d_vector(a, device):
    # get shape
    batch_size, num_channels, height_in, width_in = a.size()
    
    # compute shape of output tensor
    height_out = int((height_in - 2)/2) + 1
    width_out = int((height_in - 2)/2) + 1

    # get im2col
    a_im2col = im2col(a, 2, height_out, True, device)
    
    # get maxpool
    a_out = a_im2col.max(2)[0]
    
    return a_out.view(batch_size, num_channels, height_out, width_out)


def relu_scalar(a, device):
    # get shape
    shape = a.size()
    
    # flatten tensor for better performance
    a = a.view(-1).to(device)
    
    # initialize output
    a_out = torch.zeros_like(a).to(device)
    
    # compute relu
    for i in range(a.size(0)):
        a_out[i] = torch.max(a[i], torch.zeros(1).to(device))

    # get the shape back
    a_out = a_out.view(shape)
    
    return a_out


def relu_vector(a, device):
    return torch.max(a, torch.zeros_like(a))


def reshape_vector(a, device):
    return a.view(a.size()[0], -1)


def reshape_scalar(a, device):
    # gets shape
    num_batches, num_channels, height, width = a.size()
    
    # compute new dimension
    l = num_channels * height * width
    
    # initialize output tensor
    a_out = torch.zeros([num_batches, l]).to(device)
    
    # compute result
    for batch_idx in range(num_batches):
        for channel_idx in range(num_channels):
            for h in range(height):
                for w in range(width):
                    l = channel_idx * height * width + height * h + w
                    a_out[batch_idx, l] = a[batch_idx, channel_idx, h, w]
    
    return a_out

def fc_layer_scalar(a, weight, bias, device):
    # get shape of the input tensor
    batch_size, dim_in = a.size()

    # get shape of the weights
    dim_out, _ = weight.size()
    
    # initialize output
    a_out = torch.zeros([batch_size, dim_out]).to(device)

    # compute fc
    for batch_idx in range(batch_size):
        for dim_idx in range(dim_out):
            a_out[batch_idx, dim_idx] = (a[batch_idx] * weight[dim_idx]).sum() + bias[dim_idx]
    
    return a_out


def fc_layer_vector(a, weight, bias, device):
    a_out = torch.matmul(a, weight.transpose(0, 1)) + bias
    return a_out