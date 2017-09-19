import numpy as np
import math


class SuperResolutionProvider(object):
    """
    provider for mobile traffic data (or other 3D data)
    ------------------------------------------------------
    :arg
    input_size: 3-element tuple(x*y*feature map), the shape of input we want
    output_size: 2-element tuple (x*y), the shape of output we want
    prediction_gap: int, the distant between the last input frame and output frame
    flatten: bool, whether flatten the output or not
    batchsize: int, the size of batch, default -1 (take all data generated)
    stride: 2-element tuple, the stride when selecting data
    shuffle: bool, default True. shuffle the data or not
    pad: 2-element or None, the size of padding, default None
    pad_value: float, padding values


    """

    def __init__(self, input_size, output_size, batchsize=-1, stride=(1, 1),
                 shuffle=True):

        self.input_size = input_size
        self.output_size = output_size
        self.batchsize = batchsize
        self.stride = stride
        self.shuffle = shuffle

    def DataSlicer_3D(self, inputs, excerpt):
        """
        generate data from input frames
        ------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        excerpt: list, the index of start frame of inputs
        flatten: bool, flatten target
        external: np.array (x*y*t), target from another resource, should has the shape (x'*y'*t)

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y)
        external_data: np.array, with dim (batchsize*1)
        """

        x_max, y_max, z_max = inputs.shape
        x_num = int(math.ceil((x_max - self.input_size[0] + 1.0) / self.stride[0]))
        y_num = int(math.ceil((y_max - self.input_size[1] + 1.0) / self.stride[1]))
        total = x_num * y_num * len(excerpt)
        if self.batchsize <= 0:
            self.batchsize = total
        input_data = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2], total))

        target_data = np.zeros((self.output_size[0], self.output_size[1], total))

        data_num = 0

        for frame in xrange(len(excerpt)):

            input_frame = inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]
            target_frame = inputs[:, :, excerpt[frame] + self.input_size[2] - 1]

            for x in xrange(self.input_size[0], x_max + 1, self.stride[0]):
                for y in xrange(self.input_size[1], y_max + 1, self.stride[1]):
                    input_data[:, :, :, data_num] = input_frame[x - self.input_size[0]:x, y - self.input_size[1]:y, :]
                    target_data[:, :, data_num] = target_frame[x - self.input_size[0]:
                        x - self.input_size[0] + self.output_size[0], y - self.input_size[1]:
                            y - self.input_size[1] + self.output_size[1]]

                    data_num += 1

        if self.shuffle:
            indices = np.random.permutation(total)
            return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, indices[0:self.batchsize]], (2, 0, 1)).reshape(self.batchsize, -1))

        else:
            return (np.transpose(input_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, 0:self.batchsize], (2, 0, 1)).reshape(self.batchsize, -1))

    def feed(self, inputs, framebatch, mean=0, std=1, norm_tar=False):
        """
        iterate over mini-batch
        --------------------------------------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        framebatch: int, maximum frames we selected in one mini-batch
        mean: float, inputs normalized constant, mean
        std: float, inputs normalized constant, standard error
        norm_tar: bool, target normalized as input, default False

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y) or flatten one
        """

        frame_max = inputs.shape[2] - self.input_size[2] + 1

        if self.shuffle:
            indices = np.random.permutation(frame_max)

        for start_idx in range(0, frame_max, framebatch):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + framebatch]
            else:
                excerpt = range(start_idx, min((start_idx + framebatch), frame_max))

            net_inputs, net_targets = self.DataSlicer_3D(inputs=inputs, excerpt=excerpt)
            if norm_tar:
                net_targets = ((net_targets - mean) / float(std))
            yield (net_inputs - mean) / float(std), net_targets


class SpecialSuperResolutionProvider(object):
    """
    provider for mobile traffic data (or other 3D data)
    ------------------------------------------------------
    :arg
    input_size: 3-element tuple(x*y*feature map), the shape of input we want
    output_size: 2-element tuple (x*y), the shape of output we want
    prediction_gap: int, the distant between the last input frame and output frame
    flatten: bool, whether flatten the output or not
    batchsize: int, the size of batch, default -1 (take all data generated)
    stride: 2-element tuple, the stride when selecting data
    shuffle: bool, default True. shuffle the data or not
    pad: 2-element or None, the size of padding, default None
    pad_value: float, padding values


    """

    def __init__(self, input_size, output_size, batchsize=-1, stride=(1, 1),
                 shuffle=True):

        self.input_size = input_size
        self.output_size = output_size
        self.batchsize = batchsize
        self.stride = stride
        self.shuffle = shuffle

    def special(self, inputs, keepdims=True):

        if not keepdims:
            output = np.zeros((20, 20, inputs.shape[-2], inputs.shape[-1]))
        else:
            output = np.zeros(inputs.shape)

        index_s = np.zeros((20, 20))
        index_l = np.zeros((80, 80))

        index10s = [(0, 0), (0, 3), (0, 6), (0, 9), (0, 10), (0, 13), (0, 16), (0, 19), (3, 0), (3, 19), (6, 0),
                    (6, 19), (9, 0), (9, 19), (10, 0), (10, 19),
                    (13, 0), (13, 19), (16, 0), (16, 19), (19, 0), (19, 3), (19, 6), (19, 9), (19, 10), (19, 13),
                    (19, 16), (19, 19)]

        index10l = [(0, 0), (0, 10), (0, 20), (0, 30), (0, 40), (0, 50), (0, 60), (0, 70), (10, 0), (10, 70), (20, 0),
                    (20, 70), (30, 0), (30, 70), (40, 0), (40, 70),
                    (50, 0), (50, 70), (60, 0), (60, 70), (70, 0), (70, 10), (70, 20), (70, 30), (70, 40), (70, 50),
                    (70, 60), (70, 70)]

        index_s[3:17, 3:17] = 1
        index_l[26:54, 26:54] = 1

        for x in xrange(26, 54, 2):
            for y in xrange(26, 54, 2):
                small_x = (x - 20) / 2
                small_y = (y - 20) / 2
                if not keepdims:
                    output[small_x, small_y, :, :] = np.mean(inputs[x:x + 2, y:y + 2, :, :], axis=(0, 1))
                else:
                    output[x:x + 2, y:y + 2, :, :] = np.mean(inputs[x:x + 2, y:y + 2, :, :], axis=(0, 1), keepdims=True)

        for i in xrange(len(index10s)):
            large_x = index10l[i][0]
            large_y = index10l[i][1]
            if not keepdims:
                output[index10s[i][0], index10s[i][1], :, :] = np.mean(
                    inputs[large_x:large_x + 10, large_y:large_y + 10, :, :], axis=(0, 1))
            else:
                output[large_x:large_x + 10, large_y:large_y + 10, :, :] = np.mean(
                    inputs[large_x:large_x + 10, large_y:large_y + 10, :, :], axis=(0, 1), keepdims=True)

            index_l[large_x:large_x + 10, large_y:large_y + 10] = 1
            index_s[index10s[i]] = 1

        residx_s = np.array(np.where(index_s == 0))

        for i in xrange(176):
            select_s = residx_s[:, i]
            select_l = np.unravel_index(np.argmin(index_l), (80, 80))
            if not keepdims:
                output[select_s[0], select_s[1], :, :] = np.mean(
                    inputs[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :, :], axis=(0, 1))
            else:
                output[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :] = np.mean(
                    inputs[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :, :], axis=(0, 1),
                    keepdims=True)
            index_l[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4] = 1

        return output

    def DataSlicer_3D(self, inputs, excerpt, keepdims):
        """
        generate data from input frames
        ------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        excerpt: list, the index of start frame of inputs
        flatten: bool, flatten target
        external: np.array (x*y*t), target from another resource, should has the shape (x'*y'*t)

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y)
        external_data: np.array, with dim (batchsize*1)
        """

        x_max, y_max, z_max = inputs.shape
        x_num = int(math.ceil((x_max - self.input_size[0] + 1.0) / self.stride[0]))
        y_num = int(math.ceil((y_max - self.input_size[1] + 1.0) / self.stride[1]))
        total = x_num * y_num * len(excerpt)
        if self.batchsize <= 0:
            self.batchsize = total
        input_data = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2], total))

        target_data = np.zeros((self.output_size[0], self.output_size[1], total))

        data_num = 0

        for frame in xrange(len(excerpt)):

            input_frame = inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]
            target_frame = inputs[:, :, excerpt[frame] + self.input_size[2] - 1]

            for x in xrange(self.input_size[0], x_max + 1, self.stride[0]):
                for y in xrange(self.input_size[1], y_max + 1, self.stride[1]):
                    input_data[:, :, :, data_num] = input_frame[x - self.input_size[0]:x, y - self.input_size[1]:y, :]
                    target_data[:, :, data_num] = target_frame[x - self.input_size[0]:
                        x - self.input_size[0] + self.output_size[0], y - self.input_size[1]:
                            y - self.input_size[1] + self.output_size[1]]

                    data_num += 1

        if self.shuffle:
            indices = np.random.permutation(total)
            return (np.transpose(self.special(input_data[:, :, :, indices[0:self.batchsize]], keepdims=keepdims), (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, indices[0:self.batchsize]], (2, 0, 1)).reshape(self.batchsize, -1))

        else:
            return (np.transpose(self.special(input_data[:, :, :, 0:self.batchsize], keepdims=keepdims), (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, 0:self.batchsize], (2, 0, 1)).reshape(self.batchsize, -1))

    def feed(self, inputs, framebatch, mean=0, std=1, norm_tar=False, keepdims=False):
        """
        iterate over mini-batch
        --------------------------------------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        framebatch: int, maximum frames we selected in one mini-batch
        mean: float, inputs normalized constant, mean
        std: float, inputs normalized constant, standard error
        norm_tar: bool, target normalized as input, default False

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y) or flatten one
        """

        frame_max = inputs.shape[2] - self.input_size[2] + 1

        if self.shuffle:
            indices = np.random.permutation(frame_max)

        for start_idx in range(0, frame_max, framebatch):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + framebatch]
            else:
                excerpt = range(start_idx, min((start_idx + framebatch), frame_max))

            net_inputs, net_targets = self.DataSlicer_3D(inputs=inputs, excerpt=excerpt, keepdims=keepdims)
            if norm_tar:
                net_targets = ((net_targets - mean) / float(std))
            yield (net_inputs - mean) / float(std), net_targets


class MoverProvider(object):
    """
    provider for data in 2 files
    ------------------------------------------------------
    :arg
    batchsize: int, the size of batch, default -1 (take all data generated)
    shuffle: bool, default True. shuffle the data or not
    """

    def __init__(self, length):

        self.length = length

    def special(self, inputs, keepdims=False):

        if not keepdims:
            output = np.zeros((20, 20, inputs.shape[-2], inputs.shape[-1]))
        else:
            output = np.zeros(inputs.shape)

        index_s = np.zeros((20, 20))
        index_l = np.zeros((80, 80))

        index10s = [(0, 0), (0, 3), (0, 6), (0, 9), (0, 10), (0, 13), (0, 16), (0, 19), (3, 0), (3, 19), (6, 0),
                    (6, 19), (9, 0), (9, 19), (10, 0), (10, 19),
                    (13, 0), (13, 19), (16, 0), (16, 19), (19, 0), (19, 3), (19, 6), (19, 9), (19, 10), (19, 13),
                    (19, 16), (19, 19)]

        index10l = [(0, 0), (0, 10), (0, 20), (0, 30), (0, 40), (0, 50), (0, 60), (0, 70), (10, 0), (10, 70), (20, 0),
                    (20, 70), (30, 0), (30, 70), (40, 0), (40, 70),
                    (50, 0), (50, 70), (60, 0), (60, 70), (70, 0), (70, 10), (70, 20), (70, 30), (70, 40), (70, 50),
                    (70, 60), (70, 70)]

        index_s[3:17, 3:17] = 1
        index_l[26:54, 26:54] = 1

        for x in xrange(26, 54, 2):
            for y in xrange(26, 54, 2):
                small_x = (x - 20) / 2
                small_y = (y - 20) / 2
                if not keepdims:
                    output[small_x, small_y, :, :] = np.mean(inputs[x:x + 2, y:y + 2, :, :], axis=(0, 1))
                else:
                    output[x:x + 2, y:y + 2, :, :] = np.mean(inputs[x:x + 2, y:y + 2, :, :], axis=(0, 1), keepdims=True)

        for i in xrange(len(index10s)):
            large_x = index10l[i][0]
            large_y = index10l[i][1]
            if not keepdims:
                output[index10s[i][0], index10s[i][1], :, :] = np.mean(
                    inputs[large_x:large_x + 10, large_y:large_y + 10, :, :], axis=(0, 1))
            else:
                output[large_x:large_x + 10, large_y:large_y + 10, :, :] = np.mean(
                    inputs[large_x:large_x + 10, large_y:large_y + 10, :, :], axis=(0, 1), keepdims=True)

            index_l[large_x:large_x + 10, large_y:large_y + 10] = 1
            index_s[index10s[i]] = 1

        residx_s = np.array(np.where(index_s == 0))

        for i in xrange(176):
            select_s = residx_s[:, i]
            select_l = np.unravel_index(np.argmin(index_l), (80, 80))
            if not keepdims:
                output[select_s[0], select_s[1], :, :] = np.mean(
                    inputs[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :, :], axis=(0, 1))
            else:
                output[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :] = np.mean(
                    inputs[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :, :], axis=(0, 1),
                    keepdims=True)
            index_l[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4] = 1

        return output

    def feed(self, inputs, targets, special=False, keepdims=False):

        """
        generate data from input files
        ------------------------------------------------
        :arg
        inputs: np.array, the source input data
        targets: np.array, the source target data

        :return
        input: np.array
        target: np.array
        """

        for start_idx in range(inputs.shape[-1] - self.length + 1):
            excerpt = range(start_idx, start_idx + self.length)
            if special:
                input_frame = self.special(np.expand_dims(inputs[:, :, excerpt], axis=-1), keepdims=keepdims)
                yield np.transpose(input_frame, (3, 2, 0, 1)), targets[:, :, excerpt[-1]]

            else:
                input_frame = np.transpose(inputs[:, :, excerpt], (2, 0, 1))
                yield np.expand_dims(input_frame, axis=0), targets[:, :, excerpt[-1]]
