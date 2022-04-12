# Ke Chen
# knutchen@ucsd.edu
# Zero-shot Audio Source Separation via Query-based Learning from Weakly-labeled Data
# Some Common Methods

import numpy as np
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import logging
import os
import sys
import h5py
import csv
import time
import json
import museval
import librosa
from datetime import datetime

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
                
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths

def np_to_pytorch(x, device = None):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x
    return x.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_average_energy(x):
    return np.mean(np.square(x))

def id_to_one_hot(id, classes_num):
    one_hot = np.zeros(classes_num)
    one_hot[id] = 1
    return one_hot

def ids_to_hots(ids, classes_num):
    hots = np.zeros(classes_num)
    for id in ids:
        hots[id] = 1
    return hots

def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    
def collect_fn(list_data_dict):
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    return np_data_dict

def dump_config(config, filename, include_time = False):
    save_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_json = {}
    for key in dir(config):
        if not key.startswith("_"):
            config_json[key] = eval("config." + key)
    if include_time:
        filename = filename + "_" + save_time
    with open(filename + ".json", "w") as f:      
        json.dump(config_json, f ,indent=4)


def get_segment_bgn_end_samples(anchor_index, segment_frames, hop_samples, clip_samples):
    bgn_frame = anchor_index - segment_frames // 2
    end_frame = anchor_index + segment_frames // 2
    bgn_sample = bgn_frame * hop_samples
    end_sample = end_frame * hop_samples

    segment_samples = segment_frames * hop_samples

    if bgn_sample < 0:
        bgn_sample = 0
        end_sample = segment_samples

    if end_sample > clip_samples:
        bgn_sample = clip_samples - segment_samples
        end_sample = clip_samples

    return bgn_sample, end_sample

def get_mix_data(waveforms, con_vectors, class_ids, indexes, mix_type = "mixture"):
    # define return data
    mixtures = []
    sources = []
    conditions = []
    gds = []
    for i in range(0, len(indexes), 2):
        n1 = indexes[i]
        n2 = indexes[i + 1]
        # energy normalization
        e1 = np.mean(np.square(waveforms[n1]))
        e2 = np.mean(np.square(waveforms[n2]))
        ratio = (e1 / max(1e-8, e2)) ** 0.5
        ratio = np.clip(ratio, 0.02, 50)
        waveforms[n2] *= ratio
        mixture = waveforms[n1] + waveforms[n2]
        # form data
        if mix_type == "clean":
            mixtures.append(waveforms[n1])
            mixtures.append(waveforms[n2])
            sources.append(waveforms[n1])
            sources.append(waveforms[n2])
        elif mix_type == "silence":
            mixtures.append(waveforms[n2])
            mixtures.append(waveforms[n1])
            sources.append(np.zeros_like(waveforms[n1]))
            sources.append(np.zeros_like(waveforms[n2]))
        else:
            mixtures.append(mixture)
            mixtures.append(mixture)
            sources.append(waveforms[n1])
            sources.append(waveforms[n2])
    
        conditions.append(con_vectors[n1])
        conditions.append(con_vectors[n2])
        gds.append(class_ids[n1])
        gds.append(class_ids[n2])
    return mixtures, sources, conditions, gds

# generate a list 
def get_balanced_class_list(index_path, factor = 3, black_list = None, random_seed = 0):
    # initialization
    random_state = np.random.RandomState(random_seed)
    logging.info("Load Indexes...............")
    with h5py.File(index_path, "r") as hf:
        indexes = hf["index_in_hdf5"][:]
        targets = hf["target"][:].astype(np.float32)
    (audios_num, classes_num) = targets.shape
    # set the indexes per class for balanced list
    indexes_per_class = []
    for k in range(classes_num):
        indexes_per_class.append(
            np.where(targets[:, k] == 1)[0]
        )

    logging.info("Load Indexes Succeed...............")

    return indexes_per_class

def dataset_worker_init_fn_seed(worker_id):
    seed = np.random.randint(0, 224141) + worker_id * np.random.randint(100,1000)
    print(seed)
    np.random.seed(seed)

def calculate_sdr(ref, est, scaling=False):
    s = museval.evaluate(ref[None,:,None], est[None,:,None], win = len(ref), hop = len(ref))
    return s[0][0]

def butter_lowpass_filter(data, cuton, cutoff, fs, order):
    normal_cutoff = cutoff / (0.5 * fs)
    normal_cuton = cuton / (0.5 * fs)
    b, a = butter(order, [normal_cuton, normal_cutoff], btype="band", analog=False)
    y = filtfilt(b,a, data)
    return y

def calculate_silence_sdr(mixture, est):
    sdr = 10. * (
        np.log10(np.clip(np.mean(mixture ** 2), 1e-8, np.inf)) \
        - np.log10(np.clip(np.mean(est ** 2), 1e-8, np.inf)))
    return sdr


def evaluate_sdr(ref, est, class_ids, mix_type = "mixture"):
    sdr_results = []
    if mix_type == "silence":
        for i in range(len(ref)):
            sdr = calculate_silence_sdr(ref[i,:,0], est[i,:,0])
            sdr_results.append([sdr, class_ids[i]])
    else:
        for i in range(len(ref)):
            if np.sum(ref[i,:,0]) == 0 or np.sum(est[i,:,0]) == 0:
                continue
            else:
                sdr_c = calculate_sdr(ref[i,:,0], est[i,:,0], scaling = True)
            sdr_results.append([sdr_c, class_ids[i]])
    return sdr_results

# set the audio into the format that can be fed into the model
# resample -> convert to mono -> output the audio  
# track [n_sample, n_channel]
def prepprocess_audio(track, ofs, rfs, mono_type = "mix"):
    if track.shape[-1] > 1:
        # stereo
        if mono_type == "mix":
            track = np.transpose(track, (1,0))
            track = librosa.to_mono(track)
        elif mono_type == "left":
            track = track[:, 0]
        elif mono_type == "right":
            track = track[:, 1]
    else:
        track = track[:, 0]
    # track [n_sample]
    if ofs != rfs:
        track = librosa.resample(track, ofs, rfs)
    return track

# *************************************************
# all below is referred from the wiener filter code

def atan2(y, x):
    r"""Element-wise arctangent function of y/x.
    Returns a new tensor with signed angles in radians.
    It is an alternative implementation of torch.atan2
    Args:
        y (Tensor): First input tensor
        x (Tensor): Second input tensor [shape=y.shape]
    Returns:
        Tensor: [shape=y.shape].
    """
    pi = 2 * torch.asin(torch.tensor(1.0))
    x += ((x == 0) & (y == 0)) * 1.0
    out = torch.atan(y / x)
    out += ((y >= 0) & (x < 0)) * pi
    out -= ((y < 0) & (x < 0)) * pi
    out *= 1 - ((y > 0) & (x == 0)) * 1.0
    out += ((y > 0) & (x == 0)) * (pi / 2)
    out *= 1 - ((y < 0) & (x == 0)) * 1.0
    out += ((y < 0) & (x == 0)) * (-pi / 2)
    return out


# Define basic complex operations on torch.Tensor objects whose last dimension
# consists in the concatenation of the real and imaginary parts.
def _norm(x: torch.Tensor) -> torch.Tensor:
    r"""Computes the norm value of a torch Tensor, assuming that it
    comes as real and imaginary part in its last dimension.
    Args:
        x (Tensor): Input Tensor of shape [shape=(..., 2)]
    Returns:
        Tensor: shape as x excluding the last dimension.
    """
    return torch.abs(x[..., 0]) ** 2 + torch.abs(x[..., 1]) ** 2


def _mul_add(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts.
    The result is added to the `out` tensor"""

    # check `out` and allocate it if needed
    target_shape = torch.Size([max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = out[..., 0] + (real_a * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (real_a * b[..., 1] + a[..., 1] * b[..., 0])
    else:
        out[..., 0] = out[..., 0] + (a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0])
    return out


def _mul(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts
    can work in place in case out is a only"""
    target_shape = torch.Size([max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = real_a * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = real_a * b[..., 1] + a[..., 1] * b[..., 0]
    else:
        out[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return out


def _inv(z: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise multiplicative inverse of a Tensor with complex
    entries described through their real and imaginary parts.
    can work in place in case out is z"""
    ez = _norm(z)
    if out is None or out.shape != z.shape:
        out = torch.zeros_like(z)
    out[..., 0] = z[..., 0] / ez
    out[..., 1] = -z[..., 1] / ez
    return out


def _conj(z, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise complex conjugate of a Tensor with complex entries
    described through their real and imaginary parts.
    can work in place in case out is z"""
    if out is None or out.shape != z.shape:
        out = torch.zeros_like(z)
    out[..., 0] = z[..., 0]
    out[..., 1] = -z[..., 1]
    return out


def _invert(M: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Invert 1x1 or 2x2 matrices
    Will generate errors if the matrices are singular: user must handle this
    through his own regularization schemes.
    Args:
        M (Tensor): [shape=(..., nb_channels, nb_channels, 2)]
            matrices to invert: must be square along dimensions -3 and -2
    Returns:
        invM (Tensor): [shape=M.shape]
            inverses of M
    """
    nb_channels = M.shape[-2]

    if out is None or out.shape != M.shape:
        out = torch.empty_like(M)

    if nb_channels == 1:
        # scalar case
        out = _inv(M, out)
    elif nb_channels == 2:
        # two channels case: analytical expression

        # first compute the determinent
        det = _mul(M[..., 0, 0, :], M[..., 1, 1, :])
        det = det - _mul(M[..., 0, 1, :], M[..., 1, 0, :])
        # invert it
        invDet = _inv(det)

        # then fill out the matrix with the inverse
        out[..., 0, 0, :] = _mul(invDet, M[..., 1, 1, :], out[..., 0, 0, :])
        out[..., 1, 0, :] = _mul(-invDet, M[..., 1, 0, :], out[..., 1, 0, :])
        out[..., 0, 1, :] = _mul(-invDet, M[..., 0, 1, :], out[..., 0, 1, :])
        out[..., 1, 1, :] = _mul(invDet, M[..., 0, 0, :], out[..., 1, 1, :])
    else:
        raise Exception("Only 2 channels are supported for the torch version.")
    return out



def expectation_maximization(
    y: torch.Tensor,
    x: torch.Tensor,
    iterations: int = 2,
    eps: float = 1e-10,
    batch_size: int = 200,
):
    r"""Expectation maximization algorithm, for refining source separation
    estimates.
    Args:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            initial estimates for the sources
        x (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2)]
            complex STFT of the mixture signal
        iterations (int): [scalar]
            number of iterations for the EM algorithm.
        eps (float or None): [scalar]
            The epsilon value to use for regularization and filters.
    Returns:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            estimated sources after iterations
        v (Tensor): [shape=(nb_frames, nb_bins, nb_sources)]
            estimated power spectral densities
        R (Tensor): [shape=(nb_bins, nb_channels, nb_channels, 2, nb_sources)]
            estimated spatial covariance matrices
    """
    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape[:-1]
    nb_sources = y.shape[-1]

    regularization = torch.cat(
        (
            torch.eye(nb_channels, dtype=x.dtype, device=x.device)[..., None],
            torch.zeros((nb_channels, nb_channels, 1), dtype=x.dtype, device=x.device),
        ),
        dim=2,
    )
    regularization = torch.sqrt(torch.as_tensor(eps)) * (
        regularization[None, None, ...].expand((-1, nb_bins, -1, -1, -1))
    )

    # allocate the spatial covariance matrices
    R = [
        torch.zeros((nb_bins, nb_channels, nb_channels, 2), dtype=x.dtype, device=x.device)
        for j in range(nb_sources)
    ]
    weight: torch.Tensor = torch.zeros((nb_bins,), dtype=x.dtype, device=x.device)

    v: torch.Tensor = torch.zeros((nb_frames, nb_bins, nb_sources), dtype=x.dtype, device=x.device)
    for it in range(iterations):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor

        # update the PSD as the average spectrogram over channels
        v = torch.mean(torch.abs(y[..., 0, :]) ** 2 + torch.abs(y[..., 1, :]) ** 2, dim=-2)

        # update spatial covariance matrices (weighted update)
        for j in range(nb_sources):
            R[j] = torch.tensor(0.0, device=x.device)
            weight = torch.tensor(eps, device=x.device)
            pos: int = 0
            batch_size = batch_size if batch_size else nb_frames
            while pos < nb_frames:
                t = torch.arange(pos, min(nb_frames, pos + batch_size))
                pos = int(t[-1]) + 1

                R[j] = R[j] + torch.sum(_covariance(y[t, ..., j]), dim=0)
                weight = weight + torch.sum(v[t, ..., j], dim=0)
            R[j] = R[j] / weight[..., None, None, None]
            weight = torch.zeros_like(weight)

        # cloning y if we track gradient, because we're going to update it
        if y.requires_grad:
            y = y.clone()

        pos = 0
        while pos < nb_frames:
            t = torch.arange(pos, min(nb_frames, pos + batch_size))
            pos = int(t[-1]) + 1

            y[t, ...] = torch.tensor(0.0, device=x.device)

            # compute mix covariance matrix
            Cxx = regularization
            for j in range(nb_sources):
                Cxx = Cxx + (v[t, ..., j, None, None, None] * R[j][None, ...].clone())

            # invert it
            inv_Cxx = _invert(Cxx)

            # separate the sources
            for j in range(nb_sources):

                # create a wiener gain for this source
                gain = torch.zeros_like(inv_Cxx)

                # computes multichannel Wiener gain as v_j R_j inv_Cxx
                indices = torch.cartesian_prod(
                    torch.arange(nb_channels),
                    torch.arange(nb_channels),
                    torch.arange(nb_channels),
                )
                for index in indices:
                    gain[:, :, index[0], index[1], :] = _mul_add(
                        R[j][None, :, index[0], index[2], :].clone(),
                        inv_Cxx[:, :, index[2], index[1], :],
                        gain[:, :, index[0], index[1], :],
                    )
                gain = gain * v[t, ..., None, None, None, j]

                # apply it to the mixture
                for i in range(nb_channels):
                    y[t, ..., j] = _mul_add(gain[..., i, :], x[t, ..., i, None, :], y[t, ..., j])

    return y, v, R

def _covariance(y_j):
    """
    Compute the empirical covariance for a source.
    Args:
        y_j (Tensor): complex stft of the source.
            [shape=(nb_frames, nb_bins, nb_channels, 2)].
    Returns:
        Cj (Tensor): [shape=(nb_frames, nb_bins, nb_channels, nb_channels, 2)]
            just y_j * conj(y_j.T): empirical covariance for each TF bin.
    """
    (nb_frames, nb_bins, nb_channels) = y_j.shape[:-1]
    Cj = torch.zeros(
        (nb_frames, nb_bins, nb_channels, nb_channels, 2),
        dtype=y_j.dtype,
        device=y_j.device,
    )
    indices = torch.cartesian_prod(torch.arange(nb_channels), torch.arange(nb_channels))
    for index in indices:
        Cj[:, :, index[0], index[1], :] = _mul_add(
            y_j[:, :, index[0], :],
            _conj(y_j[:, :, index[1], :]),
            Cj[:, :, index[0], index[1], :],
        )
    return Cj

def wiener(
    targets_spectrograms: torch.Tensor,
    mix_stft: torch.Tensor,
    iterations: int = 1,
    softmask: bool = False,
    residual: bool = False,
    scale_factor: float = 10.0,
    eps: float = 1e-10,
):
    """Wiener-based separation for multichannel audio.
    Returns:
        Tensor: shape=(nb_frames, nb_bins, nb_channels, complex=2, nb_sources)
            STFT of estimated sources
    """
    if softmask:
        # if we use softmask, we compute the ratio mask for all targets and
        # multiply by the mix stft
        y = (
            mix_stft[..., None]
            * (
                targets_spectrograms
                / (eps + torch.sum(targets_spectrograms, dim=-1, keepdim=True).to(mix_stft.dtype))
            )[..., None, :]
        )
    else:
        # otherwise, we just multiply the targets spectrograms with mix phase
        # we tacitly assume that we have magnitude estimates.
        angle = atan2(mix_stft[..., 1], mix_stft[..., 0])[..., None]
        nb_sources = targets_spectrograms.shape[-1]
        y = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=mix_stft.dtype, device=mix_stft.device
        )
        y[..., 0, :] = targets_spectrograms * torch.cos(angle)
        y[..., 1, :] = targets_spectrograms * torch.sin(angle)

    if residual:
        # if required, adding an additional target as the mix minus
        # available targets
        y = torch.cat([y, mix_stft[..., None] - y.sum(dim=-1, keepdim=True)], dim=-1)

    if iterations == 0:
        return y

    # we need to refine the estimates. Scales down the estimates for
    # numerical stability
    max_abs = torch.max(
        torch.as_tensor(1.0, dtype=mix_stft.dtype, device=mix_stft.device),
        torch.sqrt(_norm(mix_stft)).max() / scale_factor,
    )

    mix_stft = mix_stft / max_abs
    y = y / max_abs

    # call expectation maximization
    y = expectation_maximization(y, mix_stft, iterations, eps=eps)[0]

    # scale estimates up again
    y = y * max_abs
    return y

def split_nparray_with_overlap(array, array_size, overlap_size):
    result = []
    element_size = int(len(array) / array_size)
    for i in range(array_size):
        offset = int(i * element_size)
        last_loop = i == array_size
        chunk = array[offset : offset + element_size + (0 if last_loop else overlap_size)]
        chunk = chunk.copy()
        chunk.resize(element_size + overlap_size, refcheck = False)
        result.append(chunk)

    return np.array(result)




