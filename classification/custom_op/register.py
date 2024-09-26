import logging
from functools import reduce

from .conv2d.conv_avg import wrap_conv_layer
from .conv2d.conv_svd import wrap_convSVD
from .conv2d.conv_hosvd import wrap_convHOSVD

from .linear.linear_hosvd import wrap_linearHOSVD
from .linear.linear_svd import wrap_linearSVD


def register_filter(module, cfgs):
    logging.info("Registering Gradient Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["finetuned_layer"]:
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = wrap_conv_layer(target, cfgs['radius'], True)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_SVD_filter(module, cfgs):
    logging.info("Registering SVD compression filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["finetuned_layer"]:
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        if cfgs["type"] == "conv":
            upd_layer = wrap_convSVD(target, cfgs["explained_variance_threshold"], True, cfgs["svd_size"])
        elif cfgs["type"] == "linear":
            upd_layer = wrap_linearSVD(target, cfgs["explained_variance_threshold"], True, cfgs["svd_size"])

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_HOSVD_filter(module, cfgs):
    logging.info("Registering HOSVD compression filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["finetuned_layer"]:
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        if cfgs["type"] == "conv":
            upd_layer = wrap_convHOSVD(target, cfgs["explained_variance_threshold"], True, cfgs["k_hosvd"])
        elif cfgs["type"] == "linear":
            upd_layer = wrap_linearHOSVD(target, cfgs["explained_variance_threshold"], True, cfgs["k_hosvd"])

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)