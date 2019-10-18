import sys
import os
import chainer

import opts
import models
import dataset

def main():
    opt = opts.parse()
    for split in opt.splits:
        print('+-- Split {} --+'.format(split))
        train(opt, split)

def train(opt, split):
    model = getattr(models, opt.netType)(opt.nClasses)
    model.to_gpu()
    optimizer