#!/usr/bin/env bash

if [ ! -f "kaldi-generic-en-tdnn_fl-r20190609.tar.xz" ]
then
    wget --no-check-certificate https://goofy.zamia.org/zamia-speech/asr-models/kaldi-generic-en-tdnn_fl-r20190609.tar.xz
fi
tar -xJof kaldi-generic-en-tdnn_fl-r20190609.tar.xz
rm -f kaldi-generic-en-tdnn_fl-r20190609.tar.xz

mkdir MODEL_LICENSE
mv kaldi-generic-en-tdnn_fl-r20190609/README.md MODEL_LICENSE/README-ZAMIA.md
mv kaldi-generic-en-tdnn_fl-r20190609/AUTHORS MODEL_LICENSE
mv kaldi-generic-en-tdnn_fl-r20190609/LICENSE MODEL_LICENSE
mv kaldi-generic-en-tdnn_fl-r20190609/* .	
rm -rf kaldi-generic-en-tdnn_fl-r20190609

mkdir -p exp/nnet3_chain
mv model exp/nnet3_chain/tdnn_f
mv extractor exp/nnet3_chain/.
mv ivectors_test_hires exp/nnet3_chain/.
ln -sr exp/nnet3_chain/ivectors_test_hires/conf/ivector_extractor.conf conf/.
ln -sr ../aspire/data/test data/.
