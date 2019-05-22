# >>> kaldi initialize >>>
#    if [ -f "/home/ironbas3/pykaldi/examples/setups/zamia/path.sh" ]; then
 #       . "/home/ironbas3/pykaldi/examples/setups/zamia/path.sh"
  #  else
   #     echo "kaldi not initialized"
   # fi
# <<< kaldi initialize

export KALDI_ROOT=/home/$USER/pykaldi/tools/kaldi
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
