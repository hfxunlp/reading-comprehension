#!/bin/bash

export srcf=$1
export rsf=$2

export tmpd="temp"
export toold="tools"

if [ -d "$tmpd" ]; then  
  rm -fr $tmpd  
fi
mkdir -p $tmpd

python $toold/filq4valid.py $srcf $tmpd/cloze.valid.fq
python $toold/getzag.py $tmpd/cloze.valid.fq map.txt $tmpd/cloze.valid.targ
python $toold/map.py $tmpd/cloze.valid.fq $tmpd/cloze.valid.map map.txt
python $toold/jdata.py $tmpd/cloze.valid.map $tmpd/cloze.valid.targ $tmpd/valid.data

th predict.lua $tmpd/valid.data $tmpd/aoanscore.txt

python tools/nsel.py map.txt $tmpd/cloze.valid.fq $tmpd/aoanscore.txt $rsf

rm -fr $tmpd
