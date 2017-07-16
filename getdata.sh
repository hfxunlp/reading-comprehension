#!/bin/bash

python tools\filna.py CMRC2017_train\train.doc_query CMRC2017_train\train.answer CMRC2017_train\train.fild
python tools\filna.py CMRC2017_cloze_valid\cloze.valid.doc_query CMRC2017_cloze_valid\cloze.valid.answer CMRC2017_cloze_valid\cloze.valid.fild
python tools\vocab.py CMRC2017_train\train.fild CMRC2017_train\map.txt
python tools\filsortd.py CMRC2017_train\train.fild CMRC2017_train\train.sort
python tools\filq4valid.py CMRC2017_cloze_valid\cloze.valid.fild CMRC2017_cloze_valid\cloze.valid.fq
python tools\getag.py CMRC2017_cloze_valid\cloze.valid.fq CMRC2017_cloze_valid\cloze.valid.answer CMRC2017_train\map.txt CMRC2017_cloze_valid\cloze.valid.targ
python tools\getag.py CMRC2017_train\train.sort CMRC2017_train\train.answer CMRC2017_train\map.txt CMRC2017_train\train.targ
python tools\map.py CMRC2017_cloze_valid\cloze.valid.fq CMRC2017_cloze_valid\cloze.valid.map CMRC2017_train\map.txt
python tools\map.py CMRC2017_train\train.sort CMRC2017_train\train.map CMRC2017_train\map.txt
python tools\jdata.py CMRC2017_cloze_valid\cloze.valid.map CMRC2017_cloze_valid\cloze.valid.targ CMRC2017_cloze_valid\valid.data
python tools\jdata.py CMRC2017_train\train.map CMRC2017_train\train.targ CMRC2017_train\train.data
