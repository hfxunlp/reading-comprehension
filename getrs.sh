#!/bin/bash

wkdir=test
srcsf=aoanscore.txt
rsf=cloze.valid.predict

python tools/colimselans.py datasrc/duse/map.txt $wkdir/cloze.valid.doc_query $wkdir/$srcsf $wkdir/$rsf
