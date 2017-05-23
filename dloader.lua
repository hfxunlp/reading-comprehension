tds = require "tds"

local _fulldata = torch.load("datasrc/data.asc", 'binary', false)

traind, devd, wvec = _fulldata[1], _fulldata[2], _fulldata[3]

wvec=wvec:float()

ntrain=#traind
ndev=#devd
nword=wvec:size(1)
