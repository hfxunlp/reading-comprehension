local modf = "modrs/170713_ne_2gaoabase_088sgd_mod2_01_v50_h50/devnnmod3.asc" -- model file
local rsm = "submit/nnmod.asc"

require "nn"
require "cutorch"
require "cunn"
require "cudnn"

require "nn.Decorator"
require "dpnn"

require "nngraph"

require "deps.vecLookup"
require "deps.JoinFSeq"
require "deps.JoinBFSeq"
require "deps.SelData"
require "deps.BSelData"
require "deps.TableContainer"
require "deps.SequenceContainer"
require "deps.Coll"
require "deps.PScore"
require "deps.AoA"
require "models.NICPFullTagger"
require "deps.CScore"
require "deps.GA"
require "deps.ADim"
require "deps.fColl"

local tmod_full = torch.load(modf)
tmod_full:lightSerial()
tmod_full:clearState()
torch.save(rsm, tmod_full, 'binary', true)
tmod_full = torch.load(rsm)
tmod = tmod_full.modules[1]
tmod = tmod:clearState()
torch.save(rsm, tmod, 'binary', true)
