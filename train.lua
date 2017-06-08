print("load settings")
require"aconf"

require "utils.Logger"
require "paths"
paths.mkdir(logd)

local logmod
if cntrain then
	logmod = "a"
else
	logmod = "w"
end
local logger = Logger(logd.."/"..runid..".log", nil, nil, logmod)

logger:log("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

logger:log("load learning rate manager")
require "utils.lrSheduler"
local lrKeeper = lrSheduler(modlr, nil, expdecaycycle, lrdecaycycle, earlystop, csave, csave, true, false, "modrs/"..runid.."/nnmod", "modrs/"..runid.."/devnnmod", "modrs/"..runid.."/dnnmod", ".asc", nil, logger, true, "modrs/"..runid.."/crit.asc", "modrs/"..runid.."/critdev.asc")

logger:log("load data")
local traind, devd = unpack(require "dloader")

local function train(trainset, devset, memlimit, lrKeeper, parupdate)

	local function _train(trainset, devset, memlimit, lrKeeper, parupdate)

		logger:log("pre load package")
		require "nn"
		require "nn.Decorator"
		require "dpnn"
		--require "dp"

		local sumErr=0
		local _inner_err, _inner_gradParams

		local function gradUpdate(mlpin, x, y, criterionin, lr, optm, limit)

			local function _gradUpdate(mlpin, x, y, criterionin, lr, optm, limit)

				local function checkgpu(limit)
					local fmem, totalmem = cutorch.getMemoryUsage()
					local amem = fmem/totalmem
					if amem < limit then
						collectgarbage()
					end
					return amem
				end

				local function feval()
					return _inner_err, _inner_gradParams
				end

				_inner_gradParams:zero()

				local pred=mlpin:forward(x)
				_inner_err=criterionin:forward(pred, y)
				if _inner_err~=0 then
					local gradCriterion=criterionin:backward(pred, y)
					sumErr=sumErr+_inner_err
					pred=nil
					mlpin:backward(x, gradCriterion)

					--mlpin:maxParamNorm(2)

					if limit then
						checkgpu(limit)
					end
					optm(feval, _inner_params, {learningRate = lr})
				end

			end

			local _, err = pcall(function ()
				_gradUpdate(mlpin, x, y, criterionin, lr, optm, limit)
				end
			)
			if err then
				logger:log(err)
			end

		end

		local function mkcudaLong(din)
			local rs = {}
			for _, v in ipairs(din) do
				table.insert(rs, v:cudaLong())
			end
			return rs
		end

		local function evaDev(mlpin, criterionin, devdata)
			mlpin:evaluate()
			local serr=0
			xlua.progress(0, ndev)
			for i, id, td in devdata:subiter() do
				serr=serr+criterionin:forward(mlpin:forward(id), td)
				xlua.progress(i, ndev)
			end
			mlpin:training()
			return serr/ndev
		end

		local erate, edevrate

		logger:log("prepare environment")
		local savedir="modrs/"..runid.."/"
		paths.mkdir(savedir)

		logger:log("load optim")

		require "getoptim"
		local optmethod=getoptim()

		logger:log("design neural networks and criterion")

		require "designn"
		local nnmod=getnn()

		logger:log(nnmod)
		nnmod:training()

		local critmod=getcrit()

		nnmod:cuda()
		critmod:cuda()

		wvec=nil

		_inner_params, _inner_gradParams=nnmod:getParameters()

		logger:log("register save model to lrScheduler")
		lrKeeper.module=nn.Serial(nnmod):mediumSerial()

		logger:log("init train")
		local epochs=1
		local lr=lrKeeper.lr

		edevrate=evaDev(nnmod,critmod,devset)
		lrKeeper:feed(nil, edevrate, true)
		logger:log("Init model Dev:"..edevrate)

		local eaddtrain
		if parupdate then
			eaddtrain=ntrain%parupdate
			if eaddtrain==0 then
				eaddtrain=parupdate
			end
		else
			eaddtrain=ntrain*ieps
		end

		collectgarbage()

		logger:log("start pre train")
		for tmpi=1,warmcycle do
			for tmpj=1,ieps do
				xlua.progress(0, ntrain)
				for i, id, td in trainset:subiter() do
					gradUpdate(nnmod, id, td, critmod, lr, optmethod, memlimit)
					xlua.progress(i, ntrain)
					if parupdate and (i%parupdate==0) then
						erate=sumErr/parupdate
						lr=lrKeeper:feed(erate, nil, true)
						sumErr=0
					end
				end
				if parupdate then
					erate=sumErr/eaddtrain
					lr=lrKeeper:feed(erate, nil, true)
					sumErr=0
				end
			end
			if not parupdate then
				erate=sumErr/eaddtrain
				lrKeeper:feed(erate, nil, true)
				logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
				sumErr=0
			end
			epochs=epochs+1
		end

		if warmcycle>0 then
			logger:log("save neural network trained")
			lrKeeper:saveModel(savedir.."nnmod.asc")
		end

		epochs=1
		local icycle=1

		local cntrun=1

		collectgarbage()

		while cntrun do
			logger:log("start innercycle:"..icycle)
			for innercycle=1,gtraincycle do
				for tmpi=1,ieps do
					xlua.progress(0, ntrain)
					for i, id, td in trainset:subiter() do
						gradUpdate(nnmod, id, td, critmod, lr, optmethod, memlimit)
						xlua.progress(i, ntrain)
						if parupdate and (i%parupdate==0) then
							erate=sumErr/parupdate
							lr=lrKeeper:feed(erate, nil, nil, true)
							sumErr=0
						end
					end
					if parupdate and (tmpi<ieps) then
						erate=sumErr/eaddtrain
						lr=lrKeeper:feed(erate, nil, nil, true)
						sumErr=0
					end
				end
				erate=sumErr/eaddtrain
				edevrate=evaDev(nnmod,critmod,devset)
				logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate..",Dev:"..edevrate)
				lr, cntrun = lrKeeper:feed(erate, edevrate)
				if not cntrun then
					break
				end
				sumErr=0
				epochs=epochs+1
			end

			logger:log("save neural network trained")
			lrKeeper:saveModel(savedir.."nnmod.asc")

			logger:log("save criterion history trained")
			lrKeeper:saveCrit()

			logger:log("task finished!Minimal error rate:"..lrKeeper.minerrate.."	"..lrKeeper.mindeverrate)

			logger:log("wait for test, neural network saved at nnmod*.asc")

			icycle=icycle+1

			logger:log("collect garbage")
			collectgarbage()

		end
	end

	local _, err = pcall(function ()
		_train(trainset, devset, memlimit, lrKeeper, parupdate)
		end
	)
	if err then
		logger:log(err)
	end
end

train(traind, devd, recyclemem, lrKeeper, partupdate)

logger:shutDown()
