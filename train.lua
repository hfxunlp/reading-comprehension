print("load settings")
require"aconf"

require "utils.Logger"
require "paths"
paths.mkdir(logd)
local logger = Logger(logd.."/"..runid..".log", nil, nil, "w")

logger:log("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

logger:log("load data")
require "dloader"

local function train(trainset, devset, memlimit, storevery)

	local function _train(trainset, devset, memlimit)

		local function getarg(x, tin)
			local vocab = {}
			local curid = 1
			local curwd = 1
			local fscore = tin:reshape(tin:size(1)):totable()
			local fnd = nil
			for _, sent in ipairs(x) do
				for __, wd in ipairs(sent:reshape(sent:size(1)):totable()) do
					if not vocab[wd] then
						vocab[wd] = curid
						curid = curid + 1
					end
					if fscore[curwd] or 1 == 1 then
						fnd = wd
						break
					else
						curwd = curwd + 1
					end
				end
				if fnd then
					break
				end
			end
			return torch.Tensor({vocab[fnd] or curid - 1})
		end

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
				local gradCriterion=criterionin:backward(pred, y)
				sumErr=sumErr+_inner_err
				pred=nil
				mlpin:backward(x, gradCriterion)

				--mlpin:maxParamNorm(2)

				checkgpu(limit)
				optm(feval, _inner_params, {learningRate = lr})

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

		local function evaDev(mlpin, criterionin,devdata)
			mlpin:evaluate()
			local serr=0
			xlua.progress(0, ndev)
			for i,devu in ipairs(devdata) do
				local passage = devu[1]
				serr=serr+criterionin:forward(mlpin:forward({mkcudaLong(passage), devu[2]:cudaLong()}), getarg(passage, devu[3]):cudaLong())
				xlua.progress(i, ndev)
			end
			mlpin:training()
			return serr/ndev
		end

		local function saveObject(fname,objWrt)

			--[[local function clearModule(module)
				module:apply(function(m)
					if m.clearState and not torch.isTypeOf(m, "nn.gModule") then
						m:clearState()
					end
				end)
				return module
			end]]

			if torch.isTypeOf(objWrt, "nn.Module") then
				objWrt:clearState()
				torch.save(fname, objWrt, 'binary', true)
			else
				torch.save(fname, objWrt, 'binary', false)
			end

		end

		local crithis={}
		local cridev={}

		local erate=0
		local edevrate=0
		local storemini=1
		local storedevmini=1
		local storepoch=1
		local minerrate=starterate
		local mindeverrate=minerrate

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
		local savennmod=nn.Serial(nnmod):mediumSerial()

		logger:log("init train")
		local epochs=1
		local lr=modlr

		mindeverrate=evaDev(nnmod,critmod,devset)
		logger:log("Init model Dev:"..mindeverrate)

		local eaddtrain=ntrain*ieps

		collectgarbage()

		logger:log("start pre train")
		for tmpi=1,warmcycle do
			for tmpj=1,ieps do
				xlua.progress(0, ntrain)
				for i,trainu in ipairs(trainset) do
					local passage = trainu[1]
					gradUpdate(nnmod,{mkcudaLong(passage), trainu[2]:cudaLong()},getarg(passage, trainu[3]):cudaLong(),critmod,lr,optmethod,memlimit)
					xlua.progress(i, ntrain)
				end
			end
			local erate=sumErr/eaddtrain
			if erate<minerrate then
				minerrate=erate
			end
			table.insert(crithis,erate)
			logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
			sumErr=0
			epochs=epochs+1
		end

		if warmcycle>0 then
			logger:log("save neural network trained")
			--savennmod:clearState()
			saveObject(savedir.."nnmod.asc",savennmod)
		end

		epochs=1
		local icycle=1

		local aminerr=1
		local amindeverr=1
		local lrdecayepochs=1

		local cntrun=true

		collectgarbage()

		while cntrun do
			logger:log("start innercycle:"..icycle)
			for innercycle=1,gtraincycle do
				for tmpi=1,ieps do
					xlua.progress(0, ntrain)
					for i,trainu in ipairs(trainset) do
						local passage = trainu[1]
						gradUpdate(nnmod,{mkcudaLong(passage), trainu[2]:cudaLong()},getarg(passage, trainu[3]):cudaLong(),critmod,lr,optmethod,memlimit)
						xlua.progress(i, ntrain)
					end
				end
				local erate=sumErr/eaddtrain
				table.insert(crithis,erate)
				local edevrate=evaDev(nnmod,critmod,devset)
				table.insert(cridev,edevrate)
				logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate..",Dev:"..edevrate)
				--logger:log("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
				local modsavd=false
				if edevrate<mindeverrate then
					mindeverrate=edevrate
					amindeverr=1
					aminerr=1--reset aminerr at the same time
					--savennmod:clearState()
					saveObject(savedir.."devnnmod"..storedevmini..".asc",savennmod)
					storedevmini=storedevmini+1
					if storedevmini>csave then
						storedevmini=1
					end
					modsavd=true
					logger:log("new minimal dev error found, model saved")
				else
					if earlystop and amindeverr>earlystop then
						logger:log("early stop")
						cntrun=false
						break
					end
					amindeverr=amindeverr+1
				end
				if erate<minerrate then
					minerrate=erate
					aminerr=1
					if not modsavd then
						--savennmod:clearState()
						saveObject(savedir.."nnmod"..storemini..".asc",savennmod)
						storemini=storemini+1
						if storemini>csave then
							storemini=1
						end
						logger:log("new minimal error found, model saved")
					end
				else
					if aminerr>=expdecaycycle then
						aminerr=0
						if lrdecayepochs>lrdecaycycle then
							modlr=lr
							lrdecayepochs=1
						end
						lrdecayepochs=lrdecayepochs+1
						lr=modlr/(lrdecayepochs)
					end
					if storevery then
						saveObject(savedir.."dnnmod"..storepoch..".asc",savennmod)
						storepoch=storepoch+1
						if storepoch>csave then
							storepoch=1
						end
					end
					aminerr=aminerr+1
				end
				sumErr=0
				epochs=epochs+1
			end

			logger:log("save neural network trained")
			saveObject(savedir.."nnmod.asc",savennmod)

			logger:log("save criterion history trained")
			local critensor=torch.Tensor(crithis)
			saveObject(savedir.."crit.asc",critensor)
			local critdev=torch.Tensor(cridev)
			saveObject(savedir.."critdev.asc",critdev)

			--[[logger:log("plot and save criterion")
			gnuplot.plot(critensor)
			gnuplot.figprint(savedir.."crit.png")
			gnuplot.figprint(savedir.."crit.eps")
			gnuplot.plotflush()
			gnuplot.plot(critdev)
			gnuplot.figprint(savedir.."critdev.png")
			gnuplot.figprint(savedir.."critdev.eps")
			gnuplot.plotflush()]]

			critensor=nil
			critdev=nil

			logger:log("task finished!Minimal error rate:"..minerrate.."	"..mindeverrate)
			--logger:log("task finished!Minimal error rate:"..minerrate)

			logger:log("wait for test, neural network saved at nnmod*.asc")

			icycle=icycle+1

			logger:log("collect garbage")
			collectgarbage()

		end
	end

	local _, err = pcall(function ()
		_train(trainset, devset, memlimit)
		end
	)
	if err then
		logger:log(err)
	end
end

train(traind, devd, recyclemem or 0.05, storedebug)

logger:shutDown()
