local TableContainer, parent = torch.class('nn.TableContainer', 'nn.Container')

function TableContainer:__init(module, shareModule)
	parent.__init(self)
	self.network = module
	self.network:training()
	self:add(module)
	self.nets = {}
	self.shareModule = shareModule
end

local function sharedClone(module, shareParams, shareGradParams, SequencePointer)
	shareParams = (shareParams == nil) and true or shareParams
	shareGradParams = (shareGradParams == nil) and true or shareGradParams

	local pointers = {} -- to params/gradParams (dont clone params/gradParams)
	local scdone = {}
	local _parameters = {'weight', 'bias'}
	local _gradParameters = {'gradWeight', 'gradBias'}

	-- 1. remove all params/gradParams 
	local function recursiveRemove(obj) -- remove modules
		local moduleTree
		local isTable = type(obj) == 'table' 
		if torch.isTypeOf(obj, 'nn.Module') then
			if scdone[torch.pointer(obj)] then
				moduleTree = scdone[torch.pointer(obj)]
			else
				-- remove the params, gradParams. Save for later.
				local params = {}
				
				if shareParams then
					for i,paramName in ipairs(_parameters) do
						local param = obj[paramName]
						if param then
							params[paramName] = param
							obj[paramName] = nil
							if torch.isTensor(param) and param.storage and param:storage() then
								pointers[torch.pointer(param:storage():data())] = true
							end
						end
					end
				end

				if shareGradParams then
					for i,paramName in ipairs(_gradParameters) do
						local gradParam = obj[paramName]
						if gradParam then
							params[paramName] = gradParam
							obj[paramName] = nil
							if torch.isTensor(gradParam) and gradParam.storage and gradParam:storage() then
								pointers[torch.pointer(gradParam:storage():data())] = true
							end
						end
					end
				end

				-- find all obj.attribute tensors that share storage with the shared params
				for paramName, param in pairs(obj) do
					if torch.isTensor(param) and param:storage() then
						if pointers[torch.pointer(param:storage():data())] then
							params[paramName] = param
							obj[paramName] = nil
						end
					end
				end

				moduleTree = params

				scdone[torch.pointer(obj)] = moduleTree

				for k,v in pairs(obj) do
					moduleTree[k], obj[k] = recursiveRemove(v)
				end

			end
		elseif isTable then
			if scdone[torch.pointer(obj)] then
				moduleTree = scdone[torch.pointer(obj)]
			else
				moduleTree = {}
				for k,v in pairs(obj) do
					moduleTree[k], obj[k] = recursiveRemove(v)
				end 
				scdone[torch.pointer(obj)] = moduleTree
			end

		end

		return moduleTree, obj
	end

	local moduleTree, original = recursiveRemove(module)

	-- 2. clone everything but parameters, gradients and modules (removed above)
	
	local clone = module:clone()
 
	-- 3. add back to module/clone everything that was removed in step 1

	local function recursiveSet(clone, original, moduleTree)
		if scdone[torch.pointer(original)] then
			for k,param in pairs(moduleTree) do
				if torch.isTypeOf(param,'nn.Module') then
					clone[k] = param
					original[k] = param
				elseif torch.isTensor(param) then
					if param.storage then
						clone[k] = param.new():set(param)
						original[k] = param
					else -- for torch.MultiCudaTensor
						clone[k] = param
						original[k] = param
					end
				elseif type(param) == 'table' then
					recursiveSet(clone[k], original[k], param)
				end
			end 
			scdone[torch.pointer(original)] = nil
		end

	end

	recursiveSet(clone, module, moduleTree)

	if SequencePointer then
		local sequencerContainers = {}

		module:apply(function(m)
			if (torch.isTypeOf(m, 'nn.TableContainer') or torch.isTypeOf(m, 'nn.SequenceContainer')) and m.syncstep then
				table.insert(sequencerContainers, m)
			end
		end)

		if #sequencerModules > 0 then
			local curid = 1
			clone:apply(function(m)
				if (torch.isTypeOf(m, 'nn.TableContainer') or torch.isTypeOf(m, 'nn.SequenceContainer')) and m.syncstep then
					m = sequencerContainers[curid]
					curid = curid + 1
				end
			end)
		end
	end

	return clone
end

function TableContainer:net(t)
	if self.train and not self.shareModule then
		if not self.nets[t] then
			self.nets[t] = sharedClone(self.network, self.shareParams, self.shareGradParams, self.synchild)
		end
		return self.nets[t]
	else
		if not self.nets[1] then
			self.nets[1] = sharedClone(self.network, self.shareParams, self.shareGradParams, self.synchild)
		end
		return self.nets[1]
	end
end

function TableContainer:updateOutput(input)
	if type(input) == "table" then
		self.output = {}
	else
		self.output = tds.Vec()
	end
	if self.shareModule or not self.train then
		for _, v in ipairs(input) do
			self.output[_] = self:net(_):updateOutput(v):clone()
		end
	else
		for _, v in ipairs(input) do
			self.output[_] = self:net(_):updateOutput(v)
		end
	end
	return self.output
end

function TableContainer:updateGradInput(input, gradOutput)
	if type(input) == "table" then
		self.gradInput = {}
	else
		self.gradInput = tds.Vec()
	end
	if self.shareModule then
		for _, v in ipairs(gradOutput) do
			self.gradInput[_] = self:net(_):updateGradInput(input[_], v):clone()
		end
	else
		for _, v in ipairs(gradOutput) do
			self.gradInput[_] = self:net(_):updateGradInput(input[_], v)
		end
	end
	return self.gradInput
end

function TableContainer:accGradParameters(input, gradOutput, scale)
	for _, v in ipairs(gradOutput) do
		self:net(_):accGradParameters(input[_], v, scale)
	end
end

function TableContainer:backward(input, gradOutput, scale)
	if type(input) == "table" then
		self.gradInput = {}
	else
		self.gradInput = tds.Vec()
	end
	if self.shareModule then
		for _, v in ipairs(gradOutput) do
			self.gradInput[_] = self:net(_):backward(input[_], v, scale):clone()
		end
	else
		for _, v in ipairs(gradOutput) do
			self.gradInput[_] = self:net(_):backward(input[_], v, scale)
		end
	end
	return self.gradInput
end

function TableContainer:training()
	--[[for _, net in ipairs(self.nets) do
		net:training()
	end
	parent.training(self)]]
	self:net(1):training()
end

function TableContainer:evaluate()
	self:net(1):evaluate()
	--parent.evaluate(self)
end

function TableContainer:clearState()
	self.nets = {}
	return parent.clearState(self)
end
