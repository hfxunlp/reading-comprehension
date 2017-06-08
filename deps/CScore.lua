local CScore, parent = torch.class('nn.CScore', 'nn.Module')

function CScore:__init()
	parent.__init(self)
	self.gradP = torch.Tensor()
	self.gradQ = torch.Tensor()
end

local function expandT(tin, nexp, vsize)
	return tin:reshape(1, 1, vsize):expand(nexp, 1, vsize)
end

local function expandO(tin, vsize, plen)
	return tin:reshape(plen, 1, 1):expand(plen, 1, vsize)
end

function CScore:updateOutput(input)

	local pas, question = unpack(input)
	local plen = pas:size(1)
	local qlen = question:size(1)
	local vsize = question:size(3)
	local rsize = torch.LongStorage({plen, qlen})
	if not self.output:isSize(rsize) then
		self.output:resize(rsize)
	end
	for i = 1, qlen do
		self.output:select(2, i):copy(torch.cmul(pas, expandT(question[i], plen, vsize)):sum(3))
	end

	return self.output
end

function CScore:updateGradInput(input, gradOutput)

	local pas, question = unpack(input)
	if not self.gradQ:isSize(question:size()) then
		self.gradQ:resizeAs(question)
	end
	if not self.gradP:isSize(pas:size()) then
		self.gradP:resizeAs(pas):zero()
	end
	local plen = pas:size(1)
	local vsize = pas:size(3)
	for i = 1, question:size(1) do
		local _curG = expandO(gradOutput:select(2, i), vsize, plen)
		self.gradQ[i]:copy(torch.cmul(_curG, pas):sum(1))
		self.gradP:add(torch.cmul(expandT(question[i], plen, vsize), _curG))
	end
	self.gradInput = {self.gradP, self.gradQ}

	return self.gradInput
end

function CScore:clearState()
	self.gradP:set()
	self.gradQ:set()
	return parent.clearState(self)
end