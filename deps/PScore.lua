local PScore, parent = torch.class('nn.PScore', 'nn.SequenceContainer')

function PScore:__init(Calcer)
	parent.__init(self, Calcer)
	self.gradP = torch.Tensor()
	self.gradQ = torch.Tensor()
end

local function expandT(tin, nexp)
	local usize = tin:size()
	local bsize = usize[1]
	local vsize = usize[2]
	return tin:reshape(1, bsize, vsize):expand(nexp, bsize, vsize)
end

function PScore:updateOutput(input)

	local extpas, question = unpack(input)
	local plen = extpas:size(1)
	local qlen = question:size(1)
	local rsize = torch.LongStorage({plen, qlen})
	if not self.output:isSize(rsize) then
		self.output:resize(rsize)
	end
	if not self._qsize then
		self._qsize = question:size(3)
		self._sind = self._qsize + 1
		self._slen = extpas:size(3) - self._qsize
	end
	local _cP = extpas:narrow(3, 1, self._qsize)
	for _ = 1, qlen do
		_cP:copy(expandT(question[_], plen))
		self.output:select(2, _):copy(self:net(_):updateOutput(extpas))
	end

	return self.output
end

function PScore:updateGradInput(input, gradOutput)

	local extpas, question = unpack(input)
	if not self.gradQ:isSize(question:size()) then
		self.gradQ:resizeAs(question):zero()
	end
	if not self.gradP:isSize(extpas) then
		self.gradP:resizeAs(extpas):zero()
	end
	local plen = extpas:size(1)
	local _cP = extpas:narrow(3, 1, self._qsize)
	local _gP = self.gradP:narrow(3, self._sind, self._slen)
	for _ = 1, question:size(1) do
		_cP:copy(expandT(question[_], plen))
		local _curG = self:net(_):updateGradInput(extpas, gradOutput:select(2, _))
		self.gradQ:add(_curG:narrow(3, 1, self._qsize))
		_gP:add(_curG:narrow(3, self._sind, self._slen))
	end
	self.gradInput = {self.gradP, self.gradQ}

	return self.gradInput
end

function PScore:accGradParameters(input, gradOutput, scale)
	local extpas, question = unpack(input)
	local plen = extpas:size(1)
	local _cP = extpas:narrow(3, 1, self._qsize)
	for _ = 1, question:size(1) do
		_cP:copy(expandT(question[_], plen))
		self:net(_):accGradParameters(extpas, gradOutput:select(2, _), scale)
	end
end

function PScore:backward(input, gradOutput, scale)

	local extpas, question = unpack(input)
	if not self.gradQ:isSize(question:size()) then
		self.gradQ:resizeAs(question):zero()
	end
	if not self.gradP:isSize(extpas) then
		self.gradP:resizeAs(extpas):zero()
	end
	local plen = extpas:size(1)
	local _cP = extpas:narrow(3, 1, self._qsize)
	local _gP = self.gradP:narrow(3, self._sind, self._slen)
	for _ = 1, question:size(1) do
		_cP:copy(expandT(question[_], plen))
		local _curG = self:net(_):backward(extpas, gradOutput:select(2, _), scale)
		self.gradQ:add(_curG:narrow(3, 1, self._qsize))
		_gP:add(_curG:narrow(3, self._sind, self._slen))
	end
	self.gradInput = {self.gradP, self.gradQ}

	return self.gradInput
end

function PScore:clearState()
	self.gradP:set()
	self.gradQ:set()
	return parent.clearState(self)
end