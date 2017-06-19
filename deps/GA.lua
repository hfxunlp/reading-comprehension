local GA, parent = torch.class('nn.GA', 'nn.Container')

function GA:__init(gaunit)
	parent.__init(self)
	self.network = gaunit
	self:add(self.network)
end

function GA:updateOutput(input)
	local doc, q = unpack(input)
	local _s = doc:size()
	if not self.output:isSize(_s) then
		self.output:resize(_s)
	end
	local slen = _s[1]
	local bsize = _s[2]
	local vsize = _s[3]
	self.rInput = {q:repeatTensor(1, slen, 1), doc:reshape(slen * bsize, vsize)}
	self.output = self.network:updateOutput(self.rInput):reshape(slen, bsize, vsize)
	return self.output
end

function GA:updateGradInput(input, gradOutput)
	local doc, q = unpack(input)
	local _s = doc:size()
	local slen = _s[1]
	local bsize = _s[2]
	local vsize = _s[3]
	local rGrad = gradOutput:reshape(slen * bsize, vsize)
	local gradQ, gradD = unpack(self.network:updateGradInput(self.rInput, rGrad))
	local qlen = gradQ:size(1)
	self.gradInput = {gradD:reshape(slen, bsize, vsize), gradQ:reshape(qlen, slen, bsize, vsize):sum(2):squeeze(2)}
	return self.gradInput
end

function GA:accGradParameters(input, gradOutput, scale)
	local _s = input[1]:size()
	self.network:accGradParameters(self.rInput, gradOutput:reshape(_s[1] * _s[2], _s[3]), scale)
end

function GA:backward(input, gradOutput, scale)
	local doc, q = unpack(input)
	local _s = doc:size()
	local slen = _s[1]
	local bsize = _s[2]
	local vsize = _s[3]
	local rGrad = gradOutput:reshape(slen * bsize, vsize)
	local gradQ, gradD = unpack(self.network:backward(self.rInput, rGrad, scale))
	local qlen = gradQ:size(1)
	self.gradInput = {gradD:reshape(slen, bsize, vsize), gradQ:reshape(qlen, slen, bsize, vsize):sum(2):squeeze(2)}
	return self.gradInput
end

function GA:clearState()
	self.rInput = nil
	return parent.clearState(self)
end