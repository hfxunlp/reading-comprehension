local HiTagger, parent = torch.class('nn.HiTagger', 'nn.SequenceContainer')

-- SentEnc and PEnc should not enable fullout
function HiTagger:__init(SentEnc, PEnc, Classifier, flatten)
	parent.__init(self, SentEnc)
	self.PEnc = PEnc or sentEnc
	self:add(self.PEnc)
	self.CM = Classifier
	self:add(self.CM)
	self.flatten = flatten
	self.cells = torch.Tensor()
	self.cache = torch.Tensor()
	self.grad_output = torch.Tensor()
end

-- input should be table which contains its sentences
function HiTagger:updateOutput(input)
	local function expandT(tin, nexp)
		local usize = tin:size()
		local bsize = usize[1]
		local vsize = usize[2]
		return tin:reshape(1, bsize, vsize):expand(nexp, bsize, vsize)
	end
	local seql = #input
	local _output = self:net(1):updateOutput(input[1])
	local _osize = _output:size()
	local stdsize = torch.LongStorage({seql, _osize[1], _osize[2]})
	if not self.cells:isSize(stdsize) then
		self.cells:resize(stdsize)
	end
	self.cells[1]:copy(_output)
	for _, sent in ipairs(input) do
		if _ > 1 then
			self.cells[_]:copy(self:net(_):updateOutput(sent))
		end
	end
	local _pEnc = self.PEnc:updateOutput(self.cells)
	if not self._isize then
		self._isize = input[1]:size(3)
		self._csize = self.cells:size(3)
		self._psize = _pEnc:size(2)
		self._csind = 1 + self._isize
		self._psind = self._csind + self._csize
	end
	self._nWords = {}
	self._totalWords = 0
	for _, v in ipairs(input) do
		local curwds = v:size(1)
		table.insert(self._nWords, curwds)
		self._totalWords = self._totalWords + curwds
	end
	local stdSize = torch.LongStorage({self._totalWords, self._isize + self._csize + self._psize})
	if not self.cache:isSize(stdSize) then
		self.cache:resize(stdSize)
	end
	self.cache:narrow(2, self._psind, self._psize):copy(expandT(_pEnc, self._totalWords))
	local curid = 1
	for _, nc in ipairs(self._nWords) do
		local curT = self.cache:narrow(1, curid, nc)
		curT:narrow(2, 1, self._isize):copy(input[_])
		curT:narrow(2, self._csind, self._csize):copy(expandT(self.cells[_], nc))
		curid = curid + nc
	end
	self._output = self.CM:updateOutput(self.cache)
	if self.flatten then
		self.output = self._output
	else
		self.output = {}
		curid = 1
		for _, nc in ipairs(self._nWords) do
			table.insert(self.output, self._output:narrow(1, curid, nc))
			curid = curid + nc
		end
	end
	self.doupgi = nil
	return self.output
end

function HiTagger:updateGradInput(input, gradOutput)
	if not self.doupgi then
		if self.flatten then
			self.grad_output = gradOutput
		else
			local usize = gradOutput[1]:size()
			local stdsize = torch.LongStorage({self._totalWords, usize[1], usize[2]})
			if not self.grad_output:isSize(stdsize) then
				self.grad_output:resize(stdsize)
			end
			local curid = 1
			for _, nc in ipairs(self._nWords) do
				self.grad_output:narrow(1, curid, nc):copy(gradOutput[_])
				curid = curid + nc
			end
		end
		local gradCache = self.CM:updateGradInput(self.cache, self.grad_output)
		self.gradPEnc = gradCache:narrow(2, self._psind, self._psize):sum(1):squeeze(1)
		self.gradCell = self.PEnc:updateGradInput(self.cells, self.gradPEnc)
		local curid = 1
		local _gP = gradCache:narrow(2, self._csind, self._csize)
		for _, nc in ipairs(self._nWords) do
			self.gradCell[_]:add(_gP:narrow(1, curid, nc):sum(1))
			curid = curid + 1
		end
		self.gradInput = {}
		curid = 1
		_gP = gradCache:narrow(2, 1, self._isize)
		for _, v in ipairs(input) do
			local nc = self._nWords[_]
			local _curGrad = self:net(_):updateGradInput(v, self.gradCell[_])
			_curGrad:add(_gP:narrow(1, curid, nc))
			table.insert(self.gradInput, _curGrad)
			curid = curid + nc
		end
		self.doupgi = true
	end
	return self.gradInput
end

function HiTagger:accGradParameters(input, gradOutput, scale)
	if not self.doupgi then
		self:updateGradInput(input, gradOutput)
	end
	self.CM:accGradParameters(self.cache, self.grad_output, scale)
	self.PEnc:accGradParameters(self.cells, self.gradPEnc, scale)
	for _, v in ipairs(input) do
		self:net(_):accGradParameters(v, self.gradCell[_], scale)
	end
end

function HiTagger:backward(input, gradOutput, scale)
	if self.flatten then
		self.grad_output = gradOutput
	else
		local usize = gradOutput[1]:size()
		local stdsize = torch.LongStorage({self._totalWords, usize[1], usize[2]})
		if not self.grad_output:isSize(stdsize) then
			self.grad_output:resize(stdsize)
		end
		local curid = 1
		for _, nc in ipairs(self._nWords) do
			self.grad_output:narrow(1, curid, nc):copy(gradOutput[_])
			curid = curid + nc
		end
	end
	local gradCache = self.CM:backward(self.cache, self.grad_output, scale)
	self.gradPEnc = gradCache:narrow(2, self._psind, self._psize):sum(1):squeeze(1)
	self.gradCell = self.PEnc:backward(self.cells, self.gradPEnc, scale)
	local curid = 1
	local _gP = gradCache:narrow(2, self._csind, self._csize)
	for _, nc in ipairs(self._nWords) do
		self.gradCell[_]:add(_gP:narrow(1, curid, nc):sum(1))
		curid = curid + 1
	end
	self.gradInput = {}
	curid = 1
	_gP = gradCache:narrow(2, 1, self._isize)
	for _, v in ipairs(input) do
		local nc = self._nWords[_]
		local _curGrad = self:net(_):backward(v, self.gradCell[_], scale)
		_curGrad:add(_gP:narrow(1, curid, nc))
		table.insert(self.gradInput, _curGrad)
		curid = curid + nc
	end
	return self.gradInput
end

function HiTagger:clearState()
	self.cells:set()
	self.cache:set()
	self._output:set()
	self.grad_output:set()
	self.gradPEnc:set()
	self._nWords = {}
	self._totalWords = 0
	self.doupgi = nil
	return parent.clearState(self)
end