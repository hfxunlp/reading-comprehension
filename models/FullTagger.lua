local FullTagger, parent = torch.class('nn.FullTagger', 'nn.SequenceContainer')

-- SentEnc should enable fullout
-- PEnc should enable fullout
function FullTagger:__init(SentEnc, PEnc, Classifier, flatten)
	parent.__init(self, SentEnc)
	self.PEnc = PEnc or SentEnc
	self:add(self.PEnc)
	self.CM = Classifier
	self:add(self.CM)
	self.flatten = flatten
	self.cells = torch.Tensor()
	self.cache = torch.Tensor()
	self.grad_output = torch.Tensor()
	self.gradPEnc = torch.Tensor()
	self.train = true
end

-- input should be table which contains its sentences
function FullTagger:updateOutput(input)
	local function expandT(tin, nexp)
		local usize = tin:size()
		local bsize = usize[1]
		local vsize = usize[2]
		return tin:reshape(1, bsize, vsize):expand(nexp, bsize, vsize)
	end
	local hinput, feat = unpack(input)
	local seql = #hinput
	local _output = self:net(1):updateOutput({feat, hinput[1]})[-1]
	local _osize = _output:size()
	local stdsize = torch.LongStorage({seql, _osize[1], _osize[2]})
	if not self.cells:isSize(stdsize) then
		self.cells:resize(stdsize)
	end
	self.cells[1]:copy(_output)
	for _, sent in ipairs(hinput) do
		if _ > 1 then
			self.cells[_]:copy(self:net(_):updateOutput({feat, sent})[-1])
		end
	end
	local _pEnc_full = self.PEnc:updateOutput({feat, self.cells})
	local _pEnc = _pEnc_full[-1]
	if not self._isize then
		self._isize = hinput[1]:size(3)--输入维度
		self._csize = self.cells:size(3)--句向量维度
		self._psize = _pEnc:size(2)--篇章向量维度
		self._fsize = feat:size(2)--问题向量维度
		self._csind = 1 + self._isize--句词特征起始索引
		self._clsind = self._csind + self._csize--句特征起始索引
		self._plsind = self._clsind + self._csize--篇章句特征起始索引
		self._psind = self._plsind + self._psize--篇章特征起始索引
		self._fsind = self._psind + self._psize--问题特征起始索引
	end
	self._nWords = {}
	self._totalWords = 0
	for _, v in ipairs(hinput) do
		local curwds = v:size(1)
		table.insert(self._nWords, curwds)
		self._totalWords = self._totalWords + curwds
	end
	local stdSize = torch.LongStorage({self._totalWords, self._isize + self._csize * 2 + self._psize * 2 + self._fsize})
	if not self.cache:isSize(stdSize) then
		self.cache:resize(stdSize)
	end
	self.cache:narrow(2, self._psind, self._psize):copy(expandT(_pEnc, self._totalWords))
	self.cache:narrow(2, self._fsind, self._fsize):copy(expandT(feat, self._totalWords))
	local curid = 1
	for _, nc in ipairs(self._nWords) do
		local curT = self.cache:narrow(1, curid, nc)
		curT:narrow(2, 1, self._isize):copy(hinput[_])
		curT:narrow(2, self._csind, self._csize):copy(self:net(_).output)
		curT:narrow(2, self._clsind, self._csize):copy(expandT(self.cells[_], nc))
		curT:narrow(2, self._plsind, self._psize):copy(expandT(_pEnc_full[_], nc))
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
	return self.output
end

function FullTagger:updateGradInput(input, gradOutput)
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
	local hinput, feat = unpack(input)
	local gradCache = self.CM:updateGradInput(self.cache, self.grad_output)
	if not self.gradPEnc:isSize(self.PEnc.output) then
		self.gradPEnc:resizeAs(self.PEnc.output)
	end
	local curid = 1
	local _gP = gradCache:narrow(2, self._plsind, self._psize)
	for _, nc in ipairs(self._nWords) do
		self.gradPEnc[_]:copy(_gP:narrow(1, curid, nc):sum(1))
		curid = curid + nc
	end
	self.gradPEnc:add(gradCache:narrow(2, self._psind, self._psize):sum(1))
	self.gradFeat, self.gradCell = unpack(self.PEnc:updateGradInput({feat, self.cells}, self.gradPEnc))
	curid = 1
	for _, nc in ipairs(self._nWords) do
		self.gradCell[_]:add(gradCache:narrow(1, curid, nc):narrow(2, self._clsind, self._csize):sum(1))
		curid = curid + 1
	end
	local _gradInput = {}
	curid = 1
	for _, v in ipairs(hinput) do
		local nc = self._nWords[_]
		local _curGradO = gradCache:narrow(1, curid, nc):narrow(2, self._csind, self._csize)
		_curGradO[-1]:add(self.gradCell[_])
		local _curGradF, _curGrad = unpack(self:net(_):updateGradInput({feat, v}, _curGradO))
		_curGrad:add(gradCache:narrow(1, curid, nc):narrow(2, 1, self._isize))
		table.insert(_gradInput, _curGrad)
		self.gradFeat:add(_curGradF)
		curid = curid + nc
	end
	self.gradFeat:add(gradCache:narrow(2, self._fsind, self._fsize):sum(1):squeeze(1))
	self.gradInput = {_gradInput, self.gradFeat}
	return self.gradInput
end

function FullTagger:accGradParameters(input, gradOutput, scale)
	if not (self.grad_output and self.gradPEnc) then
		self:updateGradInput(input, gradOutput)
	end
	self.CM:accGradParameters(self.cache, self.grad_output, scale)
	self.PEnc:accGradParameters(self.cells, self.gradPEnc, scale)
	local hinput, feat = unpack(input)
	for _, v in ipairs(hinput) do
		self:net(_):accGradParameters({feat, v}, self.gradCell[_], scale)
	end
end

function FullTagger:backward(input, gradOutput, scale)
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
	local hinput, feat = unpack(input)
	local gradCache = self.CM:backward(self.cache, self.grad_output, scale)
	if not self.gradPEnc:isSize(self.PEnc.output) then
		self.gradPEnc:resizeAs(self.PEnc.output)
	end
	local curid = 1
	local _gP = gradCache:narrow(2, self._plsind, self._psize)
	for _, nc in ipairs(self._nWords) do
		self.gradPEnc[_]:copy(_gP:narrow(1, curid, nc):sum(1))
		curid = curid + nc
	end
	self.gradPEnc:add(gradCache:narrow(2, self._psind, self._psize):sum(1))
	self.gradFeat, self.gradCell = unpack(self.PEnc:backward({feat, self.cells}, self.gradPEnc, scale))
	curid = 1
	for _, nc in ipairs(self._nWords) do
		self.gradCell[_]:add(gradCache:narrow(1, curid, nc):narrow(2, self._clsind, self._csize):sum(1))
		curid = curid + 1
	end
	local _gradInput = {}
	curid = 1
	for _, v in ipairs(hinput) do
		local nc = self._nWords[_]
		local _curGradO = gradCache:narrow(1, curid, nc):narrow(2, self._csind, self._csize)
		_curGradO[-1]:add(self.gradCell[_])
		local _curGradF, _curGrad = unpack(self:net(_):backward({feat, v}, _curGradO, scale))
		_curGrad:add(gradCache:narrow(1, curid, nc):narrow(2, 1, self._isize))
		table.insert(_gradInput, _curGrad)
		self.gradFeat:add(_curGradF)
		curid = curid + nc
	end
	self.gradFeat:add(gradCache:narrow(2, self._fsind, self._fsize):sum(1):squeeze(1))
	self.gradInput = {_gradInput, self.gradFeat}
	return self.gradInput
end

function FullTagger:evaluate()
	parent.evaluate(self)
	self.train = true
end

function FullTagger:clearState()
	self.cells:set()
	self.cache:set()
	self._output:set()
	self.grad_output:set()
	self.gradPEnc:set()
	self.gradFeat:set()
	self.gradCell:set()
	self._nWords = {}
	self._totalWords = 0
	return parent.clearState(self)
end