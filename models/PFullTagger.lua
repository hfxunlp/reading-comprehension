local PFullTagger, parent = torch.class('nn.PFullTagger', 'nn.SequenceContainer')

-- SentEnc should enable fullout
-- PEnc should enable fullout
function PFullTagger:__init(SentEnc, PEnc, Classifier, flatten, ploc)
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
	self.gradSEnc = torch.Tensor()
	self.train = true
	self.ploc = ploc or 0
end

-- input should be table which contains its sentences
function PFullTagger:updateOutput(input)
	local function expandT(tin, nexp)
		local usize = tin:size()
		local bsize = usize[1]
		local vsize = usize[2]
		return tin:reshape(1, bsize, vsize):expand(nexp, bsize, vsize)
	end
	local hinput, feat_full = unpack(input)
	local feat = feat_full[-1]
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
		self._sind = self.ploc + 1
		self._csind = self._sind + self._isize--句词特征起始索引
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
	local stdSize = torch.LongStorage({self._totalWords, self.ploc + self._isize + self._csize * 2 + self._psize * 2 + self._fsize})
	if not self.cache:isSize(stdSize) then
		self.cache:resize(stdSize)
	end
	self.cache:narrow(2, self._psind, self._psize):copy(expandT(_pEnc, self._totalWords))
	self.cache:narrow(2, self._fsind, self._fsize):copy(expandT(feat, self._totalWords))
	local curid = 1
	for _, nc in ipairs(self._nWords) do
		local curT = self.cache:narrow(1, curid, nc)
		curT:narrow(2, self._sind, self._isize):copy(hinput[_])
		curT:narrow(2, self._csind, self._csize):copy(self:net(_).output)
		curT:narrow(2, self._clsind, self._csize):copy(expandT(self.cells[_], nc))
		curT:narrow(2, self._plsind, self._psize):copy(expandT(_pEnc_full[_], nc))
		curid = curid + nc
	end
	self._output = self.CM:updateOutput({self.cache, feat_full})
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

function PFullTagger:updateGradInput(input, gradOutput)
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
		local hinput, feat_full = unpack(input)
		local feat = feat_full[-1]
		local gradCache
		gradCache, self.gradFeat = unpack(self.CM:updateGradInput({self.cache, feat_full}, self.grad_output))
		local _gPFeat = self.gradFeat[-1]
		local _gPEncSize = self.PEnc.output:size()
		if not self.gradPEnc:isSize(_gPEncSize) then
			self.gradPEnc:resize(_gPEncSize)
		end
		local curid = 1
		local _gP = gradCache:narrow(2, self._plsind, self._psize)
		for _, nc in ipairs(self._nWords) do
			self.gradPEnc[_]:copy(_gP:narrow(1, curid, nc):sum(1))
			curid = curid + nc
		end
		self.gradPEnc[-1]:add(gradCache:narrow(2, self._psind, self._psize):sum(1))
		local _gradFeat
		_gradFeat, self.gradCell = unpack(self.PEnc:updateGradInput({feat, self.cells}, self.gradPEnc))
		_gPFeat:add(_gradFeat)
		curid = 1
		_gP = gradCache:narrow(2, self._clsind, self._csize)
		for _, nc in ipairs(self._nWords) do
			self.gradCell[_]:add(_gP:narrow(1, curid, nc):sum(1))
			curid = curid + 1
		end
		local _gradInput = {}
		curid = 1
		self.gradSEnc = gradCache:narrow(2, self._csind, self._csize)
		local _gP1 = gradCache:narrow(2, self._sind, self._isize)
		for _, v in ipairs(hinput) do
			local nc = self._nWords[_]
			local _curGradO = self.gradSEnc:narrow(1, curid, nc)
			_curGradO[-1]:add(self.gradCell[_])
			local _curGradF, _curGrad = unpack(self:net(_):updateGradInput({feat, v}, _curGradO))
			_curGrad:add(_gP1:narrow(1, curid, nc))
			table.insert(_gradInput, _curGrad)
			_gPFeat:add(_curGradF)
			curid = curid + nc
		end
		_gPFeat:add(gradCache:narrow(2, self._fsind, self._fsize):sum(1):squeeze(1))
		self.gradInput = {_gradInput, self.gradFeat}
		self.doupgi = true
	end
	return self.gradInput
end

function PFullTagger:accGradParameters(input, gradOutput, scale)
	if not self.doupgi then
		self:updateGradInput(input, gradOutput)
	end
	local hinput, feat_full = unpack(input)
	local feat = feat_full[-1]
	self.CM:accGradParameters({self.cache, feat_full}, self.grad_output, scale)
	self.PEnc:updateGradInput({feat, self.cells}, self.gradPEnc)
	curid = 1
	for _, v in ipairs(hinput) do
		local nc = self._nWords[_]
		self:net(_):accGradParameters({feat, v}, self.gradSEnc:narrow(1, curid, nc), scale)
		curid = curid + nc
	end
end

function PFullTagger:backward(input, gradOutput, scale)
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
	local hinput, feat_full = unpack(input)
	local feat = feat_full[-1]
	local gradCache
	gradCache, self.gradFeat = unpack(self.CM:backward({self.cache, feat_full}, self.grad_output, scale))
	local _gPFeat = self.gradFeat[-1]
	local _gPEncSize = self.PEnc.output:size()
	if not self.gradPEnc:isSize(_gPEncSize) then
		self.gradPEnc:resize(_gPEncSize)
	end
	local curid = 1
	local _gP = gradCache:narrow(2, self._plsind, self._psize)
	for _, nc in ipairs(self._nWords) do
		self.gradPEnc[_]:copy(_gP:narrow(1, curid, nc):sum(1))
		curid = curid + nc
	end
	self.gradPEnc[-1]:add(gradCache:narrow(2, self._psind, self._psize):sum(1))
	local _gradFeat
	_gradFeat, self.gradCell = unpack(self.PEnc:backward({feat, self.cells}, self.gradPEnc, scale))
	_gPFeat:add(_gradFeat)
	curid = 1
	_gP = gradCache:narrow(2, self._clsind, self._csize)
	for _, nc in ipairs(self._nWords) do
		self.gradCell[_]:add(_gP:narrow(1, curid, nc):sum(1))
		curid = curid + 1
	end
	local _gradInput = {}
	curid = 1
	self.gradSEnc = gradCache:narrow(2, self._csind, self._csize)
	local _gP1 = gradCache:narrow(2, self._sind, self._isize)
	for _, v in ipairs(hinput) do
		local nc = self._nWords[_]
		local _curGradO = self.gradSEnc:narrow(1, curid, nc)
		_curGradO[-1]:add(self.gradCell[_])
		local _curGradF, _curGrad = unpack(self:net(_):backward({feat, v}, _curGradO, scale))
		_curGrad:add(_gP1:narrow(1, curid, nc))
		table.insert(_gradInput, _curGrad)
		_gPFeat:add(_curGradF)
		curid = curid + nc
	end
	_gPFeat:add(gradCache:narrow(2, self._fsind, self._fsize):sum(1):squeeze(1))
	self.gradInput = {_gradInput, self.gradFeat}
	self.doupgi = true
	return self.gradInput
end

function PFullTagger:evaluate()
	parent.evaluate(self)
	self.train = true
end

function PFullTagger:clearState()
	self.cells:set()
	self.cache:set()
	self._output:set()
	self.grad_output:set()
	self.gradPEnc:set()
	self.gradFeat:set()
	self.gradCell:set()
	self.gradSEnc:set()
	self._nWords = {}
	self._totalWords = 0
	self.doupgi = nil
	return parent.clearState(self)
end