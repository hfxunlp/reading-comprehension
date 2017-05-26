local CPFullTagger, parent = torch.class('nn.CPFullTagger', 'nn.SequenceContainer')

-- SentEnc should enable fullout
-- PEnc should enable fullout
function CPFullTagger:__init(SentEnc, PEnc, Classifier, flatten, ploc)
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
function CPFullTagger:updateOutput(input)
	local function expandT(tin, nexp)
		local usize = tin:size()
		local bsize = usize[1]
		local vsize = usize[2]
		return tin:reshape(1, bsize, vsize):expand(nexp, bsize, vsize)
	end
	local hinput, feat_full_c = unpack(input)--输入篇章向量以及问题向量
	local feat_full = feat_full_c:narrow(3, 1, self._fsize or feat_full_c:size(3)/2)--取出问题词向量
	local feat = feat_full[-1]:narrow(3, self._sqind or feat_full_c:size(3)/2 + 1, self._fsize or feat_full_c:size(3)/2)--取出问题向量
	local seql = #hinput--输入篇章中句子数量
	local _output = self:net(1):updateOutput({feat, hinput[1]})[-1]--取出句子编码器中最后一个向量
	local _osize = _output:size()--获取句子编码器输出尺寸
	local stdSize = torch.LongStorage({seql, _osize[1], self._csize or _osize[2]/2})--生成篇章编码器输入尺寸
	if not self.cells:isSize(stdSize) then
		self.cells:resize(stdSize)--为篇章编码器配置输入内存
	end
	self.cells[1]:copy(_output:narrow(2, self._scind or 1 + _osize[2]/2, self._csize or _osize[2]/2))--放置篇章编码器输入
	for _, sent in ipairs(hinput) do
		if _ > 1 then
			self.cells[_]:copy(self:net(_):updateOutput({feat, sent})[-1]:narrow(2, self._scind or 1 + _osize[2]/2, self._csize or _osize[2]/2))
		end
	end
	local _pEnc_full_c = self.PEnc:updateOutput({feat, self.cells})--篇章编码器编码
	if not self._isize then
		self._isize = hinput[1]:size(3)--输入维度
		self._csize = self.cells:size(3)--句向量维度
		self._gcsize = self._csize * 2--到句向量编码器的误差的维度
		self._scind = self._csize + 1--句向量片索引
		self._psize = _pEnc_full_c:size(3) / 2--篇章向量维度
		self._spind = self._psize + 1--篇章向量片索引
		self._fsize = feat:size(2)--问题向量维度
		self._sqind = self._fsize + 1--问题片向量索引
		self._sind = self.ploc + 1--可用cache起始地址
		self._csind = self._sind + self._isize--句词特征起始索引
		self._clsind = self._csind + self._csize--句特征起始索引
		self._plsind = self._clsind + self._csize--篇章句特征起始索引
		self._psind = self._plsind + self._psize--篇章特征起始索引
		self._fsind = self._psind + self._psize--问题特征起始索引
		self._cachedim = self.ploc + self._isize + self._csize * 2 + self._psize * 2 + self._fsize--cache尺寸
	end
	local _pEnc_full = _pEnc_full_c:narrow(3, 1, self._psize)--篇章句特征
	local _pEnc = _pEnc_full_c[-1]:narrow(3, self._spind, self._psize)--篇章特征
	self._nWords = {}
	self._totalWords = 0
	for _, v in ipairs(hinput) do
		local curwds = v:size(1)
		table.insert(self._nWords, curwds)
		self._totalWords = self._totalWords + curwds
	end--统计篇章长度信息
	stdSize = torch.LongStorage({self._totalWords, self._cachedim})--配置cache尺寸
	if not self.cache:isSize(stdSize) then
		self.cache:resize(stdSize)
	end
	self.cache:narrow(2, self._psind, self._psize):copy(expandT(_pEnc, self._totalWords))--放置篇章特征
	self.cache:narrow(2, self._fsind, self._fsize):copy(expandT(feat, self._totalWords))--放置问题特征
	local curid = 1
	for _, nc in ipairs(self._nWords) do
		local curT = self.cache:narrow(1, curid, nc)
		curT:narrow(2, self._sind, self._isize):copy(hinput[_])--放置输入向量
		curT:narrow(2, self._csind, self._csize):copy(self:net(_).output:narrow(3, 1, self._csize))--放置句词特征
		curT:narrow(2, self._clsind, self._csize):copy(expandT(self.cells[_], nc))--放置句特征
		curT:narrow(2, self._plsind, self._psize):copy(expandT(_pEnc_full[_], nc))--放置篇章句特征
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

function CPFullTagger:updateGradInput(input, gradOutput)
	if not self.doupgi then
		if self.flatten then
			self.grad_output = gradOutput
		else
			local usize = gradOutput[1]:size()
			local stdSize = torch.LongStorage({self._totalWords, usize[1], usize[2]})
			if not self.grad_output:isSize(stdSize) then
				self.grad_output:resize(stdSize)
			end
			local curid = 1
			for _, nc in ipairs(self._nWords) do
				self.grad_output:narrow(1, curid, nc):copy(gradOutput[_])
				curid = curid + nc
			end
		end
		local hinput, feat_full_c = unpack(input)--输入篇章向量以及问题向量
		local feat_full = feat_full_c:narrow(3, 1, self._fsize)--取出问题词向量
		local feat = feat_full[-1]:narrow(3, self._sqind, self._fsize)--取出问题向量
		if not self.gradFeat:isSize(feat_full_c:size()) then
			self.gradFeat:resizeAs(feat_full_c)
		end
		self.gradFeat:narrow(3, self._sqind, self._fsize):zero()--准备到feat的grad
		local gradCache, _gP = unpack(self.CM:updateGradInput({self.cache, feat_full}, self.grad_output))--分类模型反向传播
		self.gradFeat:narrow(3, 1, self._fsize):copy(_gP)--传递到问题词特征的误差
		local _gPFeat = self.gradFeat[-1]:narrow(3, self._sqind, self._fsize)--锁定问题特征的误差
		local _gPEncSize = self.PEnc.output:size()
		if not self.gradPEnc:isSize(_gPEncSize) then
			self.gradPEnc:resize(_gPEncSize)
		end--准备篇章编码器的误差
		self.gradPEnc:narrow(3, self._spind, self._psize):zero()
		local curid = 1
		_gP = gradCache:narrow(2, self._plsind, self._psize)--锁定篇章句特征传入误差
		local _gP1 = self.gradPEnc:narrow(3, 1, self._psize)--锁定篇章句特征误差
		for _, nc in ipairs(self._nWords) do
			_gP1[_]:copy(_gP:narrow(1, curid, nc):sum(1))
			curid = curid + nc
		end
		self.gradPEnc[-1]:narrow(3, self._spind, self._psize):copy(gradCache:narrow(2, self._psind, self._psize):sum(1))--传递篇章特征误差
		_gP, self.gradCell = unpack(self.PEnc:updateGradInput({feat, self.cells}, self.gradPEnc))--篇章编码器反向传播
		_gPFeat:copy(_gP)--传递问题特征误差
		curid = 1
		_gP = gradCache:narrow(2, self._clsind, self._csize)--锁定句特征误差
		for _, nc in ipairs(self._nWords) do
			self.gradCell[_]:add(_gP:narrow(1, curid, nc):sum(1))--累积来自分类器的句特征误差
			curid = curid + 1
		end
		local stdSize = torch.LongStorage({self._totalWords, 1, self._gcsize})
		if not self.gradSEnc:isSize(stdSize) then
			self.gradSEnc:resize(stdSize)
		end
		_gP = self.gradSEnc:narrow(3, self._scind, self._csize)--锁定到句特征的误差
		_gP:zero()--清空
		self.gradSEnc:narrow(3, 1, self._csize):copy(gradCache:narrow(2, self._csind, self._csize))--传递分类器到句词特征的误差
		_gP1 = gradCache:narrow(2, self._sind, self._isize)--锁定到输入词向量的误差
		local _gradInput = {}
		curid = 1
		local sid = 0
		for _, v in ipairs(hinput) do
			local nc = self._nWords[_]
			sid = sid + nc
			_gP[sid]:copy(self.gradCell[_])--传递到句特征的误差
			local _curGradF, _curGrad = unpack(self:net(_):updateGradInput({feat, v}, self.gradSEnc:narrow(1, curid, nc)))--句编码器反向传播
			_curGrad:add(_gP1:narrow(1, curid, nc))--累加来自分类器的误差
			table.insert(_gradInput, _curGrad)
			_gPFeat:add(_curGradF)--积累到问题特征的误差
			curid = curid + nc
		end
		_gPFeat:add(gradCache:narrow(2, self._fsind, self._fsize):sum(1):squeeze(1))--积累从分类器到问题特征的误差
		self.gradInput = {_gradInput, self.gradFeat}
		self.doupgi = true
	end
	return self.gradInput
end

function CPFullTagger:accGradParameters(input, gradOutput, scale)
	if not self.doupgi then
		self:updateGradInput(input, gradOutput)
	end
	local hinput, feat_full_c = unpack(input)
	local feat_full = feat_full_c:narrow(3, 1, self._fsize)
	local feat = feat_full[-1]:narrow(3, self._sqind, self._fsize)
	self.CM:accGradParameters({self.cache, feat_full}, self.grad_output, scale)
	self.PEnc:accGradParameters({feat, self.cells}, self.gradPEnc, scale)
	local curid = 1
	for _, v in ipairs(hinput) do
		local nc = self._nWords[_]
		self:net(_):accGradParameters({feat, v}, self.gradSEnc:narrow(1, curid, nc), scale)
		curid = curid + nc
	end
end

function CPFullTagger:backward(input, gradOutput, scale)
	if self.flatten then
		self.grad_output = gradOutput
	else
		local usize = gradOutput[1]:size()
		local stdSize = torch.LongStorage({self._totalWords, usize[1], usize[2]})
		if not self.grad_output:isSize(stdSize) then
			self.grad_output:resize(stdSize)
		end
		local curid = 1
		for _, nc in ipairs(self._nWords) do
			self.grad_output:narrow(1, curid, nc):copy(gradOutput[_])
			curid = curid + nc
		end
	end
	local hinput, feat_full_c = unpack(input)--输入篇章向量以及问题向量
	local feat_full = feat_full_c:narrow(3, 1, self._fsize)--取出问题词向量
	local feat = feat_full[-1]:narrow(3, self._sqind, self._fsize)--取出问题向量
	if not self.gradFeat:isSize(feat_full_c:size()) then
		self.gradFeat:resizeAs(feat_full_c)
	end
	self.gradFeat:narrow(3, self._sqind, self._fsize):zero()--准备到feat的grad
	local gradCache, _gP = unpack(self.CM:backward({self.cache, feat_full}, self.grad_output, scale))--分类模型反向传播
	self.gradFeat:narrow(3, 1, self._fsize):copy(_gP)--传递到问题词特征的误差
	local _gPFeat = self.gradFeat[-1]:narrow(3, self._sqind, self._fsize)--锁定问题特征的误差
	local _gPEncSize = self.PEnc.output:size()
	if not self.gradPEnc:isSize(_gPEncSize) then
		self.gradPEnc:resize(_gPEncSize)
	end--准备篇章编码器的误差
	self.gradPEnc:narrow(3, self._spind, self._psize):zero()
	local curid = 1
	_gP = gradCache:narrow(2, self._plsind, self._psize)--锁定篇章句特征传入误差
	local _gP1 = self.gradPEnc:narrow(3, 1, self._psize)--锁定篇章句特征误差
	for _, nc in ipairs(self._nWords) do
		_gP1[_]:copy(_gP:narrow(1, curid, nc):sum(1))
		curid = curid + nc
	end
	self.gradPEnc[-1]:narrow(3, self._spind, self._psize):copy(gradCache:narrow(2, self._psind, self._psize):sum(1))--传递篇章特征误差
	_gP, self.gradCell = unpack(self.PEnc:backward({feat, self.cells}, self.gradPEnc, scale))--篇章编码器反向传播
	_gPFeat:copy(_gP)--传递问题特征误差
	curid = 1
	_gP = gradCache:narrow(2, self._clsind, self._csize)--锁定句特征误差
	for _, nc in ipairs(self._nWords) do
		self.gradCell[_]:add(_gP:narrow(1, curid, nc):sum(1))--累积来自分类器的句特征误差
		curid = curid + 1
	end
	local stdSize = torch.LongStorage({self._totalWords, 1, self._gcsize})
	if not self.gradSEnc:isSize(stdSize) then
		self.gradSEnc:resize(stdSize)
	end
	_gP = self.gradSEnc:narrow(3, self._scind, self._csize)--锁定到句特征的误差
	_gP:zero()--清空
	self.gradSEnc:narrow(3, 1, self._csize):copy(gradCache:narrow(2, self._csind, self._csize))--传递分类器到句词特征的误差
	_gP1 = gradCache:narrow(2, self._sind, self._isize)--锁定到输入词向量的误差
	local _gradInput = {}
	curid = 1
	local sid = 0
	for _, v in ipairs(hinput) do
		local nc = self._nWords[_]
		sid = sid + nc
		_gP[sid]:copy(self.gradCell[_])--传递到句特征的误差
		local _curGradF, _curGrad = unpack(self:net(_):backward({feat, v}, self.gradSEnc:narrow(1, curid, nc), scale))--句编码器反向传播
		_curGrad:add(_gP1:narrow(1, curid, nc))--累加来自分类器的误差
		table.insert(_gradInput, _curGrad)
		_gPFeat:add(_curGradF)--积累到问题特征的误差
		curid = curid + nc
	end
	_gPFeat:add(gradCache:narrow(2, self._fsind, self._fsize):sum(1):squeeze(1))--积累从分类器到问题特征的误差
	self.gradInput = {_gradInput, self.gradFeat}
	self.doupgi = true
	return self.gradInput
end

function CPFullTagger:evaluate()
	parent.evaluate(self)
	self.train = true
end

function CPFullTagger:clearState()
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