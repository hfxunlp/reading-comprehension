local fColl, parent = torch.class('nn.fColl', 'nn.Module')

--Warning : this module is not batchable now!

function fColl:__init()
	parent.__init(self)
end

function fColl:updateOutput(input)
	local passage, fscore = unpack(input)
	self.vocab = {}
	local score = {}
	local _fscore = fscore:reshape(fscore:size(1)):totable()
	local curwd = 1
	local curid = 1
	for _, wd in ipairs(passage:reshape(passage:size(1)):totable()) do
		local wid
		if not self.vocab[wd] then
			self.vocab[wd] = curid
			wid = curid
			curid = curid + 1
		else
			wid = self.vocab[wd]
		end
		local curscore = _fscore[curwd] or 0
		curwd = curwd + 1
		score[wid] = (score[wid] or 0) + curscore
	end
	self.output = torch.Tensor(score):typeAs(fscore)
	return self.output
end

function fColl:updateGradInput(input, gradOutput)
	local function buildZero(tin)
		local rs = tin.new():resizeAs(tin)
		rs:zero()
		return rs
	end
	local passage, fscore = unpack(input)
	local gscore = {}
	for _, wd in ipairs(passage:reshape(passage:size(1)):totable()) do
		table.insert(gscore, gradOutput[self.vocab[wd]])
	end
	self.gradInput = {buildZero(passage), torch.Tensor(gscore):reshape(#gscore, 1):typeAs(fscore)}
	return self.gradInput
end

function fColl:clearState()
	self.vocab = {}
	return parent.clearState(self)
end