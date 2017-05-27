local AColl, parent = torch.class('nn.AColl', 'nn.Module')

--Warning : this module is not batchable now!

function AColl:__init()
	parent.__init(self)
end

function AColl:updateOutput(input)
	local passage, fscore = unpack(input)
	self.vocab = {}
	self.vcount = {}
	local score = {}
	local _fscore = fscore:reshape(fscore:size(1)):totable()
	local curwd = 1
	local curid = 1
	for _, sent in ipairs(passage) do
		for __, wd in ipairs(sent:reshape(sent:size(1)):totable()) do
			local wid
			if not self.vocab[wd] then
				self.vocab[wd] = curid
				wid = curid
				curid = curid + 1
				self.vcount[wd] = 1
			else
				wid = self.vocab[wd]
				self.vcount[wd] = self.vcount[wd] + 1
			end
			local curscore = _fscore[curwd] or 0
			curwd = curwd + 1
			score[wid] = (score[wid] or 0) + curscore
		end
	end
	for wd, wid in ipairs(self.vocab) do
		score[wid] = score[wid] / self.vcount[wd]
	end
	self.output = torch.Tensor(score):typeAs(fscore)
	return self.output
end

function AColl:updateGradInput(input, gradOutput)
	local function buildTableZero(tin)
		local rs = {}
		for _, v in ipairs(tin) do
			local tmp = v.new():resizeAs(v)
			tmp:zero()
			table.insert(rs, tmp)
		end
		return rs
	end
	for wd, wid in ipairs(self.vocab) do
		gradOutput[wid]:div(self.vcount[wd])
	end
	local passage, fscore = unpack(input)
	local gscore = {}
	for _, sent in ipairs(passage) do
		for __, wd in ipairs(sent:reshape(sent:size(1)):totable()) do
			table.insert(gscore, gradOutput[self.vocab[wd]])
		end
	end
	self.gradInput = {buildTableZero(passage), torch.Tensor(gscore):reshape(#gscore, 1):typeAs(fscore)}
	return self.gradInput
end

function AColl:clearState()
	self.vocab = {}
	return parent.clearState(self)
end