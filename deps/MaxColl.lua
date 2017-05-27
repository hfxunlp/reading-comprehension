local MaxColl, parent = torch.class('nn.MaxColl', 'nn.Module')

--Warning : this module is not batchable now!

function MaxColl:__init()
	parent.__init(self)
end

function MaxColl:updateOutput(input)
	local passage, fscore = unpack(input)
	self.vocab = {}
	self.mid = {}
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
			else
				wid = self.vocab[wd]
			end
			local curscore = _fscore[curwd] or -math.huge
			if not score[wid] then
				score[wid] = curscore
				self.mid[wd] = curwd
			elseif score[wid] < curscore then
				score[wid] = curscore
				self.mid[wd] = curwd
			end
			curwd = curwd + 1
		end
	end
	self.output = torch.Tensor(score):typeAs(fscore)
	return self.output
end

function MaxColl:updateGradInput(input, gradOutput)
	local function buildTableZero(tin)
		local rs = {}
		for _, v in ipairs(tin) do
			local tmp = v.new():resizeAs(v)
			tmp:zero()
			table.insert(rs, tmp)
		end
		return rs
	end
	local passage, fscore = unpack(input)
	local gscore = {}
	local _g = gradOutput:totable()
	local curwd = 1
	for _, sent in ipairs(passage) do
		for __, wd in ipairs(sent:reshape(sent:size(1)):totable()) do
			if curwd == self.mid[wd] then
				table.insert(gscore, _g[self.vocab[wd]])
			else
				table.insert(gscore, 0)
			end
			curwd = curwd + 1
		end
	end
	self.gradInput = {buildTableZero(passage), torch.Tensor(gscore):reshape(#gscore, 1):typeAs(fscore)}
	return self.gradInput
end

function MaxColl:clearState()
	self.vocab = {}
	self.mid = {}
	return parent.clearState(self)
end