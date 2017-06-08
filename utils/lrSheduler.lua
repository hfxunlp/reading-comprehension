local lrSheduler = torch.class("lrSheduler")

function lrSheduler:__init(startlr, minlr, expdecaycycle, lrdecaycycle, earlystop, nsave_train, nsave_dev, autosave, save_debug, savethead, savevhead, savedhead, savetail, savemodel, logger, record_err, terrf, derrf, mindeverrate, minerrate)
		self.startlr = startlr
		self.lr = startlr
		self.minlr = (minlr==nil) and startlr/8192 or minlr
		if self.minlr > self.startlr then
			self.minlr = self.startlr
		end
		self.expdecaycycle = expdecaycycle or 1
		self.lrdecaycycle = lrdecaycycle or 1
		self.lrdecayepochs = 1
		self.earlystop = earlystop
		self.nsave_train = nsave_train or 1
		self.nsave_dev = nsave_dev or 1
		self.nsave_debug = (save_debug==nil) and 1 or save_debug
		self.storet = 1
		self.storev = 1
		self.stored = 1
		self.amindeverr = 1
		self.aminerr = 1
		self.mindeverrate = mindeverrate or math.huge
		self.minerrate = minerrate or math.huge
		self.autosave = autosave
		self.savethead = savethead
		self.savevhead = savevhead
		self.savedhead = savedhead
		self.savetail = savetail
		self.module = savemodel
		self.logger = logger
		self.record = record_err
		self.crit = {}
		self.critdev = {}
		self.terrf = terrf
		self.derrf = derrf
end

function lrSheduler:feed(trainerr, deverr, nosave, silent)
	local store_file = nil
	local runsig = 1
	local savefor = nil
	if  deverr then
		if self.record then
			table.insert(self.critdev, deverr)
		end
		if deverr <= self.mindeverrate then--assume that new model is better
			self.mindeverrate = deverr
			self.amindeverr = 1
			self.aminerr = 1 -- this is not correct! but have some reason
			if self.nsave_dev > 1 then
				store_file = self.savevhead..tostring(self.storev)..self.savetail
				if self.storev >= self.nsave_dev then
					self.storev = 1
				else
					self.storev = self.storev + 1
				end
			else
				store_file = self.savevhead..self.savetail
			end
			savefor = "dev"
		else
			if self.earlystop and self.amindeverr > self.earlystop then
				runsig = false
				if self.logger then
					self.logger:log("send earlystop signal")
				end
			end
			self.amindeverr = self.amindeverr + 1
		end
	end
	if trainerr then
		if self.record then
			table.insert(self.crit, trainerr)
		end
		if trainerr <= self.minerrate then--assume that new model is better
			self.minerrate = trainerr
			self.aminerr = 1
			if not store_file then
				if self.nsave_train > 1 then
					store_file = self.savethead..tostring(self.storet)..self.savetail
					if self.storet >= self.nsave_train then
						self.storet = 1
					else
						self.storet = self.storet + 1
					end
				else
					store_file = self.savethead..self.savetail
				end
				savefor = "train"
			end
		else
			if self.nsave_debug then
				if self.nsave_debug > 1 then
					store_file = self.savedhead..tostring(self.stored)..self.savetail
					if self.stored >= self.nsave_debug then
						self.stored = 1
					else
						self.stored = self.stored + 1
					end
				else
					store_file = self.savedhead..self.savetail
				end
				savefor = "debug"
			end
			if self.aminerr >= self.expdecaycycle then
				self.aminerr = 1
				if self.minlr then
					if self.lr > self.minlr then
						if self.lrdecayepochs > self.lrdecaycycle then
							self.startlr = self.lr
							self.lr = self.startlr / 2
							self.lrdecayepochs = 2
						else
							self.lrdecayepochs = self.lrdecayepochs + 1
							self.lr = self.startlr / self.lrdecayepochs
						end
						if self.lr > self.minlr then
							self.lr = self.minlr
							if self.logger then
								self.logger:log("minimal lr reached")
							end
						end
					end
				else
					if self.lrdecayepochs > self.lrdecaycycle then
						self.startlr = self.lr
						self.lr = self.startlr / 2
						self.lrdecayepochs = 2
					else
						self.lrdecayepochs = self.lrdecayepochs + 1
						self.lr = self.startlr / self.lrdecayepochs
					end
				end
			else
				self.aminerr = self.aminerr + 1
			end
		end
	end
	if self.autosave and store_file and (not nosave) then
		self:saveModel(store_file)
		if self.logger and (not silent) then
			if savefor == "dev" then
				self.logger:log("new minimal dev error found, model saved")
			elseif savefor == "train" then
				self.logger:log("new minimal error found, model saved")
			elseif store_file then
				self.logger:log("debug model saved")
			end
		end
	end
	return self.lr, runsig, savefor, store_file
end

local function saveObject(fname,objWrt)
	if torch.isTypeOf(objWrt, "nn.Module") then
		objWrt:clearState()
		torch.save(fname, objWrt, 'binary', true)
	else
		torch.save(fname, objWrt, 'binary', false)
	end
end

function lrSheduler:saveModel(fname)
	saveObject(fname, self.module)
end

function lrSheduler:getLossHistory()
	return torch.FloatTensor(self.crit), torch.FloatTensor(self.critdev)
end

function lrSheduler:saveCrit()
	local terr, derr = self:getLossHistory()
	saveObject(self.terrf, terr)
	saveObject(self.derrf, derr)
end

function lrSheduler:clearState()
	self.crit = {}
	self.critdev = {}
end
