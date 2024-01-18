
function isGameOver()
	-- end of game
	if data.lap > 3 then
		return true
	else
		return false
	end
end


function isDone()
	return isGameOver()
end


previous_checkpoint = 0
function calculatecheckpointReward()
	local current_checkpoint = data.checkpoint
	local delta = 0
	if current_checkpoint > previous_checkpoint then
		delta = 1
		previous_checkpoint = current_checkpoint
	end
	return delta
end


function calculateReward()
	return calculatecheckpointReward()
end
