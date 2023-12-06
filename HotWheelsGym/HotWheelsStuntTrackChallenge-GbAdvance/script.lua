
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


previous_progress = 0
function calculateProgressReward()
	local current_progress = data.progress
	local delta = 0
	if current_progress > previous_progress then
		delta = 1
		previous_progress = current_progress
	end
	return delta
end


function calculateReward()
	return calculateProgressReward()
end
