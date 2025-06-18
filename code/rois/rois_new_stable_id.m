% Generate a new stable id for ROIs, as a sequence.
% The id is guaranteed to be >= 0.
function res = rois_new_stable_id()
	persistent id;
	if isempty(id)
		id = 1;
	else
		id = id + 1;
	end
	res = id;
end
