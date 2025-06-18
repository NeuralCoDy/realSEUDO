% Select the ROIs from the merge trace of a particular ROI that get merged
% at a particular frame
function rois = get_merge(r, frame_id)
	rois = [];
	stack = [];
	for i = 1:length(r.merge_trace)
		f = r.merge_trace{i, 2};
		if f ~= frame_id
			continue;
		end
		op = mtt(r.merge_trace, i, 1);
		troi = mtt(r.merge_trace, i, 4);

		if op == "<}"
			stack = [stack troi];
		elseif op == ">}"
			roi1 = stack(end);
			stack = stack(1:end-1);
			rois = [rois roi1 troi];
		elseif op == "<+" || op == ">+" || op == "<-" || op == ">-"
			rois = [rois troi];
		end
	end
end
