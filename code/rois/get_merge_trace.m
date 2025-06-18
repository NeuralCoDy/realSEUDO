% Extract all inputs, going pairwise, form the merge trace of a ROI.
%
% Args:
% r - ROI to examine
%
% Result:
% rois - ROIs that represent the inputs for every merge in the trace,
%   every two ROIs represent an input to a merge
function rois = get_merge_trace(r)
	rois = [];
	stack = [];
	for i = 1:size(r.merge_trace, 1)
		f = r.merge_trace{i, 2};
		op = r.merge_trace{i, 1};
		troi = r.merge_trace{i, 4};

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
