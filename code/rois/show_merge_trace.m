% Show all merges rogether in a merge trace or a ROI.
%
% Args:
% r - ROI to examine
function show_merge_trace(r)
	rois = get_merge_trace(r);
	show_rois(rois, 0)
end
