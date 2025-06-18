% Create a ROI object from a plain matrix of points.
%
% Args:
% id - id for the ROI
% matrix - the matrix of raw values
function r = roi_from_matrix(id, matrix)
	r.id = id;
	r.stable_id = id;
	r.frame_id = 0;
	r.frame_id_trace = [0 0];
	r.event_frame_id = 0;
	r.merge_trace = {};

	r.frame = matrix;
	r.mask_blur_rad = 0;
	r.min_roi_size = 0;
	r.min_avg_px = 0;
	r.cutoff = 0;

	r.mask = matrix  > 0;
	r.raw = matrix;
	r.count = sum( reshape( r.mask, 1, []) );

	r.subtracted = 0;
	r = roi_postcreate(r);
end
