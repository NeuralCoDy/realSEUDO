
% Find the potential ROIs in a still frame from a movie. To reduce noise,
% the frame is created by averaging multiple sequential frames. This is a
% convenience wrapper over find_still_rois_in().
%
% Args:
% movie - frames of the movie as a 3-dimensional array (Y) * (X) * (total_frames)
% frame_id - first frame id out of the temporal fragment where we'll be searching for ROIs
% params - parameters produced with roi_params()
%
% Returns:
% rois - vector of found ROIs, if any. Each ROI is a structure with fields as
%  described in find_still_rois_in().
function rois = find_still_rois(movie, frame_id, params)
	frame = rois_extract_frame(movie, frame_id, params);
	rois = find_still_rois_in(frame, params);
end

