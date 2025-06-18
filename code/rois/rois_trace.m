
% Extract a time trace of a list of ROIs. The events get loaded from rois_event("get").
% The value per frame is computed as the highest "detect" event brightness in
% this frame of any constituent part that went into building of this ROI, and
% if none are detected then of the highest constituent "detect_early".
%
% Args:
% rois - the ROIs to trace
%
% Parameters:
% norm_mode - same as in rois_params, to get proper datamust match the value
%  used when detecting the events
% nocached (bool) - skip the cached trace and get it afresh
% last_frame - the number of the last frame, if <= 0 then auto-guessed
%
% Returns a vector of trace values, one per frame of the movie
function trace = rois_trace(rois, varargin)
	debug = 0;

	p = inputParser;
	p.addParameter('norm_mode', 0);
	p.addParameter('nocached', 0);
	p.addParameter('last_frame', 0);
	parse(p,varargin{:});
	norm_mode = p.Results.norm_mode;
	nocached = p.Results.nocached;
	last_frame = p.Results.last_frame;

	last_event_frame = rois_event("get_last_frame");

	nrois = length(rois);

	if last_frame < 1
		if ~nocached && nrois > 0 
			% have a cached trace, use its length
			last_frame = 0;
			for i=1:nrois
				if length(rois(i).trace) > 1 || sum(~isnan(rois(i).trace))
					last_frame = max(last_frame, length(rois(i).trace));
				end
			end
			last_event_frame = last_frame;
		end
	end
	if last_frame < last_event_frame
		last_frame = last_event_frame;
	end
	trace = zeros(nrois, last_frame);
	if last_event_frame == 0
		return % there are no events
	end

	for ri = 1:nrois
		if ~nocached && length(rois(ri).trace) > 1 || sum(~isnan(rois(ri).trace))
			% found a cached trace
			trace(ri, 1:length(rois(ri).trace)) = rois(ri).trace;
			continue;
		end

		if norm_mode == 0
			trace(ri, :) = rois_event("get_trace", last_frame, rois(ri).stable_id, rois(ri).normalize_coeff2, norm_mode);
		elseif norm_mode == 1
			trace(ri, :) = rois_event("get_trace", last_frame, rois(ri).stable_id, rois(ri).total, norm_mode);
		elseif norm_mode == 2
			trace(ri, :) = rois_event("get_trace", last_frame, rois(ri).stable_id, rois(ri).brightness, norm_mode);
		else
			error "Unsupported normalization mode " + string(norm_mode);
		end
	end
end
