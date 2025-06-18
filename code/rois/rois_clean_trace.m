% Clean up the trace(s) by removing the noise at the bottom, similarly
% to how it's made with the video frames.
%
% Args:
% traces - the input set of traces, 2 dimensions: (rois, frames)
%
% Parameters:
% min_frames - the minimum number of frames to extend the traces too;
%  because it may differ in results of different algorithms when an
%  averaging across multiple frames reduces their count, this is a convenient
%  place to add 0s at the end, defaults to 0
% cutoff - the cut-off level for trace values, in multiples of
%  auto-detected noise amplitude, defaults to 1.1.
%
% Returns:
% otraces - the output set of traces after noise reduction and possible
%   0-extension
function otraces = rois_clean_trace(traces, varargin)
	p = inputParser;
	p.addParameter('min_frames', 0);
	p.addParameter('cutoff', 1.1);
	parse(p,varargin{:});
	min_frames = p.Results.min_frames;
	cutoff = p.Results.cutoff;

	n = size(traces, 1);
	nframes = size(traces, 2);

	otraces = zeros(n, max(nframes, min_frames));

	if nframes > 0
		for i = 1:n
			sorted = sort(traces(i, :));
			min_level = sorted(1);
			median_level = sorted(floor(nframes / 2));

			noise_amp = median_level - min_level;

			cutoff_level = median_level + noise_amp * cutoff;

			otraces(i, 1:nframes) = (traces(i, :) - median_level)  .* (traces(i, :) > cutoff_level);
		end
	end
end
