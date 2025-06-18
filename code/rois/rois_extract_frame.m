% Extract and normalize one frame of a movie, according to parameters.
%
% Args:
% movie - frames of the movie as a 3-dimensional array (Y) * (X) * (total_frames)
% frame_id - first frame id out of the temporal fragment where we'll be searching for ROIs
% params - parameters produced with roi_params()
%
% Returns:
% frame - the frame described as a structure with fields:
%     frame_id - first frame id of the temporal fragment where we'll be searching for ROIs
%     ht - frame height
%     wd - frame width
%     pixels - the pixel values, preprocessed according to params
%     median - the median pixel value (normally the pixel values get pre-shifted to make
%       the median 0)
%     medadj - the matrix of pixel adjustment from the chunked median (i.e. the median
%       gets computed by chunks to find the local variations in lighting, and is then
%       used to even out these variations in lighting by subtracting the computed
%       variations), or an empty mattrix if not computed. Adding this matrix back can
%       be used to obtain the original frame pixels.
%     noise_level - the auto-determined noise level, above median by as much as the
%       median is above the lowest pixel
%     cutoff - the cut-off pixel value for inclusion into ROIs, gets driven by the
%       parameters and the noise level
function frame = rois_extract_frame(movie, frame_id, params)
	frame.frame_id = frame_id;

	movsz = size(movie);
	frame.ht = movsz(1);
	frame.wd = movsz(2);
	
	% Build the frame for detection by averaging the frames from the temporal
	% fragment.
	if params.avg_frames > 1
		% The meaning is this but more Matlab-efficient:
		%   frame.pixels = movie(:, :, frame_id);
		%   for ff = 1: params.avg_frames-1
		%     frame.pixels = frame.pixels + movie(:, :, frame_id + ff);
		%   end
		%   frame.pixels = frame.pixels ./ params.avg_frames;
		frame.pixels = sum(movie(:, :, frame_id:frame_id+params.avg_frames-1), 3) ./ params.avg_frames;
	else
		frame.pixels = movie(:, :, frame_id);
	end
	
	if params.med_chunks_x > 1 && params.med_chunks_y > 1
		frame.medadj = frame_median(frame.pixels, frame.wd / params.med_chunks_x, frame.ht / params.med_chunks_y);
		frame.pixels = frame.pixels - frame.medadj;
	else
		frame.medadj = [];
	end

	% The pixels will be considered belonging to the ROIs when they are bright,
	% i.e. high-contrast, on the darker background.  This starts by sorting all
	% the pixel values in the frame.

	min_pixel = min(frame.pixels, [], "all");
	median_pixel = median(frame.pixels, "all");

	% The normalized SEUDO inputs % contain the below-median values as negative.
	% The median can shift from averaging multiple frames, so this
	% adjustment would be needed even if the input data was already normalized like
	% this. But this also handles the case if the input data was not normalized yet.
	frame.median = 0;
	frame.pixels = frame.pixels - median_pixel;

	% The idea is that the difference between median and the lowest pixel
	% would represent half the amplitude of the noise. And then to get above noise,
	% we'd step by the same amount above the median.
	frame.noise_level = frame.median + median_pixel - min_pixel;

	% Determine the cut-off for "lighted" pixels based on the noise level and
	% parameters.
	if params.min_proper_px < 0
		frame.cutoff = -params.min_proper_px * frame.noise_level;
	else
		frame.cutoff = params.min_proper_px;
	end
end
