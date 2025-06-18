% Show a heatmap (summary of brightness across the movie) with outlines of the ROIs
% drawn over it.
%
% Args:
% frames - the movie frames (at least 5), or a single ready heatmap frame
% rois - the detected ROIs
% upscale [optional] - scaling factor to make the outlines thinner, use the
%   powers of 2, the typical good values are 4 (default) or 8
% filename [optional] - if specified, write the result into a file instead
%   of showing it
function show_rois_outlines(frames, rois, varargin)
	if size(rois, 2) == 0
		return;
	end

	p = inputParser;
	p.addOptional('upscale', 4, @isnumeric);
	p.addOptional('filename', [], @ischar);
	parse(p,varargin{:});
	upscale = p.Results.upscale;
	filename = p.Results.filename;

	if size(frames, 3) == 1
		% assume that the single frame is a ready heatmap
		hmap = frames(:, :, 1);
	else
		hmap = rois_heatmap(frames);
	end

	raw = rois_outlines(hmap, rois, [1:length(rois)], 'upscale', upscale);

	f = figure;
	cmap = colormap(f);
	% remap the color for outlines (255) to be drawn in white instead of yellow
	cmap(256,:) = [1, 1, 1];
	cmap(255,:) = [1, 1, 1];
	
	if length(filename) == 0
		colormap(f, cmap);
		img = image(raw, 'CDataMapping','scaled');
	else
		imwrite(round(raw), cmap, filename);
		close(f);
	end
end

