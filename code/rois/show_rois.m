% Shows an image with ROIs in a grid.
%
% Args:
% rois - the vector of ROIs, each represented as a structure, or a 3-d array as in SEUDO.
%   The .normalized element will be shown.
% normalized [optional] - if 0 then show raw values, if 1 [default] then show
%   the normalized1 values (as they are kept in main SEUDO code), if 2 then
%   normalized2 (as they are actually applied in SEUDO), if 3 then show the
%   values normalized by brightness (i.e. normalized3), otherwise raw;
%   has no effect on the 3-d array that gets always shown raw as-is
% filename [optional] - if specified, the image will be written to a file
%   instead of being shown
%
% Parameters:
% color - value 1, 2, or 3, selecting the color of image
function show_rois(rois, varargin)
	if length(rois) == 0
		return;
	end

	p = inputParser;
	p.addOptional('normalized', 1);
	validateFname = @(x) isempty(x) || ischar(x) || isstring(x);
	p.addOptional('filename', [], validateFname);
	p.addParameter('color', 0);
	parse(p,varargin{:});
	normalized = p.Results.normalized;
	filename = p.Results.filename;
	overcolor = p.Results.color;

	color = 1;
	if length(size(rois)) ~= 3
		if isfield(rois(1), 'color')
			color = rois(1).color;
		end
		[ht, wd] = size(rois(1).raw);
		if normalized == 1
			rois = reshape([rois(:).normalized1], ht, wd, []);
		elseif normalized == 2
			rois = reshape([rois(:).normalized2], ht, wd, []);
		elseif normalized == 3
			rois = reshape([rois(:).normalized3], ht, wd, []);
		else
			rois = reshape([rois(:).raw], ht, wd, []);
		end
	else
		[ht, wd] = size(rois, 1, 2);
	end

	square = ceil(sqrt(size(rois, 3)));

	maxval = max(reshape(rois, 1, []));
	col_sep = maxval * ones(ht, 1);
	row_sep = maxval * ones(1, 1 + square * (1 + wd));
	placeholder = zeros(ht, wd);

	im = row_sep;

	for r = 0:square-1
		row = col_sep;
		for c = 1:square
			idx = r * square + c;
			if idx > size(rois, 3)
				row = [row placeholder col_sep];
			else
				% sanitize by removing values < 0
				row = [row (rois(:, :, idx) .* (rois(:, :, idx) > 0)) col_sep];
			end
		end
		im = [im; row; row_sep];
	end

	resfig = figure;

	cmap = colormap(resfig);
	if overcolor ~= 0
		color = overcolor;
	end
	if color == 2
		cmap = [cmap(:,2), cmap(:,3), cmap(:, 1)];
	elseif color == 3
		cmap = [cmap(:,3), cmap(:,1), cmap(:, 2)];
	end
	colormap(resfig, cmap);

	if length(filename) == 0
		imagesc(im, [0 maxval]);
	else
		imwrite(round(im * (255 / maxval)), colormap(resfig), filename);
		close(resfig);
	end
end

