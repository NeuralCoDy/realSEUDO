
% Show an extracted activation trace.
%
% Args:
% traces - traces to show
% equalsize - 1 to fit all the traces into the same vertical spacing,
%   0 to reduce the spacing to the bounding box for each trace to
%   conserve space. In both cases the vertical scaling of all the graphs
%   is the same within the figure. 2 to show each graph at its own scale,
%   fully filling the bounding box, the bounding boxes the same for all graphs.
% Parameters:
% filename - if specified, write the result into a file instead
%   of showing it
% rois - if specified, the ROI images will be put next to each trace;
%   if the rois have the oprional field "color" set, then the value 2
%   switches the color palette for this roi to green, 3 to red, the rest are blue
% heatmap - if specified, show a heatmap outline along with ROI image
% thickness - on heatmap, make the outlines this many pixels thick
% linewidth - line width for the graphs (0 means default)
% targetnum - target this many traces per page for the vertical sizing
%    (will be auto-set to at least the actual number)
function show_trace(traces, equalsize, varargin)
	p = inputParser;
	validateFname = @(x) isempty(x) || ischar(x) || isstring(x);
	p.addParameter('filename', [], validateFname);
	p.addParameter('rois', []);
	p.addParameter('heatmap', []);
	p.addParameter('thickness', 5);
	p.addParameter('linewidth', 0);
	p.addParameter('targetnum', 0);
	parse(p,varargin{:});
	filename = p.Results.filename;
	rois = p.Results.rois;
	heatmap = p.Results.heatmap;
	thickness = p.Results.thickness;
	linewidth = p.Results.linewidth;
	targetnum = p.Results.targetnum;

	if linewidth < 1
		linewidth = 1;
		if length(filename) ~= 0
			linewidth = 5;
		end
	end

	figscale = 1;
	if length(filename) ~= 0
		% for a better-quality bitmap output, increase figure scale
		figscale = 2;
	end

	fig = figure;

	n = size(traces, 1);
	nframes = size(traces, 2);
	maxvals = max(traces, [], 2)';
	minvals = min(traces, [], 2)';

	nbreaks = 0;
	% how the breaks are vertically sized relative to the graphs
	breakspace = 0.3;
	if ~isempty(rois)
		for i = 1:length(rois)
			if isfield(rois(i), 'isbreak') && rois(i).isbreak
				nbreaks = nbreaks + 1;
			end
		end
	end

	if equalsize
		% just divide the space equally
		scale = n + nbreaks * breakspace;
		if targetnum > scale
			scale = targetnum;
		end
	else
		% find normalization so that the sum of all ranges min to max is 1
		scale = sum(maxvals - minvals);
	end

	maxtop = max(maxvals);
	minbot = min(minvals);

	topmargin = 0.05;
	botmargin = 0.05;

	% The cells are placed in order from the top, but the coordinates
	% go from the bottom.
	vpos = 1 - topmargin;
	ri = 0; % rois index
	for i = 1:n
		insbreak = 0; % how many breaks to insert
		ri = ri + 1;
		while ~isempty(rois) && isfield(rois(ri), 'isbreak') && rois(ri).isbreak
			insbreak = insbreak + 1;
			ri = ri + 1;
		end

		if equalsize == 2
			minv = minvals(i);
			maxv = maxvals(i);
			if maxv <= minv
				maxv = minv + 1;
			end
			vsize = (1 - topmargin - botmargin) / scale;
			vpos = vpos - insbreak * breakspace * vsize;
		elseif equalsize == 1
			minv = minbot;
			maxv = maxtop;
			vsize = (1 - topmargin - botmargin) / scale;
			vpos = vpos - insbreak * breakspace * vsize;
		else
			minv = minvals(i);
			maxv = maxvals(i);
			vsize = (1 - topmargin - botmargin) * (maxv-minv) / scale;
		end
		vpos = vpos - vsize;
		hpos = 0.15;
		hsize = 0.75;
		color = 'b';

		% axis for writing a tag
		tagax = axes('Position', [0.1, vpos, 0.05, vsize]);
		tagax.FontSize = 6;
		tagax.XTickLabelMode = 'manual';
		tagax.XTickLabels = [];
		tagax.XAxis.Visible = 'off';
		tagax.YTickLabelMode = 'manual';
		tagax.YTickLabels = [];
		tagax.YAxis.Visible = 'off';

		if n > 1
			if ~isempty(rois) && isfield(rois(ri), 'original_id')
				tag = rois(ri).original_id;
			else
				tag = i;
			end
			text(tagax, 0, 0.5, '  ' + string(tag), 'FontSize', 6*figscale);
		end

		if ~isempty(rois)
			imgax = axes('Position', [hpos, vpos, 0.05, vsize]);
			hpos = hpos + 0.05;
			hsize = hsize - 0.05;

			% zero out the occasional below-0 values to avoid messing up the background
			imagesc(imgax, rois(ri).normalized2 .* (rois(ri).normalized2 > 0), 'CDataMapping', 'scaled');

			imgax.XTickLabelMode = 'manual';
			imgax.XTickLabels = [];
			imgax.YTickLabelMode = 'manual';
			imgax.YTickLabels = [];

			cmap = colormap(imgax);

			if isfield(rois(ri), 'color')
				if rois(ri).color == 2
					color = 'g';
					cmap = [cmap(:,2), cmap(:,3), cmap(:, 1)];
				elseif rois(ri).color == 3
					color = 'r';
					cmap = [cmap(:,3), cmap(:,1), cmap(:, 2)];
				end
				colormap(imgax, cmap);
			end

			if ~isempty(heatmap)
				outlax = axes('Position', [hpos, vpos, 0.05, vsize]);
				hpos = hpos + 0.05;
				hsize = hsize - 0.05;

				% zero out the occasional below-0 values to avoid messing up the background
				imagesc(outlax, rois_outlines(heatmap, rois(ri), [], 'upscale', 1, 'thickness', thickness), 'CDataMapping', 'scaled');

				ocmap = cmap;
				% remap the color for outlines (255) to be drawn in white instead of yellow
				ocmap(256,:) = [1, 1, 1];
				ocmap(255,:) = [1, 1, 1];
				colormap(outlax, ocmap);

				outlax.XTickLabelMode = 'manual';
				outlax.XTickLabels = [];
				outlax.YTickLabelMode = 'manual';
				outlax.YTickLabels = [];
			end
		end

		ax = axes('Position', [hpos, vpos, hsize, vsize], ...
			'XGrid', 1, ...
			'Box', 'on');

		plot([1:nframes], traces(i, :), 'LineWidth', linewidth, 'Color', color);
		axis(ax, [1, nframes, minv, maxv]);

		midtick = floor((minv + maxv) / 2);
		if midtick < 1
			midtick = (minv + maxv) / 2;
		end
		if midtick > 0
			ax.YTick = [0, midtick];
		else
			ax.YTick = [0];
		end

		ax.XGrid = 1;
		ax.GridColor = [0 0 0];
		ax.GridAlpha = 0.3;
		ax.FontSize = 6*figscale;

		ax.YAxisLocation = 'right';

		% put the tick labels at the very top and bottom only
		if i ~= 1 && i ~= n
			ax.XTickLabelMode = 'manual';
			ax.XTickLabels = [];
		end
		if i == 1 && n > 1
			ax.XAxisLocation = 'top';
		end
	end

	if length(filename) ~= 0
		%fig.Position = [0, 0, 1000, 5000];
		% fig.InnerPosition = [0, 0, 1000, 5000];
		fig.PaperPosition = [0.5*figscale, 0.5*figscale, 7.5*figscale, 10*figscale];
		fig.PaperSize = [8.5*figscale, 11*figscale];
		fig.PaperUnits = 'inches';
		saveas(fig, filename);
		close(fig);
	end
end
