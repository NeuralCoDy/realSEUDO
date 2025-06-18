% Show the ROIs from two sources in the order they are matched with
% each other.
%
% Args:
% ra - the ROIs from first source, must have .trace field set.
%      Will be marked with color 1 (blue), drive the ordering of traces,
%      and are always included.
% rb - the ROIs from second source, must have .trace field set.
%      Will be marked with color 3 (red).
% params - rois_params() 
%
% Parameters:
% filename - if specified, write the result into a file instead
%   of showing it (with pagesize, the file name is a printf pattern)
% qpdfname - automatically run qpdf on the result of a multi-page
%   plot, and collect the result into a single pdf file; this is the
%   result file name, requires pagesize > 0 and filename to specify
%   the pdf files.
% heatmap - if specified, show a heatmap outline along with ROI image
% thickness - on heatmap, make the outlines this many pixels thick
% linewidth - line width for the graphs (0 means default)
% allb (bool) - draw all the ROIs from rb, even if they have no good match
% score - pre-computed score matrix, will skip the computation
% filter (bool) - remove the low-quality matches when computing score
% minval - the minimal score to accept at all when computing it
% pagesize - number of traces per page; if both pagesize and
%   filename are used then filename must contain a printf pattern with %d
%   (or its variation) to insert the page number.
% equalsize - as equalsize parameter of show_trace(), 2 is the default
%   to account for differences in scale.

function show_matched_trace(ra, rb, params, varargin)
	p = inputParser;
	validateFname = @(x) isempty(x) || ischar(x) || isstring(x);
	p.addParameter('filename', [], validateFname);
	p.addParameter('qpdfname', [], validateFname);
	p.addParameter('heatmap', []);
	p.addParameter('thickness', 5);
	p.addParameter('linewidth', 0);
	p.addParameter('allb', 0);
	p.addParameter('score', []);
	p.addParameter('filter', 0);
	p.addParameter('minval', 0);
	p.addParameter('pagesize', 0);
	p.addParameter('equalsize', 2);
	parse(p,varargin{:});
	filename = p.Results.filename;
	qpdfname = p.Results.qpdfname;
	heatmap = p.Results.heatmap;
	thickness = p.Results.thickness;
	linewidth = p.Results.linewidth;
	allb = p.Results.allb;
	score = p.Results.score;
	filter = p.Results.filter;
	minval = p.Results.minval;
	pagesize = p.Results.pagesize;
	equalsize = p.Results.equalsize;

	na = length(ra);
	nb = length(rb);

	% all file names for qpdf
	allfiles = "";

	if pagesize > 0 && ~isempty(filename)
		f = sprintf(filename, 999);
		ff = sprintf("%s", filename);
		if f == ff
			error("If both params pagesize and filename are used, filename should contain a %d for the page number");
		end
	end

	if ~isempty(qpdfname) && isempty(filename)
		error("Parameter qpdfname requires parameter filename");
	end

	% mark the colors
	for i=1:na
		ra(i).color = 1;
		ra(i).original_id = i;
		ra(i).isbreak = 0;
	end
	for i=1:nb
		rb(i).color = 3;
		rb(i).original_id = i;
		rb(i).isbreak = 0;
	end

	% create a special pseudo-ROI to indicate a break
	% between matched groups
	breakroi = roi_from_matrix(0, [1]);
	breakroi.color = 1;
	breakroi.original_id = i;
	breakroi.isbreak = 1;

	% compute the normalized trace length
	alltracelen = [];
	for i=1:na
		alltracelen = [alltracelen length(ra(i).trace)];
	end
	for i=1:nb
		alltracelen = [alltracelen length(rb(i).trace)];
	end
	tracelen = max(alltracelen);
		
	if isempty(score)
		score = match_rois_all(ra, rb, params, 'filter', filter, 'minval', minval);
	end

	% grpid is the last used group id
	[groupa, groupb, grpid] = group_rois_by_score(score);

	if allb
		% start a new group
		grpid = grpid + 1;
		% effective group id
		egrp = grpid;

		% draw whatever is left in B
		count = 0;
		for b = 1:nb
			if ~groupb(b)
				groupb(b) = egrp;
				count = count + 1;

				% break up into groups of at most 5 at a time
				if count >= 5
					% start a new group
					grpid = grpid + 1;
					% effective group id
					egrp = grpid;
					count = 0;
				end
			end
		end
	end

	% the merged set of ROIs for drawing traces
	rois = [];
	% the break ROIs have no entry in traces
	traces = [];

	page = 1;

	% draw the groups
	for egrp = 1:grpid
		lista = find(groupa == egrp);
		listb = find(groupb == egrp);
		grpsz = length(lista) + length(listb);

		if grpsz < 1
			continue;
		end

		if pagesize > 0 && length(rois) + grpsz > pagesize
			% produce a page
			if ~isempty(filename)
				fname = sprintf(filename, page);
				allfiles = allfiles + fname + " ";
			else
				fname = [];
			end
			show_trace(traces, equalsize, 'heatmap', heatmap, 'rois', rois, ...
				'targetnum', length(rois), 'thickness', thickness, 'linewidth', linewidth, 'filename', fname);

			rois = [];
			traces = [];
			page = page + 1;
		end

		% Find and draw the ROIs from A
		for a = lista
			rois = [rois ra(a)];
			p = ra(a).trace;
			traces = [traces; p zeros(1, tracelen - length(p))];
		end

		% Find and draw the ROIs from B
		for b = listb
			rois = [rois rb(b)];
			p = rb(b).trace;
			traces = [traces; p zeros(1, tracelen - length(p))];
		end

		rois = [rois breakroi];
	end

	if isempty(filename)
		fname = [];
	elseif pagesize > 0
		fname = sprintf(filename, page);
		allfiles = allfiles + fname + " ";
	else
		fname = filename;
	end

	if pagesize > 0
		targetnum = max(length(rois), pagesize);
	else
		targetnum = 50;
	end

	show_trace(traces, equalsize, 'heatmap', heatmap, 'rois', rois, ...
		'targetnum', targetnum, 'thickness', thickness, 'linewidth', linewidth, ...
		'filename', fname);

	if pagesize > 0 && ~isempty(qpdfname)
		% reset the library path, since qpdf might not work with matlab's glibc
		system("LD_LIBRARY_PATH= qpdf --empty --pages " + allfiles + " -- " + string(qpdfname));
	end
end
