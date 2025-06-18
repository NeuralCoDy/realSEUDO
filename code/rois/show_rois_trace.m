
% Extract and show a time trace of a ROI. The events get loaded from rois_event("get").
%
% Args:
% rois - the ROIs to trace
% equalsize - 1 to fit all the traces into the same vertical spacing,
%   0 to reduce the spacing to the bounding box for each trace to
%   conserve space. In both cases the vertical scaling of all the graphs
%   is the same within the figure.
% Parameters:
% filename - if specified, write the result into a file instead
%   of showing it
% heatmap - if specified, show a heatmap outline along with ROI image
% thickness - on heatmap, make the outlines this many pixels thick
function show_rois_trace(rois, equalsize, varargin)
	p = inputParser;
	validateFname = @(x) isempty(x) || ischar(x) || isstring(x);
	p.addParameter('filename', [], validateFname);
	p.addParameter('heatmap', []);
	p.addParameter('thickness', 5);
	parse(p,varargin{:});
	filename = p.Results.filename;
	heatmap = p.Results.heatmap;
	thickness = p.Results.thickness;
	show_trace(rois_trace(rois), equalsize, 'filename', filename, 'rois', rois, 'heatmap', heatmap, 'thickness', thickness);
end
