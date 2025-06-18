% Create an array of ROI objects from a plain representation.
%
% Args:
% plain - an array of plain matrices (X, Y, N)
%
% Options:
% trace - the array of traces for these ROIs (N, F)
% minsize - include only ROIs that have at least this many non-0 pixels
function rois = rois_from_plain(plain, varargin)
	p = inputParser;
	p.addParameter('trace', []);
	p.addParameter('minsize', 0);
	p.addParameter('params', []);
	parse(p,varargin{:});
	trace = p.Results.trace;
	minsize = p.Results.minsize;
	params = p.Results.params;

	if minsize == 0 && ~isempty(params)
		minsize = params.min_roi_size;
	end

	n = size(plain, 3);
	tracen = size(trace, 1);
	rois = [];

	if tracen > 0 && tracen ~= n
		error("Profile has a different count: " + string(tracen) + " traces vs " + string(n) + " rois");
	end

	if minsize > 0
		% indexes of ROIs to include
		idxs = find(sum(plain ~= 0, [1, 2]) >= minsize);
		plain = plain(:, :, idxs);
		if tracen == n
			trace = trace(idxs, :);
		end
		n = size(plain, 3);
		tracen = size(trace, 1);
	end

	blob_weights = [];

	for i = 1:n
		if ~isempty(params) && params.blobify
			frame = plain(:,:,i);
			[blob_weights, steps] = seudo_native(frame, params.blobify_blob, params.blob_spacing, ...
				[], blob_weights, [], params.fit_tol, params.fit_max_steps, ...
				params.fit_l_mode, params.fit_stop_mode, false, params.fit_parallel);

			% rebuild the smoothed frame from blobs and ROIs
			frame = convn(reshape(blob_weights, size(frame, 1), size(frame, 2)), params.blobify_blob, 'same');

			rr = roi_from_matrix(i, frame);
		else
			rr = roi_from_matrix(i, plain(:,:,i));
		end
		if tracen == n
			rr.trace = [trace(i, :)];
		end
		rois = [rois rr];
	end
end
