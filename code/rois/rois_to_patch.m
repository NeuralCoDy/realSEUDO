% Convert the rois dimensions to become relative to the patch,
% discarding the ROIs that don't fully fit into the patch.
%
% Args:
% rois - array of ROIs to convert
% offset_x - patch offset by X, starting at 0
% offset_y - patch offset by Y, starting at 0
% sz_x - patch size by X
% sz_y - patch size by Y
%
% Results:
% orois - output ROIs that fit into the patch, translated and shrunk to
%   discard the empty space around the patch. The ids get updated
%   to the new order, stable_ids are left unchanged.
function orois = rois_to_patch(rois, offset_x, offset_y, sz_x, sz_y)
	from_x = 1 + offset_x;
	from_y = 1 + offset_y;
	to_x = sz_x + offset_x;
	to_y = sz_y + offset_y;

	orois = [];
	for i = 1:length(rois)
		r = rois(i);

		% fits fully into the patch
		fullfit = (r.bbox.row_min >= from_y && r.bbox.row_max <= to_y ...
			&& r.bbox.col_min >= from_x && r.bbox.col_max <= to_x);

		% fits at least partially into the patch
		partfit = (r.bbox.row_min <= to_y && r.bbox.row_max >= from_y ...
			&& r.bbox.col_min <= to_x && r.bbox.col_max >= from_x);
			
		if ~fullfit && ~partfit
			continue
		end

		r.id = length(orois) + 1;
		r.frame = r.frame(from_y:to_y, from_x: to_x);
		r.mask = r.mask(from_y:to_y, from_x: to_x);
		r.raw = r.raw(from_y:to_y, from_x: to_x);

		if fullfit
			r.bbox.row_min = r.bbox.row_min - offset_y;
			r.bbox.row_max = r.bbox.row_max - offset_y;
			r.bbox.col_min = r.bbox.col_min - offset_x;
			r.bbox.col_max = r.bbox.col_max - offset_x;
			r.bbox_direct = [r.bbox.row_min, r.bbox.row_max, r.bbox.col_min, r.bbox.col_max];

			r.normalized1 = r.normalized1(from_y:to_y, from_x: to_x);
			r.normalized2 = r.normalized2(from_y:to_y, from_x: to_x);
		else
			r.count = sum(r.mask, "all");
            if r.count == 0
                % bbox overlaps with the patch but the mask doesn't
                continue
            end

			norm2 = r.normalize_coeff2;
			trace = r.trace;

			% this also recomputes the bbox and wipes out the trace
			r = roi_postcreate(r);

			% adjust the trace for normalization
			if ~isnan(trace)
				trace = trace .* (r.normalize_coeff2 / norm2);
			end
			r.trace = trace;
		end

		orois = [orois r];
	end
end


