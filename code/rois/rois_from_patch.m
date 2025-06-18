% Convert the rois dimensions from a patch back to a biger frame.
%
% Args:
% rois - array of ROIs to convert
% offset_x - patch offset by X, starting at 0
% offset_y - patch offset by Y, starting at 0
% sz_x - patch size by X
% sz_y - patch size by Y
% movie_sz_x - movie size by X
% movie_sz_y - movie size by Y
%
% Results:
% orois - output ROIs that fit into the patch, translated and expanded to
%   fit the place of the patch in the greater movie.
function orois = rois_from_patch(rois, offset_x, offset_y, sz_x, sz_y, movie_sz_x, movie_sz_y)
	from_x = 1 + offset_x;
	from_y = 1 + offset_y;
	to_x = sz_x + offset_x;
	to_y = sz_y + offset_y;

	% an empty movie frame, the patch in it will be repeatedly replaced
	% with various kinds of data
	movie = zeros(movie_sz_y, movie_sz_x);

	orois = [];
	for i = 1:length(rois)
		r = rois(i);

		movie(from_y:to_y, from_x: to_x) = r.frame;
		r.frame = movie;

		movie(from_y:to_y, from_x: to_x) = r.mask;
		r.mask = movie;

		movie(from_y:to_y, from_x: to_x) = r.raw;
		r.raw = movie;

		r.bbox.row_min = r.bbox.row_min + offset_y;
		r.bbox.row_max = r.bbox.row_max + offset_y;
		r.bbox.col_min = r.bbox.col_min + offset_x;
		r.bbox.col_max = r.bbox.col_max + offset_x;
		r.bbox_direct = [r.bbox.row_min, r.bbox.row_max, r.bbox.col_min, r.bbox.col_max];

		movie(from_y:to_y, from_x: to_x) = r.normalized1;
		r.normalized1 = movie;

		movie(from_y:to_y, from_x: to_x) = r.normalized2;
		r.normalized2 = movie;

		orois = [orois r];
	end
end


