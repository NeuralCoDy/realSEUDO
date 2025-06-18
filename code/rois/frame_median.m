% Find the frame median in chunks
%
% Args:
% frame - frame (Y) * (X)
% chuwd - chunk width, will be rounded up to an even integer
% chuht - chunk height, will be rounded up to an even integer
%
% Returns:
% med - the same-sized map (Y) * (X) representing the chunk-wise median 
function med = frame_median(frame, chuwd, chuht)
	framesz = size(frame);
	fraht = framesz(1);
	frawd = framesz(2);

	% make the chunk dimensions even
	chuwd = ceil(chuwd);
	chuwd = chuwd + mod(chuwd, 2);
	chuht = ceil(chuht);
	chuht = chuht + mod(chuht, 2);

	med = zeros(framesz);

	% linear progression of weights as going horizontally from the left to the center of the chunk,
	% and then to the right
	wt_hor = [[0:chuwd/2-1] [chuwd/2-1:-1:0]] ./ (chuwd/2-1);
	% same vertically
	wt_vert = [[0:chuht/2-1] [chuht/2-1:-1:0]] ./ (chuht/2-1);

	% matrix of weights that goes from 0 at the edges to 1 in the center
	wt_mat = repelem(wt_hor, chuht, 1) .* repelem(wt_vert', 1, chuwd);

	% overlapping chunks
	for row = -chuht/2:chuht/2:fraht-1
		% indexing of Matlab arrays from 1 makes this complicated:
		% row and row_off are indexed form 0 while row_from and row_to are indexed from 1
		row_from = max(row+1, 1);
		row_off = row_from - (row+1);
		row_to = min(row + chuht, fraht);
		row_ht = row_to - row_from + 1;

		for col = -chuwd/2:chuwd/2:frawd-1
			% indexing of Matlab arrays from 1 makes this complicated:
			% col and col_off are indexed form 0 while col_from and col_to are indexed from 1
			col_from = max(col+1, 1);
			col_off = col_from - (col+1);
			col_to = min(col + chuwd, frawd);
			col_wd = col_to - col_from + 1;

			% the part of weights that fits into this chunk
			wt_chunk = wt_mat(row_off+1:row_off+row_ht, col_off+1:col_off+col_wd);

			mval = median(frame(row_from:row_to, col_from:col_to), "all");
			% for a test, hardcode mval = 1, then the resulting median must be 1
			med(row_from:row_to, col_from:col_to) = med(row_from:row_to, col_from:col_to) + wt_chunk .* mval;
		end
	end
			
	% compute the shift from the common median
	med = med - median(frame, "all");

	% make sure that we don't drive the minimum value down,
	% since it determines the noise amplitude
	minval = min(frame, [], "all");
	adj = frame - med;
	low = adj < minval;
	med = med .* ~low;
end
