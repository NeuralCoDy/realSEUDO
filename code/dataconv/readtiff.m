
% read the data from tiff file
%
% Args:
% filename - tiff file name
% first - first frame to load
% last - last frame to load
function m = readtiff(filename, first, last)
	img = single(imread(filename, 'Index', first));
	rows = size(img, 1);
	cols = size(img, 2);

	offset = first - 1;
	m = zeros(rows, cols, last - first + 1);
	m(:,:,first - offset) = img;

	for frame = first:last
		img = single(imread(filename, 'Index', frame));
		m(:,:,frame - offset) = img;
	end
end
